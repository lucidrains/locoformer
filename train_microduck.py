#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fire",
#     "imageio",
#     "imageio-ffmpeg",
#     "locoformer",
#     "mujoco",
#     "numpy",
#     "opencv-python",
#     "torch",
#     "tqdm",
# ]
# [tool.uv.sources]
# locoformer = { path = ".", editable = true }
# ///

from __future__ import annotations

import shutil
import subprocess
from collections import deque
from multiprocessing import Pipe, get_context
from pathlib import Path

import fire
import imageio
import mujoco
import numpy as np
import torch
from torch import nn
from torch.nn import Module
from torch.optim import Adam
from tqdm import tqdm

from locoformer.locoformer import Locoformer

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# ensure microduck_rl repository is available

MICRODUCK_REPO_URL = 'https://github.com/pollen-robotics/microduck_rl.git'
MICRODUCK_ASSET_SUBPATH = Path('src') / 'mjlab_microduck' / 'robot' / 'microduck' / 'scene_walk.xml'

def ensure_microduck_assets(repo_dir: str = 'microduck_rl') -> Path:
    for candidate in (Path(repo_dir), Path('..') / repo_dir):
        scene = candidate / MICRODUCK_ASSET_SUBPATH
        if scene.exists():
            return scene

    repo = Path(repo_dir)
    if not repo.exists():
        subprocess.run(['git', 'clone', '--depth', '1', MICRODUCK_REPO_URL, str(repo)], check = True)

    scene = repo / MICRODUCK_ASSET_SUBPATH
    assert scene.exists(), f'microduck assets not found at {scene}'
    return scene

MJCF = ensure_microduck_assets()

# shared interface constants

KP_MULT = 3.0
FORCED_SWING = 0.45
COMMAND = (0.15, 0.0)
ACTION_INTERFACE_RANGE = (-1.5, 1.5)

# microduck environment

NUM_FEET = 2
FEET = ('left', 'right')
JOINT_QPOS_OFFSET = 7
TRUNK_BODY = 1
WORLD_UP = np.array([0.0, 0.0, 1.0])
SPATIAL_DIM = 3

CPG_JOINT_NAMES = ('left_hip_pitch', 'left_knee', 'right_hip_pitch', 'right_knee')
CPG_CTRL_DIM = 4

class MicroduckEnv:
    def __init__(
        self,
        mjcf_path: Path | str | None = None,
        *,
        horizon: int = 400,
        seed: int = 42,
        control_hz: int = 50,
        n_substeps: int = 5,
        kp_mult: float = KP_MULT,
        kv: float = 0.1,
        cpg: bool = True,
        forced_swing: float | None = FORCED_SWING,
        fixed_command: tuple[float, float] | None = COMMAND,
        action_scale: float = 0.5,
        policy_range: tuple[float, float] = (-1.0, 1.0),
        cpg_freq: float = 2.0,
        cpg_swing_max: float = 0.5,
        cpg_knee_lag: float = np.pi / 2,
        track_weight: float = 2.0,
        track_std: tuple[float, float] = (0.02, 0.15),
        progress_weight: float = 5.0,
        upright_weight: float = 2.0,
        upright_std: float = 0.05,
        pose_weight: float = 1.0,
        action_rate_weight: float = -0.1,
        air_time_weight: float = 3.0,
        air_time_min: float = 0.10,
        air_time_max: float = 0.25,
        air_time_ramp_episodes: int = 960,
        nan_penalty: float = -10.0,
        floor_clip_penalty: float = -5.0,
        body_ang_vel_weight: float = -0.05,
        angular_momentum_weight: float = -0.02,
        foot_slip_weight: float = -0.1,
        foot_clearance_weight: float = -0.5,
        foot_clearance_target: float = 0.02,
        self_collision_weight: float = -1.0,
        foot_height_weight: float = 3.0,
        foot_height_clear: float = 0.01,
        foot_height_cap: float = 0.02,
        min_trunk_z: float = 0.06,
        min_uprightness: float = 0.5,
        standing_trunk_z: float = 0.095,
        standing_uprightness: float = 0.85,
        floor_clip_eps: float = 0.02,
        reset_z_range: tuple[float, float] = (0.0, 0.01),
        reset_joint_noise: float = 0.05,
        reset_vel_noise: float = 0.1,
    ):
        mjcf_path = Path(default(mjcf_path, MJCF))

        self.horizon = horizon
        self.cpg = cpg
        self.forced_swing = forced_swing
        self.fixed_command = np.asarray(fixed_command, dtype = np.float64) if exists(fixed_command) else None
        self.random_commands = not exists(self.fixed_command)

        self.action_scale = action_scale
        self.policy_range = policy_range
        self.control_hz, self.n_substeps = control_hz, n_substeps
        self.physics_dt = 1.0 / (control_hz * n_substeps)
        self.control_dt = self.physics_dt * n_substeps

        self.cpg_freq, self.cpg_swing_max, self.cpg_knee_lag = cpg_freq, cpg_swing_max, cpg_knee_lag

        self.track_weight, self.track_std, self.progress_weight = track_weight, track_std, progress_weight
        self.upright_weight, self.upright_std, self.pose_weight = upright_weight, upright_std, pose_weight
        self.action_rate_weight = action_rate_weight
        self.air_time_weight, self.air_time_min, self.air_time_max, self.air_time_ramp_episodes = air_time_weight, air_time_min, air_time_max, air_time_ramp_episodes
        self.nan_penalty, self.floor_clip_penalty = nan_penalty, floor_clip_penalty
        self.body_ang_vel_weight, self.angular_momentum_weight = body_ang_vel_weight, angular_momentum_weight
        self.foot_slip_weight, self.foot_clearance_weight, self.foot_clearance_target = foot_slip_weight, foot_clearance_weight, foot_clearance_target
        self.self_collision_weight = self_collision_weight
        self.foot_height_weight, self.foot_height_clear, self.foot_height_cap = foot_height_weight, foot_height_clear, foot_height_cap
        self.min_trunk_z, self.min_uprightness = min_trunk_z, min_uprightness
        self.standing_trunk_z, self.standing_uprightness = standing_trunk_z, standing_uprightness
        self.floor_clip_eps, self.reset_z_range = floor_clip_eps, reset_z_range
        self.reset_joint_noise, self.reset_vel_noise = reset_joint_noise, reset_vel_noise

        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.model.opt.timestep = self.physics_dt

        self.model.actuator_gainprm[:, 0] *= kp_mult
        self.model.actuator_biasprm[:, 1] = -self.model.actuator_gainprm[:, 0]
        self.model.actuator_biasprm[:, 2] = -kv

        self.data = mujoco.MjData(self.model)
        stand = self.model.key('STAND')
        self.home_qpos = stand.qpos.copy()
        self.home_ctrl = stand.ctrl.copy()

        self.action_dim = self.model.nu
        self.obs_dim = 2 * SPATIAL_DIM + 3 * self.action_dim + 2

        self.rng = np.random.default_rng(seed)
        self.last_action = np.zeros(self.action_dim)
        self.command = np.zeros(2)
        self.steps = 0
        self.episodes = 0
        self.foot_air = np.zeros(NUM_FEET)

        self.foot_geoms = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f'{side}_foot_collision')
            for side in FEET
        ])
        self.foot_bodies = np.array([self.model.geom_bodyid[g] for g in self.foot_geoms])
        self.non_foot_bodies = np.array([i for i in range(self.model.nbody) if i not in self.foot_bodies and i != 0])

        joint_id = lambda name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) - 1
        self.cpg_ctrl = np.array([joint_id(n) for n in CPG_JOINT_NAMES])
        self.direct_ctrl = np.setdiff1d(np.arange(self.action_dim), self.cpg_ctrl)

        self.pose_stds = np.array([
            0.30, 0.05, 0.40, 0.40, 0.25,
            0.35, 0.35, 0.35, 0.35,
            0.30, 0.05, 0.40, 0.40, 0.25,
        ])
        self.leg_mask = np.array([True] * 5 + [False] * 4 + [True] * 5)

    def velocity(self, local: bool = False) -> np.ndarray:
        v = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, TRUNK_BODY, v, flg_local = local)
        return v

    def observe(self, safe: bool = False) -> np.ndarray:
        if safe:
            return np.concatenate([
                np.zeros(6),
                np.zeros(self.action_dim * 2),
                self.last_action,
                self.command,
            ]).astype(np.float32)

        g = -self.data.xmat[TRUNK_BODY].reshape(SPATIAL_DIM, SPATIAL_DIM).T @ WORLD_UP

        return np.concatenate([
            g,
            self.velocity(local = True)[:SPATIAL_DIM],
            self.data.qpos[JOINT_QPOS_OFFSET:] - self.home_qpos[JOINT_QPOS_OFFSET:],
            self.data.qvel[6:],
            self.last_action,
            self.command,
        ]).astype(np.float32)

    def reset(self, seed: int | None = None) -> np.ndarray:
        if exists(seed):
            self.rng = np.random.default_rng(seed)

        self.data.qpos[:] = self.home_qpos
        self.data.qpos[2] += self.rng.uniform(*self.reset_z_range)
        self.data.qpos[JOINT_QPOS_OFFSET:] += self.rng.uniform(-self.reset_joint_noise, self.reset_joint_noise, size = self.action_dim)
        self.data.qvel[:] = self.rng.uniform(-self.reset_vel_noise, self.reset_vel_noise, size = self.model.nv)
        mujoco.mj_forward(self.model, self.data)

        if self.random_commands:
            self.command = np.zeros(2) if self.rng.uniform() < 0.25 else self.rng.uniform((-0.3, -0.5), (0.3, 0.5))
        else:
            self.command = self.fixed_command.copy()

        self.last_action[:] = 0.0
        self.steps = 0
        self.foot_air[:] = 0
        self.episodes += 1

        self.start_x = float(self.data.qpos[0])
        self.foot_rest_z = self.data.geom_xpos[self.foot_geoms][:, 2].copy()
        self.in_air = np.zeros(NUM_FEET, dtype = bool)
        self.air_time_seg = np.zeros(NUM_FEET)
        self.foot_peak_h = np.zeros(NUM_FEET)
        self.steps_in_air = 0

        return self.observe()

    def step(self, action: np.ndarray):
        action = np.nan_to_num(action, nan = 0.0, posinf = 0.0, neginf = 0.0)
        action_eff = np.clip(action, *self.policy_range)
        action_rate = float(np.mean((action_eff - np.clip(self.last_action, *self.policy_range)) ** 2))

        if self.cpg:
            cpg, direct = action[:CPG_CTRL_DIM], action[CPG_CTRL_DIM:]
            t = self.steps * self.control_dt

            swing = default(self.forced_swing, np.clip(cpg[0], 0.0, 1.0) * self.cpg_swing_max)
            phi_l, phi_r = cpg[2] * np.pi, cpg[3] * np.pi

            hip_l = swing * np.sin(2 * np.pi * self.cpg_freq * t + phi_l)
            knee_l = swing * np.sin(2 * np.pi * self.cpg_freq * t + phi_l - self.cpg_knee_lag)
            hip_r = swing * np.sin(2 * np.pi * self.cpg_freq * t + phi_r)
            knee_r = swing * np.sin(2 * np.pi * self.cpg_freq * t + phi_r - self.cpg_knee_lag)

            ctrl = self.home_ctrl.copy()
            ctrl[self.cpg_ctrl] += [hip_l, knee_l, hip_r, knee_r]
            ctrl[self.direct_ctrl] += self.action_scale * np.clip(direct, *self.policy_range)
        else:
            ctrl = self.home_ctrl + self.action_scale * action_eff

        self.data.ctrl[:] = ctrl

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        self.steps += 1
        self.last_action[:] = action

        if not (np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()):
            info = dict(
                x = float(self.data.qpos[0] - self.start_x) if np.isfinite(self.data.qpos[0]) else 0.0,
                steps_in_air = int(self.steps_in_air),
                foot_pk_cm = float(self.foot_peak_h.max() * 100.0) if np.isfinite(self.foot_peak_h.max()) else 0.0,
            )
            return self.observe(safe = True), self.nan_penalty, True, False, info

        q = self.data.qpos
        w, x, y, z = q[3:7]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        v_world = self.velocity(local = False)
        vx = np.cos(yaw) * v_world[SPATIAL_DIM] + np.sin(yaw) * v_world[SPATIAL_DIM + 1]
        wz = self.velocity(local = True)[2]

        uprightness = self.data.xmat[TRUNK_BODY][8]
        terminated = bool(uprightness < self.min_uprightness or q[2] < self.min_trunk_z)
        truncated = self.steps >= self.horizon

        body_z = self.data.xpos[self.non_foot_bodies, 2]
        floor_clip_depth = max(0.0, -float(body_z.min()))
        if floor_clip_depth > self.floor_clip_eps:
            terminated = True

        cmd_moving = abs(self.command[0]) > 0.01 or abs(self.command[1]) > 0.01

        r_track_lin = self.track_weight * np.exp(-((vx - self.command[0]) ** 2) / self.track_std[0])
        r_track_ang = self.track_weight * np.exp(-((wz - self.command[1]) ** 2) / self.track_std[1])
        r_upright = self.upright_weight * np.exp(-((1.0 - uprightness) ** 2) / self.upright_std)

        pose_scale = 1.5 if cmd_moving else 0.5
        joint_dev = (q[JOINT_QPOS_OFFSET:] - self.home_qpos[JOINT_QPOS_OFFSET:])[self.leg_mask]
        r_pose = self.pose_weight * np.exp(-((joint_dev / (self.pose_stds[self.leg_mask] * pose_scale)) ** 2).mean())
        r_action_rate = self.action_rate_weight * action_rate

        reward = r_track_lin + r_track_ang + r_upright + r_pose + r_action_rate

        if floor_clip_depth > 0.0:
            reward += self.floor_clip_penalty * floor_clip_depth * 100.0

        if abs(self.command[0]) > 0.01:
            reward += self.progress_weight * vx * np.sign(self.command[0])

        is_standing_gait = bool((uprightness >= self.standing_uprightness) and (q[2] >= self.standing_trunk_z) and (floor_clip_depth == 0.0))

        geom1, geom2 = self.data.contact.geom1, self.data.contact.geom2
        contacted = np.zeros(NUM_FEET, dtype = bool)
        for i, geom in enumerate(self.foot_geoms):
            contacted[i] = (geom1 == geom).any() or (geom2 == geom).any()

        foot_z = self.data.geom_xpos[self.foot_geoms][:, 2]

        for j in range(NUM_FEET):
            if contacted[j]:
                if self.in_air[j] and self.air_time_seg[j] > 0.04:
                    self.steps_in_air += 1
                self.in_air[j] = False
                self.air_time_seg[j] = 0.0
            else:
                self.in_air[j] = True
                self.air_time_seg[j] += self.control_dt
                self.foot_peak_h[j] = max(self.foot_peak_h[j], float(foot_z[j] - self.foot_rest_z[j]))

        if cmd_moving and is_standing_gait:
            self.foot_air[contacted] = 0
            self.foot_air[~contacted] += 1

            air_time = self.foot_air * self.control_dt
            ramp = min(1.0, self.episodes / self.air_time_ramp_episodes)
            reward += self.air_time_weight * ramp * ((air_time > self.air_time_min) & (air_time < self.air_time_max)).sum()
        else:
            self.foot_air[:] = 0

        if is_standing_gait:
            reward += self.foot_height_weight * np.clip((foot_z - self.foot_height_clear) / self.foot_height_cap, 0.0, 1.0).sum()

        ang_vel = self.velocity(local = True)[:SPATIAL_DIM]
        reward += self.body_ang_vel_weight * float(np.sum(ang_vel ** 2))

        i_comp = self.data.cinert[TRUNK_BODY][:6]
        inertia = np.array([
            [i_comp[0], i_comp[3], i_comp[4]],
            [i_comp[3], i_comp[1], i_comp[5]],
            [i_comp[4], i_comp[5], i_comp[2]],
        ])
        reward += self.angular_momentum_weight * float(np.sum((inertia @ ang_vel) ** 2))

        self_hits = sum(1 for i in range(self.data.ncon) if 0 not in (int(self.data.contact[i].geom1), int(self.data.contact[i].geom2)))
        reward += self.self_collision_weight * float(self_hits > 0)

        if cmd_moving:
            slip = 0.0
            for body in self.foot_bodies:
                v = np.zeros(6)
                mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, body, v, flg_local = 0)
                slip += v[SPATIAL_DIM] ** 2 + v[SPATIAL_DIM + 1] ** 2

            reward += self.foot_slip_weight * slip + self.foot_clearance_weight * np.clip((self.foot_clearance_target - foot_z) / self.foot_clearance_target, 0.0, 1.0).mean()

        info = dict(
            x = float(q[0] - self.start_x),
            steps_in_air = int(self.steps_in_air),
            foot_pk_cm = float(self.foot_peak_h.max() * 100.0),
        )
        return self.observe(), float(reward), terminated, truncated, info

    def close(self):
        pass

# parallel environment workers

def _create_env(mjcf_path: Path | str, *, seed: int, horizon: int = 400, forced_swing: float = FORCED_SWING):
    return MicroduckEnv(mjcf_path, horizon = horizon, seed = seed, forced_swing = forced_swing)

def _worker_main(conn, seed, mjcf_path, forced_swing):
    env = _create_env(mjcf_path, seed = seed, forced_swing = forced_swing)

    while True:
        try:
            cmd, payload = conn.recv()
        except EOFError:
            break

        if cmd == 'reset':
            conn.send(env.reset(payload))
        elif cmd == 'step':
            conn.send(env.step(payload))
        elif cmd == 'close':
            conn.close()
            break

class VecEnv:
    def __init__(
        self,
        num_envs: int = 12,
        seed: int = 0,
        mjcf_path: Path | str | None = None,
        forced_swing: float = FORCED_SWING,
        horizon: int = 400,
    ):
        mjcf_path = Path(default(mjcf_path, MJCF))
        ctx = get_context('spawn')
        self._conns, self._procs = [], []

        for i in range(num_envs):
            parent, child = ctx.Pipe()
            proc = ctx.Process(target = _worker_main, args = (child, seed + i * 1000, str(mjcf_path), forced_swing), daemon = True)
            proc.start()
            child.close()
            self._conns.append(parent)
            self._procs.append(proc)

        self.num_envs = num_envs
        self.forced_swing = forced_swing

        probe = MicroduckEnv(mjcf_path, seed = seed, forced_swing = forced_swing)
        self.obs_dim, self.action_dim = probe.obs_dim, probe.action_dim
        probe.close()

    def reset(self):
        for conn in self._conns:
            conn.send(('reset', None))
        return np.stack([conn.recv() for conn in self._conns])

    def step(self, actions: np.ndarray):
        for conn, action in zip(self._conns, np.asarray(actions)):
            conn.send(('step', action))

        results = [conn.recv() for conn in self._conns]
        infos = [res[4] for res in results]
        resetting = [i for i, res in enumerate(results) if res[2] or res[3]]

        for i in resetting:
            self._conns[i].send(('reset', None))
        for i in resetting:
            results[i] = (self._conns[i].recv(), results[i][1], results[i][2], results[i][3], infos[i])

        return (
            np.stack([res[0] for res in results]),
            np.array([res[1] for res in results], dtype = np.float32),
            np.array([res[2] for res in results], dtype = bool),
            np.array([res[3] for res in results], dtype = bool),
            infos,
        )

    def close(self):
        for conn in self._conns:
            try:
                conn.send(('close', None))
            except (BrokenPipeError, OSError):
                pass
        for proc in self._procs:
            proc.join(timeout = 2)
            if proc.is_alive():
                proc.terminate()

# evaluation & rendering

class GreedyPolicy(Module):
    def __init__(self, agent):
        super().__init__()
        self.agent = agent
        self._stateful_forward = None

    def reset(self):
        self._stateful_forward = None

    @torch.no_grad()
    def forward(self, x):
        if self._stateful_forward is None:
            self._stateful_forward = self.agent.get_stateful_forward(has_batch_dim = True, has_time_dim = False, inference_mode = True)
        logits = self._stateful_forward(x, state_embed_kwargs = dict(state_type = 'raw'))
        dist = self.agent.unembedder.continuous_dist.dist(logits)
        return self.agent.unembedder.rescale_from_native(dist.mean, ACTION_INTERFACE_RANGE)

@torch.no_grad()
def evaluate_gait(agent, seeds = (0, 1, 2), horizon = 400, mjcf_path: Path | str | None = None, forced_swing: float = FORCED_SWING):
    agent.eval()
    mjcf_path = Path(default(mjcf_path, MJCF))
    policy = GreedyPolicy(agent) if not isinstance(agent, GreedyPolicy) else agent
    outs = []

    for seed in seeds:
        env = MicroduckEnv(mjcf_path, seed = seed, horizon = horizon, forced_swing = forced_swing)
        policy.reset()
        obs = env.reset(seed)
        total, steps = 0.0, 0
        last_info = dict(x = 0.0, steps_in_air = 0, foot_pk_cm = 0.0)

        for _ in range(horizon):
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            obs, r, term, trunc, info = env.step(policy(obs_t).squeeze(0).cpu().numpy())
            total += float(r)
            steps += 1
            last_info = info
            if term or trunc:
                break

        outs.append((steps, last_info['x'], total, last_info['steps_in_air'], last_info['foot_pk_cm']))
        env.close()

    agent.train()
    return tuple(float(np.mean([o[i] for o in outs])) for i in range(5))

@torch.no_grad()
def render_gait(
    policy,
    mjcf_path: Path | str | None = None,
    *,
    horizon: int = 400,
    out_path: str = 'videos/microduck_gait.mp4',
    forced_swing: float = FORCED_SWING,
    seed: int = 42,
    total_steps: int | None = None,
):
    import cv2
    mjcf_path = Path(default(mjcf_path, MJCF))
    env = MicroduckEnv(mjcf_path, seed = seed, horizon = horizon, forced_swing = forced_swing)

    renderer = mujoco.Renderer(env.model, height = 480, width = 640)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (0.3, 0.0, 0.15)
    camera.distance = 1.0
    camera.azimuth = 150.0
    camera.elevation = -15.0

    if hasattr(policy, 'reset'):
        policy.reset()

    obs = env.reset(seed)
    frames = []

    for step in tqdm(range(horizon), desc = 'rendering', leave = False):
        obs_t = torch.from_numpy(obs).float().unsqueeze(0)
        obs, _, term, trunc, info = env.step(policy(obs_t).squeeze(0).cpu().numpy())

        renderer.update_scene(env.data, camera = camera)
        frame = renderer.render()

        if exists(total_steps):
            text = f'{total_steps / 1e6:.2f}M steps | step {step + 1} | x: {info["x"]:+.2f}m | air: {info["steps_in_air"]}'
            cv2.putText(frame, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        frames.append(frame)
        if term or trunc:
            break

    out = Path(out_path)
    out.parent.mkdir(parents = True, exist_ok = True)
    imageio.mimsave(out, frames, fps = 50)

    renderer.close()
    env.close()

# training

def train(
    mjcf_path: str = str(MJCF),
    save_path: str = 'checkpoints-microduck/locoformer_microduck.pt',
    total_env_steps: int = 10_000_000,
    num_updates: int | None = None,
    num_envs: int = 12,
    steps_per_env: int = 256,
    epochs: int = 4,
    learning_rate: float = 5e-4,
    dim: int = 64,
    depth: int = 1,
    heads: int = 4,
    dim_head: int = 32,
    window_size: int = 16,
    forced_swing: float = FORCED_SWING,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip: float = 0.2,
    ent_coef: float = 0.002,
    eval_every: int = 20,
    eval_video: bool = True,
    video_dir: str = 'videos',
    clear_videos: bool = False,
    seed: int = 42,
    resume: bool = True,
    smoke: bool = False,
):
    if smoke:
        total_env_steps, num_updates, num_envs, steps_per_env, epochs, eval_every = 200, 2, 2, 32, 1, 1

    steps_per_update = num_envs * steps_per_env
    num_updates = default(num_updates, int(np.ceil(total_env_steps / steps_per_update)))

    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    resolved_mjcf = Path(mjcf_path).resolve()
    assert resolved_mjcf.exists(), f'MJCF not found at {resolved_mjcf}'

    resolved_save_path = Path(save_path).resolve()
    resolved_save_path.parent.mkdir(parents = True, exist_ok = True)
    vdir = Path(video_dir)
    vdir.mkdir(parents = True, exist_ok = True)

    if clear_videos:
        for f in vdir.glob('*.mp4'):
            f.unlink()

    print(f'[Locoformer] Microduck training ({total_env_steps:,} max steps, {num_updates} updates, {num_envs} envs, forced_swing = {forced_swing})')

    venv = VecEnv(num_envs = num_envs, seed = seed, mjcf_path = resolved_mjcf, forced_swing = forced_swing)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Locoformer(
        embedder = dict(dim = dim, dim_state = venv.obs_dim),
        unembedder = dict(
            dim = dim,
            num_continuous = venv.action_dim,
            continuous_dist_type = 'beta',
            continuous_dist_kwargs = dict(unimodal = True),
            selectors = [list(range(venv.action_dim))]
        ),
        transformer = dict(dim = dim, dim_head = dim_head, heads = heads, depth = depth, window_size = window_size),
        policy_network = nn.Sequential(nn.Linear(dim, dim), nn.SiLU()),
        value_network = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim), nn.SiLU()),
        dim_value_input = dim,
        reward_range = (-100.0, 2000.0),
        action_rescale_ranges = [ACTION_INTERFACE_RANGE],
        discount_factor = gamma,
        gae_lam = gae_lambda,
        ppo_eps_clip = clip,
        ppo_entropy_weight = ent_coef,
    ).to(device)

    optim = Adam(model.parameters(), lr = learning_rate)

    best_gait_score = -float('inf')
    best_eval_steps, best_eval_x, best_eval_air = 0.0, -float('inf'), 0.0
    start_update, total_steps = 1, 0

    if resume and resolved_save_path.exists():
        try:
            ckpt = torch.load(resolved_save_path, map_location = device)
            model.load_state_dict(ckpt['model'])
            best_gait_score = ckpt.get('best_gait_score', -float('inf'))
            best_eval_steps, best_eval_x, best_eval_air = ckpt.get('best_eval_steps', 0.0), ckpt.get('best_eval_x', -float('inf')), ckpt.get('best_eval_air', 0.0)
            start_update = ckpt.get('update', 0) + 1
            total_steps = ckpt.get('total_steps', (start_update - 1) * steps_per_update)
            print(f'[Locoformer] Resumed from {resolved_save_path} (update {start_update - 1}, steps {total_steps:,}, best_x = {best_eval_x:+.2f}m, best_air = {best_eval_air:.0f})')
        except Exception as e:
            print(f'[Locoformer] Warning: failed to resume checkpoint: {e}')

    obs = venv.reset()
    ep_returns, ep_lens, ep_x, ep_air = deque(maxlen = 100), deque(maxlen = 100), deque(maxlen = 100), deque(maxlen = 100)
    cur_ep_ret, cur_ep_len = np.zeros(num_envs), np.zeros(num_envs)
    ep_id = torch.zeros(num_envs, dtype = torch.long, device = device)

    pbar = tqdm(range(start_update, num_updates + 1), desc = 'Locoformer Microduck', dynamic_ncols = True)

    try:
        for update in pbar:
            obs_t = torch.from_numpy(obs).float().to(device)
            stateful_fwd = model.get_stateful_forward(has_batch_dim = True, has_time_dim = False, inference_mode = True)
            buf = dict(obs = [], act = [], logp = [], rew = [], val = [], done = [], ep_id = [])

            for _ in range(steps_per_env):
                with torch.no_grad():
                    logits, val = stateful_fwd(obs_t, episode_id = ep_id, return_values = True, state_embed_kwargs = dict(state_type = 'raw'))
                    action = model.unembedder.sample(logits)
                    logp = model.unembedder.log_prob(logits, action, concat = True)
                    env_action = model.unembedder.rescale_from_native(action, ACTION_INTERFACE_RANGE)

                next_obs, rew, term, trunc, infos = venv.step(env_action.cpu().numpy())
                done = term | trunc

                buf['obs'].append(obs_t)
                buf['act'].append(action)
                buf['logp'].append(logp)
                buf['rew'].append(torch.from_numpy(rew).to(device))
                buf['val'].append(val)
                buf['done'].append(torch.from_numpy(done).to(device))
                buf['ep_id'].append(ep_id.clone())

                cur_ep_ret += rew
                cur_ep_len += 1

                for i in np.where(done)[0]:
                    ep_returns.append(float(cur_ep_ret[i]))
                    ep_lens.append(float(cur_ep_len[i]))
                    if i < len(infos) and 'x' in infos[i]:
                        ep_x.append(float(infos[i]['x']))
                        ep_air.append(float(infos[i]['steps_in_air']))
                    ep_id[i] += 1

                cur_ep_ret[done] = 0.0
                cur_ep_len[done] = 0
                obs = next_obs
                obs_t = torch.from_numpy(obs).float().to(device)

            total_steps += steps_per_env * num_envs

            with torch.no_grad():
                _, last_val = stateful_fwd(obs_t, episode_id = ep_id, return_values = True, state_embed_kwargs = dict(state_type = 'raw'))

            states = torch.stack(buf['obs'], dim = 1)
            actions = torch.stack(buf['act'], dim = 1)
            logps = torch.stack(buf['logp'], dim = 1)
            rews = torch.stack(buf['rew'], dim = 1)
            vals = torch.stack(buf['val'], dim = 1)
            dones = torch.stack(buf['done'], dim = 1)
            ep_indices = torch.stack(buf['ep_id'], dim = 1)

            rews[:, -1] += gamma * last_val * (~dones[:, -1])
            lens = torch.full((num_envs,), steps_per_env, dtype = torch.long, device = device)

            for _ in range(epochs):
                model.ppo(
                    state = states,
                    internal_state = None,
                    action = actions,
                    action_log_prob = logps,
                    reward = rews,
                    value = vals,
                    done = dones,
                    episode_lens = lens,
                    optims = [optim],
                    state_embed_kwargs = dict(state_type = 'raw'),
                    action_select_kwargs = dict(selector_index = 0),
                    compute_state_pred_loss = False,
                    episode_indices = ep_indices,
                )

            eval_metrics = None
            if update % eval_every == 0 or update == num_updates or total_steps >= total_env_steps:
                eval_seeds = (0,) if smoke else (0, 1, 2)
                eval_horizon = 20 if smoke else 400

                eval_steps, eval_x, eval_ret, eval_air_steps, eval_foot_pk = evaluate_gait(
                    model, seeds = eval_seeds, horizon = eval_horizon, mjcf_path = resolved_mjcf, forced_swing = forced_swing
                )
                eval_metrics = (eval_steps, eval_x, eval_air_steps, eval_foot_pk)

                gait_score = eval_x * 50.0 + eval_air_steps * 5.0 + eval_steps * 0.5
                is_best = gait_score > best_gait_score or (eval_x > best_eval_x and eval_steps >= 20)

                pbar.write(
                    f'[Eval u{update:04d}] steps: {eval_steps:3.0f} | x: {eval_x:+.2f} m | '
                    f'air-steps: {eval_air_steps:2.0f} | foot-pk: {eval_foot_pk:3.1f} cm | ret: {eval_ret:5.0f}'
                    + (' [BEST]' if is_best else '')
                )

                if is_best:
                    best_gait_score, best_eval_steps, best_eval_x, best_eval_air = gait_score, eval_steps, eval_x, eval_air_steps
                    torch.save(dict(model = model.state_dict(), update = update, total_steps = total_steps, best_gait_score = best_gait_score, best_eval_steps = best_eval_steps, best_eval_x = best_eval_x, best_eval_air = best_eval_air), resolved_save_path)

                if eval_video:
                    vid_name = f'eval_{total_steps / 1e6:.2f}M_u{update:04d}_st{eval_steps:03.0f}_x{eval_x:+.2f}m_air{eval_air_steps:02.0f}.mp4'
                    vid_path = vdir / vid_name
                    try:
                        render_gait(GreedyPolicy(model), mjcf_path = resolved_mjcf, horizon = eval_horizon, out_path = str(vid_path), forced_swing = forced_swing, seed = eval_seeds[0], total_steps = total_steps)
                        if is_best:
                            shutil.copyfile(vid_path, vdir / 'microduck_best_gait.mp4')
                    except Exception as e:
                        print(f'\n[video failed] {e}')

            mean = lambda xs: float(np.mean(xs)) if len(xs) > 0 else 0.0
            postfix = dict(
                steps = total_steps,
                train_x = f'{mean(ep_x):+.2f}m',
                train_air = f'{mean(ep_air):.1f}',
                train_len = f'{mean(ep_lens):.0f}',
                train_r = f'{mean(ep_returns):.0f}',
                best_x = f'{best_eval_x:+.2f}m',
            )
            if eval_metrics is not None:
                postfix.update(eval_x = f'{eval_metrics[1]:+.2f}m', eval_air = f'{eval_metrics[2]:.0f}', eval_st = f'{eval_metrics[0]:.0f}')
            pbar.set_postfix(**postfix)

            if total_steps >= total_env_steps:
                break

    except KeyboardInterrupt:
        print('\n[Locoformer] Training interrupted.')
    finally:
        torch.save(dict(model = model.state_dict(), total_steps = total_steps), resolved_save_path)
        venv.close()
        pbar.close()
        print(f'\n[Locoformer] Finished at {total_steps:,} steps. Saved to {resolved_save_path}')

if __name__ == '__main__':
    fire.Fire(train)
