import pytest
param = pytest.mark.parametrize

import torch
from torch import nn
from x_mlps_pytorch import MLP

from einops import rearrange, repeat
from torch_einops_utils.shape import size

from locoformer.locoformer import Locoformer, exists

@param('gru_layers', (False, True))
@param('recurrent_cache', (False, True))
@param('has_commands', (False, True))
@param('long_term_mem_layers', ((), (1, 2)))
@param('hyper_conn', (False, True))
def test_locoformer(
    gru_layers,
    recurrent_cache,
    has_commands,
    long_term_mem_layers,
    hyper_conn
):

    model = Locoformer(
        embedder = nn.Embedding(256, 128),
        unembedder = nn.Linear(128, 256, bias = False),
        value_network = MLP(128, 64, 32),
        dim_value_input = 32,
        reward_range = (-100., 100.),
        recurrent_cache = recurrent_cache,
        transformer = dict(
            dim = 128,
            depth = 2,
            window_size = 32,
            gru_layers = gru_layers,
            dim_cond = 2 if has_commands else None,
            long_term_mem_layers = long_term_mem_layers,
            num_residual_streams = 2 if hyper_conn else 1
        )
    )

    seq = torch.randint(0, 256, (3, 32))

    commands = None
    if has_commands:
        commands = torch.randn(3, 32, 2)

    (logits, values), cache = model(seq, condition = commands, return_values = True)
    (logits, values), cache = model(seq, condition = commands, return_values = True, cache = cache)

    assert logits.shape == (3, 32, 256)

    stateful_forward = model.get_stateful_forward(has_batch_dim = True, has_time_dim = True, return_values = True, inference_mode = True)

    inference_command = torch.randn(1, 1, 2) if has_commands else None

    for state in seq.unbind(dim = -1)[:8]:
        state = rearrange(state, 'b -> b 1')

        logits, values = stateful_forward(state, condition = inference_command)
        assert logits.shape == (3, 1, 256)

def test_replay():
    from memmap_replay_buffer import ReplayBuffer

    replay_buffer = ReplayBuffer(
        './replay_data',
        max_episodes = 10_000,
        max_timesteps = 501,
        fields = dict(
            state = ('float', (8,)),
            action = 'int',
            action_log_prob = 'float',
            reward = 'float',
            value = 'float',
            done = 'bool'
        )
    )

    lens = [3, 5, 4]

    for episode_len in lens:
        with replay_buffer.one_episode():
            for _ in range(episode_len):
                state = torch.randn((8,))
                action = torch.randint(0, 4, ())
                log_prob = torch.randn(())
                reward = torch.randn(())
                value = torch.randn(())
                done = torch.randint(0, 2, ()).bool()

                replay_buffer.store(
                    state = state,
                    action = action,
                    action_log_prob = log_prob,
                    reward = reward,
                    value = value,
                    done = done
                )

    dataset = replay_buffer.dataset()

    assert len(dataset) == 3

    assert torch.is_tensor(dataset[0]['state'])

    dataloader = replay_buffer.dataloader(batch_size = 3)

    assert size(next(iter(dataloader))['state'], '[b] ...') == 3

    # we will now consider consecutive pairs of episodes as 2 trials to be used for in-context adaptation
    # but realistically there will be a function that converts a given ReplayBuffer -> Int[batch, episode_indices]

    from torch import stack, arange

    episode_indices = arange(len(replay_buffer))
    remapped_episodes = stack((episode_indices[:-1], episode_indices[1:]))

    dataset = replay_buffer.dataset()

    from locoformer.locoformer import RemappedReplayDataset

    dataset = RemappedReplayDataset(dataset, remapped_episodes)

    dataloader = replay_buffer.dataloader(
        batch_size = 1,
        dataset = dataset
    )

    assert next(iter(dataloader))['_lens'][0] == (3 + 5) # first and second episodes are concatted together timewise

def test_reward_shaping():

    model = Locoformer(
        embedder = nn.Embedding(256, 128),
        unembedder = nn.Linear(128, 256, bias = False),
        value_network = MLP(128, 64, 32),
        dim_value_input = 32,
        reward_range = (-100., 100.),
        reward_shaping_fns = [
            lambda state: (state[3] - 2.5).pow(2).mean(),
            lambda state, command: state[4:6].norm(dim = -1)
        ],
        transformer = dict(
            dim = 128,
            depth = 1,
            window_size = 512
        )
    )

    import numpy as np

    class MockEnv:
        def reset(self):
            return np.random.normal(size = (10,)), {}

        def step(self, *args, **kwargs):
            return np.random.normal(size = (10,)), 0., False, False, {}


    env = MockEnv()

    reset_fn, step_fn = model.wrap_env_functions(env)

    reset_fn()

    step_dict = step_fn(3)

    assert len(step_dict['shaped_rewards']) == 2

def test_tensor_to_dict():
    state = torch.randn(1, 3, 5)
    config = (('xyz', 3), 'vx', 'vy')

    from locoformer.locoformer import tensor_to_dict

    state_dict = tensor_to_dict(state, config)
    assert hasattr(state_dict, 'xyz') and state_dict.xyz.shape == (1, 3, 3)

def test_evo():

    model = Locoformer(
        embedder = nn.Embedding(256, 128),
        unembedder = nn.Linear(128, 256, bias = False),
        value_network = MLP(128, 64, 32),
        dim_value_input = 32,
        reward_range = (-100., 100.),
        transformer = dict(
            dim = 128,
            depth = 1,
            window_size = 512,
        )
    )

    model.evolve(lambda model: 1., num_generations = 1)

def test_unified_state():
    from torch.nn import Module, ModuleList
    from locoformer.locoformer import Locoformer

    class StateEmbed(Module):
        def __init__(self):
            super().__init__()
            self.embedders = ModuleList([
                nn.Embedding(256, 128),
                nn.Linear(2, 128)
            ])

        def forward(self, state, state_type):
            return self.embedders[state_type](state)

    model = Locoformer(
        embedder = StateEmbed(),
        unembedder = nn.Linear(128, 256, bias = False),
        value_network = MLP(128, 64, 32),
        dim_value_input = 32,
        reward_range = (-100., 100.),
        recurrent_cache = False,
        transformer = dict(
            dim = 128,
            depth = 1,
            window_size = 512,
        )
    )

    state1 = torch.randint(0, 256, (3, 512))
    state2 = torch.randn((3, 512, 2))

    logits, cache = model(state1, state_embed_kwargs = dict(state_type = 0))
    logits, cache = model(state2, state_embed_kwargs = dict(state_type = 1), cache = cache)
    logits, cache = model(state1, state_embed_kwargs = dict(state_type = 0), cache = cache)

def test_memory():
    from locoformer.locoformer import MemoryMLP

    memory = MemoryMLP(512)

    tokens = torch.randn(2, 32, 512)

    memories = None

    retrieved = memory(tokens, memories)

    tokens = tokens + retrieved

    memories = memory.store(tokens, memories)

    retrieved = memory(tokens, memories)

    tokens = tokens + retrieved

    memories = memory.store(tokens, memories)

    assert tokens.shape == (2, 32, 512)

@param('recurrent_cache', (False, True))
@param('gru_layers', (False, True))
def test_locoformer_multi_segment(recurrent_cache, gru_layers):
    model = Locoformer(
        embedder = nn.Embedding(256, 128),
        unembedder = nn.Linear(128, 256, bias = False),
        max_mem_segments = 2,
        recurrent_cache = recurrent_cache,
        transformer = dict(
            dim = 128,
            depth = 1,
            window_size = 16,
            gru_layers = gru_layers
        )
    ).eval()

    batch, window_size, num_segments = 3, 16, 4
    seq = torch.randint(0, 256, (batch, window_size * num_segments))
    ep_id = repeat(torch.arange(batch), 'b -> b n', n = window_size * num_segments)

    logits_full = []
    cache = None

    for segment, segment_ep_id in zip(seq.chunk(num_segments, dim = -1), ep_id.chunk(num_segments, dim = -1)):
        logits, cache = model(segment, episode_id = segment_ep_id, cache = cache)
        logits_full.append(logits)

        # recurrent cache rolls the layer axis, never the batch-aligned episode ids

        assert torch.equal(cache.curr_episode_ids, segment_ep_id)

    logits_full = torch.cat(logits_full, dim = 1)

    stateful_forward = model.get_stateful_forward(has_batch_dim = True, has_time_dim = True, inference_mode = True)

    logits_stateful = []

    for step_seq, step_ep_id in zip(seq.unbind(dim = -1), ep_id.unbind(dim = -1)):
        step_seq = rearrange(step_seq, 'b -> b 1')
        step_ep_id = rearrange(step_ep_id, 'b -> b 1')
        logits = stateful_forward(step_seq, episode_id = step_ep_id)
        logits_stateful.append(logits)

    logits_stateful = torch.cat(logits_stateful, dim = 1)

    assert torch.allclose(logits_full, logits_stateful, atol = 1e-5)

def test_locoformer_episode_id():
    dim, window_size = 128, 8
    model = Locoformer(
        embedder = nn.Embedding(256, dim), unembedder = nn.Linear(dim, 256),
        transformer = dict(dim = dim, depth = 1, window_size = window_size)
    )

    # basic and consistency
    seq, ep_id = torch.randint(0, 256, (1, 8)), torch.zeros((1, 8), dtype = torch.long)
    _, cache = model(seq, episode_id = ep_id)
    with pytest.raises(AssertionError):
        model(seq, cache = cache) # missing episode_id

    # isolation & stateful forward
    model = Locoformer(
        embedder = nn.Linear(window_size, dim, bias = False),
        unembedder = dict(num_continuous = 1),
        transformer = dict(dim = dim, depth = 1, window_size = 4, heads = 1),
        embed_past_action = True,
        num_latent_genes = 4,
        dim_latent = 16,
        latent_gene_pool_kwargs = dict(
            frac_tournaments = 0.75,
            frac_natural_selected = 0.75,
            frac_elitism = 0.
        )
    ).eval()

    # derived conditioning modules must be included in the base optimizer group

    base_param_ids = {id(p) for p in model.base_parameters()}
    assert all(id(p) in base_param_ids for p in model.past_action_embedder.parameters())
    assert all(id(p) in base_param_ids for p in model.to_latent_cond.parameters())

    win1, win2 = torch.randn(1, 4, window_size), torch.randn(1, 4, window_size)
    ep0, ep1 = torch.zeros((1, 4), dtype = torch.long), torch.ones((1, 4), dtype = torch.long)

    _, cache = model(win1, episode_id = ep0, detach_cache = True)
    assert not cache.kv_cache.requires_grad

    out_diff_ep, _ = model(win2, episode_id = ep1, cache = cache)
    out_clean, _ = model(win2, episode_id = ep1)

    assert torch.allclose(out_diff_ep, out_clean, atol = 1e-5)

    stateful_forward = model.get_stateful_forward(has_batch_dim = True, inference_mode = True)
    for step in win2.unbind(dim = 1):
        out = stateful_forward(step, episode_id = torch.ones((1,), dtype = torch.long))
        assert out.shape[:2] == (1, 1)

@param('use_cross_episode_attention', (False, True))
def test_locoformer_cross_episode_attention(use_cross_episode_attention):

    dim, window_size = 64, 4

    model = Locoformer(
        embedder = dict(dim = dim, dim_state = 4),
        unembedder = dict(
            dim = dim,
            num_discrete = 2,
        ),
        transformer = dict(
            dim = dim,
            dim_head = 32,
            heads = 1,
            depth = 1,
            window_size = window_size,
        ),
        actor_depth = 1,
        use_cross_episode_attention = use_cross_episode_attention,
        policy_network = nn.Identity(),
        value_network = nn.Identity(),
        dim_value_input = dim,
        reward_range = (-1., 1.),
        state_pred_network = nn.Identity()
    ).eval()

    win1, win2 = torch.randn(1, 4, 4), torch.randn(1, 4, 4)
    ep0, ep1 = torch.zeros((1, 4), dtype = torch.long), torch.ones((1, 4), dtype = torch.long)

    # state embeddings must remain sensitive to state magnitude after transformer RMSNorm

    far = model.embedder(win1, state_type = 'raw')
    near = model.embedder(win1 * 0.1, state_type = 'raw')
    unit = lambda t: t / t.norm(dim = -1, keepdim = True)

    assert not torch.allclose(unit(far), unit(near), atol = 1e-3)

    forward_kwargs = dict(
        state_embed_kwargs = dict(state_type = 'raw'),
        action_select_kwargs = dict(selector_index = 0),
        state_id_kwarg = dict(state_id = 0),
        return_values = True,
        return_state_pred = True
    )

    _, cache = model(win1, episode_id = ep0, **forward_kwargs)
    ((act_diff, state_pred_diff), value_diff), _ = model(win2, episode_id = ep1, cache = cache, **forward_kwargs)
    ((act_clean, state_pred_clean), value_clean), _ = model(win2, episode_id = ep1, **forward_kwargs)

    act_diff, = act_diff # discrete logits are returned as a 1-tuple
    act_clean, = act_clean

    # policy attention transformer may be turned on / off for cross-episode attention (in-context adaptation, as in the paper)

    if use_cross_episode_attention:
        assert not torch.allclose(act_diff, act_clean, atol = 1e-5)
    else:
        assert torch.allclose(act_diff, act_clean, atol = 1e-5)

    # value and world model heads are always masked across episodes

    assert torch.allclose(value_diff, value_clean, atol = 1e-5)
    assert torch.allclose(state_pred_diff, state_pred_clean, atol = 1e-5)

def test_locoformer_cross_episode_attention_requires_actor():
    with pytest.raises(ValueError):
        Locoformer(
            embedder = nn.Linear(4, 64, bias = False),
            unembedder = nn.Linear(64, 1),
            transformer = dict(dim = 64, depth = 1, window_size = 4),
            use_cross_episode_attention = True
        )

def test_reward_shaping_validation():
    # should pass
    Locoformer(
        embedder = nn.Embedding(256, 128),
        unembedder = nn.Linear(128, 256),
        transformer = dict(dim = 128, depth = 1, window_size = 8),
        reward_shaping_fns = [lambda s: 1.]
    )

    # should pass (2d with string path)
    Locoformer(
        embedder = nn.Embedding(256, 128),
        unembedder = nn.Linear(128, 256),
        transformer = dict(dim = 128, depth = 1, window_size = 8),
        reward_shaping_fns = [(lambda s: 1., 'reward')]
    )

    # should pass (2d with 0 index)
    Locoformer(
        embedder = nn.Embedding(256, 128),
        unembedder = nn.Linear(128, 256),
        transformer = dict(dim = 128, depth = 1, window_size = 8),
        reward_shaping_fns = [(lambda s: 1., ('reward', 0))]
    )

    # should fail (2d with non-0 index)
    with pytest.raises(AssertionError):
        Locoformer(
            embedder = nn.Embedding(256, 128),
            unembedder = nn.Linear(128, 256),
            transformer = dict(dim = 128, depth = 1, window_size = 8),
            reward_shaping_fns = [(lambda s: 1., ('reward', 1))]
        )

    # should fail (3d without store field)
    with pytest.raises(AssertionError):
        Locoformer(
            embedder = nn.Embedding(256, 128),
            unembedder = nn.Linear(128, 256),
            transformer = dict(dim = 128, depth = 1, window_size = 8),
            reward_shaping_fns = [[lambda s: 1.]]
        )

    # should pass (3d with store field)
    Locoformer(
        embedder = nn.Embedding(256, 128),
        unembedder = nn.Linear(128, 256),
        transformer = dict(dim = 128, depth = 1, window_size = 8),
        reward_shaping_fns = [[(lambda s: 1., ('reward', 1))]]
    )

def test_reward_shaping_storage():
    from memmap_replay_buffer import ReplayBuffer
    import numpy as np

    model = Locoformer(
        embedder = nn.Linear(10, 128),
        unembedder = nn.Linear(128, 10),
        transformer = dict(dim = 128, depth = 1, window_size = 8),
        reward_shaping_fns = [[
            (lambda state: 1.0, 'scalar_reward'),
            (lambda state: 2.0, ('vector_reward', 1))
        ]]
    )

    replay = ReplayBuffer(
        './replay_test_storage',
        max_episodes = 1,
        max_timesteps = 10,
        fields = dict(
            state = ('float', (10,)),
            action = ('float', (10,)),
            scalar_reward = 'float',
            vector_reward = ('float', (2,)),
            reward = 'float', # needed for wrap_env_functions
        )
    )

    # mock env output
    state = torch.randn(10)

    with replay.one_episode():
        model.state_and_command_to_rewards(state, replay_buffer = replay, env_index = 0)

    assert replay.data['scalar_reward'][0, 0] == 1.0
    assert replay.data['vector_reward'][0, 0, 1] == 2.0
    assert replay.data['vector_reward'][0, 0, 0] == 0.0

    # cleanup
    import shutil
    shutil.rmtree('./replay_test_storage', ignore_errors = True)

def test_computed_reward_shaping_input_fns():
    from locoformer.locoformer import tensor_to_dict

    OBS_CONFIG = (('vx', 1), ('vy', 1), ('rest', 8))

    model = Locoformer(
        embedder = nn.Linear(10, 128),
        unembedder = nn.Linear(128, 10),
        transformer = dict(dim = 128, depth = 1, window_size = 8),
        state_named_config = OBS_CONFIG,
        reward_shaping_fns = [
            lambda state_named: state_named.vx + state_named.vy
        ]
    )

    state = torch.randn(10)
    rewards = model.state_and_command_to_rewards(state)

    assert rewards.shape == (1,)
    assert torch.allclose(rewards, state[0] + state[1])

def test_epo():
    import numpy as np
    import shutil
    from memmap_replay_buffer import ReplayBuffer

    shutil.rmtree('./replay_test_epo', ignore_errors = True)

    model = Locoformer(
        embedder = nn.Linear(10, 128),
        unembedder = dict(
            num_continuous = 10
        ),
        value_network = MLP(128, 64, 32),
        dim_value_input = 32,
        reward_range = (-100., 100.),
        num_latent_genes = 4,
        dim_latent = 16,
        latent_gene_pool_kwargs = dict(
            frac_tournaments = 0.75,
            frac_natural_selected = 0.75,
            frac_elitism = 0.
        ),
        transformer = dict(
            dim = 128,
            depth = 1,
            window_size = 8
        )
    )

    replay = ReplayBuffer(
        './replay_test_epo',
        max_episodes = 4,
        max_timesteps = 6,
        fields = dict(
            state = ('float', 10),
            action = ('float', 10),
            action_log_prob = ('float', 10),
            reward = 'float',
            value = 'float',
            done = 'bool',
            cond_mask = 'bool',
        ),
        meta_fields = dict(
            latent_gene_id = 'int',
            cum_rewards = 'float'
        )
    )

    class MockEnv:
        def reset(self):
            return np.random.normal(size = (10,)), {}

        def step(self, action):
            reward = float(np.sum(action))
            return np.random.normal(size = (10,)), reward, False, False, {}

    env = MockEnv()

    # rollout for 4 genes

    all_fitnesses = []

    for gene_id in range(4):

        cum_reward = model.gather_experience_from_env_(
            env = env,
            replay = replay,
            num_envs = 1,
            max_timesteps = 5,
            latent_gene_id = gene_id
        )

        all_fitnesses.append(cum_reward)

    # verify storage in meta

    assert 'latent_gene_id' in replay.meta_data
    assert replay.meta_data['latent_gene_id'].shape == (4,) # 4 episodes

    # update gene pool

    fitness = torch.tensor(all_fitnesses)

    model.latent_gene_pool.genetic_algorithm_step(fitness)

    # cleanup
    import shutil
    shutil.rmtree('./replay_test_epo', ignore_errors = True)

def test_cross_episodic_rollout_memory():
    import numpy as np
    import shutil
    from memmap_replay_buffer import ReplayBuffer

    model = Locoformer(
        embedder = nn.Linear(10, 128),
        unembedder = dict(
            num_continuous = 10
        ),
        value_network = MLP(128, 64, 32),
        dim_value_input = 32,
        reward_range = (-100., 100.),
        transformer = dict(
            dim = 128,
            depth = 1,
            window_size = 8
        )
    )

    class MockEnv:
        def __init__(self):
            self.t = 0

        def reset(self):
            self.t = 0
            return np.zeros((10,), dtype = np.float32), {}

        def step(self, action):
            self.t += 1
            terminated = self.t == 5
            reward = 100. if terminated else 1.
            return np.full((10,), self.t, dtype = np.float32), reward, terminated, False, {}

    replay = ReplayBuffer(
        './replay_test_cross_episodic',
        max_episodes = 8,
        max_timesteps = 8,
        fields = dict(
            state = ('float', 10),
            action = ('float', 10),
            action_log_prob = ('float', 10),
            reward = 'float',
            value = 'float',
            done = 'bool',
            cond_mask = 'bool'
        ),
        meta_fields = dict(
            cum_rewards = 'float'
        )
    )

    gather = lambda env: model.gather_experience_from_env_(env, replay, num_envs = 1, max_timesteps = 9)

    env = MockEnv()

    cum_reward, steps = gather(env)
    assert steps == 5 and np.isclose(cum_reward, 104.)

    first_stateful_forward = model.rollout_stateful_forward
    assert exists(first_stateful_forward)

    # reuse the rollout wrapper within the same environment

    cum_reward, steps = gather(env)
    assert model.rollout_stateful_forward is first_stateful_forward
    assert steps == 5 and np.isclose(cum_reward, 104.)

    # reset the rollout wrapper on environment change

    gather(MockEnv())
    assert model.rollout_stateful_forward is not first_stateful_forward

    dl = replay.dataloader(batch_size = 3, shuffle = False)
    batch = next(iter(dl))

    # terminal transition is learnable; only the appended bootstrap node is done

    assert (batch['_lens'] == 6).all()
    assert (~batch['done'][:, :5]).all()
    assert batch['done'][:, 5].all()
    assert (batch['reward'][:, 4] == 100.).all()
    assert (batch['value'][:, 5] == 0.).all()

    # rollout log probs must match ppo, which resets the cache at episode boundaries

    logits, _ = model(batch['state'])
    recomputed_log_probs = model.unembedder.log_prob(logits, batch['action'], concat = True)

    assert torch.allclose(recomputed_log_probs[:, :5], batch['action_log_prob'][:, :5], atol = 1e-5)

    # cleanup

    shutil.rmtree('./replay_test_cross_episodic', ignore_errors = True)

def test_latent_dynamics_forward():
    from locoformer.locoformer import ForwardDynamics

    # discrete action - drawn from an embedding table of width dim_action

    dynamics = ForwardDynamics(
        dim = 64,
        num_discrete = 6,
        num_continuous = 3,
        selectors = [
            [[4, 5]],
            [[0, 1, 2, 3]],
            [0, 1],
        ],
        dim_action = 32
    )

    assert len(dynamics.to_dynamics.layers) == 5 # create_mlp: proj in + 3 hidden layers with silu + proj out

    latent = torch.randn(2, 4, 64)

    discrete_action = torch.randint(0, 2, (2, 4, 1))
    pred = dynamics(latent, discrete_action, selector_index = 0)
    assert pred.shape == (2, 4, 64)

    continuous_action = torch.randn(2, 4, 2)
    pred = dynamics(latent, continuous_action, selector_index = 2)
    assert pred.shape == (2, 4, 64)

    # loss

    target = torch.randn(2, 4, 64)
    mask = torch.ones(2, 4, dtype = torch.bool)

    loss = dynamics.calculate_loss(pred, target, mask = mask)
    assert loss.shape == (8,)
    assert loss.dtype == torch.float32

def test_latent_dynamics_residual_and_probabilistic():
    import torch.nn.functional as F
    from locoformer.locoformer import ForwardDynamics

    # without residual, the prediction should not add the latent back

    dynamics = ForwardDynamics(
        dim = 64,
        num_discrete = 6,
        num_continuous = 2,
        selectors = [
            [[4, 5]],
            [[0, 1, 2, 3]],
            [0, 1],
        ],
        residual = False
    )

    latent = torch.randn(2, 4, 64)
    action = torch.randint(0, 2, (2, 4, 1))

    pred = dynamics(latent, action, selector_index = 0)
    assert pred.shape == (2, 4, 64)

    # predicting a distribution, for a state entropy bonus during rollout

    dynamics = ForwardDynamics(
        dim = 64,
        num_discrete = 6,
        num_continuous = 2,
        selectors = [
            [[4, 5]],
            [[0, 1, 2, 3]],
            [0, 1],
        ],
        predict_dist = True
    )

    assert dynamics.predict_dist

    # forward returns the mean, in both train and eval mode

    dynamics.train()

    pred = dynamics(latent, action, selector_index = 0)
    assert pred.shape == (2, 4, 64)

    dynamics.eval()

    mean_pred = dynamics(latent, action, selector_index = 0)
    assert mean_pred.shape == (2, 4, 64)
    assert torch.allclose(pred, mean_pred)

    # entropy of the predicted distribution

    entropy = dynamics.entropy(latent, action, selector_index = 0)
    assert entropy.shape == (2, 4, 64)
    assert torch.isfinite(entropy).all()

    # cosine similarity loss, with the log variance calibrated through the likelihood on the scale

    target = torch.randn(2, 4, 64)

    mean_pred, log_var = dynamics.predict(latent, action, selector_index = 0)

    loss = dynamics.calculate_loss(mean_pred, target, log_var = log_var)
    assert loss.shape == (2, 4)

    expected_location_loss = 2. - F.cosine_similarity(mean_pred, target, dim = -1)
    expected_scale_loss = ((target - mean_pred.detach()).pow(2) / log_var.exp() + log_var) * 0.5
    expected_loss = expected_location_loss + expected_scale_loss.mean(dim = -1)

    assert torch.allclose(loss, expected_loss)

    # loss without log variance is plain cosine similarity

    dynamics = ForwardDynamics(
        dim = 64,
        num_discrete = 6,
        num_continuous = 2,
        selectors = [
            [[4, 5]],
            [[0, 1, 2, 3]],
            [0, 1],
        ],
    )

    pred = dynamics(latent, action, selector_index = 0)
    loss = dynamics.calculate_loss(pred, target)
    assert loss.shape == (2, 4)

    expected_loss = 2. - F.cosine_similarity(pred, target, dim = -1)
    assert torch.allclose(loss, expected_loss)

    # entropy not available if not predicting a distribution

    with pytest.raises(AssertionError):
        dynamics.entropy(latent, action, selector_index = 0)

@param('use_ema_target', (None, True, False))
@param('predict_dist', (False, True))
def test_latent_dynamics_ppo(use_ema_target, predict_dist):
    from torch.optim import Adam

    # `use_ema_target` = None means no SPR, otherwise SPR with or without ema target network

    has_spr = exists(use_ema_target)

    latent_dynamics = None

    if has_spr:
        latent_dynamics = dict(
            dim_action = 32,
            use_ema_target = use_ema_target,
            target_ema_decay = 0.99,
            predict_dist = predict_dist
        )

    model = Locoformer(
        embedder = dict(dim = 64, dim_state = 4),
        unembedder = dict(
            dim = 64,
            num_discrete = 2,
            num_continuous = 0,
        ),
        transformer = dict(
            dim = 64,
            dim_head = 32,
            heads = 4,
            depth = 2,
            window_size = 32,
        ),
        discount_factor = 0.99,
        policy_network = nn.Identity(),
        value_network = nn.Identity(),
        dim_value_input = 64,
        reward_range = (-300., 300.),
        latent_dynamics = latent_dynamics
    )

    assert model.has_latent_dynamics == has_spr
    assert model.use_ema_target == (use_ema_target is True)

    # ema target network is only kept when ema target is enabled

    if use_ema_target is True:
        assert model.target_embedder is not None
        assert model.target_proj_head is not None
    else:
        assert model.target_embedder is None
        assert model.target_proj_head is None

    b, n = 4, 32

    state = torch.randn(b, n, 4)
    action = torch.randint(0, 2, (b, n, 1))
    action_log_prob = torch.zeros(b, n, 1)
    reward = torch.randn(b, n)
    value = torch.zeros(b, n)
    done = torch.zeros(b, n, dtype = torch.bool)
    episode_lens = torch.full((b,), n)

    optims = [Adam(model.base_parameters(), lr = 1e-3)]

    target_weight_before = model.target_embedder.state_to_token[0].layers[0].weight.clone() if use_ema_target is True else None
    dynamics_weight_before = model.latent_dynamics.to_dynamics.layers[0][0].weight.clone() if has_spr else None
    embedder_weight_before = model.embedder.state_to_token[0].layers[0].weight.clone()

    actor_loss, critic_loss = model.ppo(
        state,
        None,
        action,
        action_log_prob,
        reward,
        value,
        done,
        episode_lens,
        optims = optims,
        state_embed_kwargs = dict(state_type = 'raw'),
        action_select_kwargs = dict(selector_index = 0),
        compute_state_pred_loss = True,
    )

    assert torch.isfinite(actor_loss) and torch.isfinite(critic_loss)

    # embedder should be trained through the ppo loss, and through the latent prediction loss when spr is on

    assert not torch.allclose(embedder_weight_before, model.embedder.state_to_token[0].layers[0].weight)

    if has_spr:
        assert not torch.allclose(dynamics_weight_before, model.latent_dynamics.to_dynamics.layers[0][0].weight)

    # target network should be updated only via ema

    if use_ema_target is True:
        assert not torch.allclose(target_weight_before, model.target_embedder.state_to_token[0].layers[0].weight)

@param('per_action_dim', (False, True))
def test_ppo_per_action_dim_and_grad_clipping(per_action_dim):
    from torch.optim import SGD
    from locoformer.locoformer import BetaDist

    # default beta concentration at zero raw params, including the unimodal floor

    unit_mean, conc = BetaDist(unimodal = True).beta.concentrations(torch.zeros(1, 2))
    assert torch.allclose(unit_mean, torch.tensor([0.5]))
    assert torch.allclose(conc, torch.tensor([8.]))

    class SpySGD(SGD):
        def step(self, *args, **kwargs):
            self.grad_norm = torch.sqrt(sum(p.grad.norm().square() for group in self.param_groups for p in group['params'] if exists(p.grad)))
            return super().step(*args, **kwargs)

    b, n, dim_state, dim_action = 4, 16, 12, 6
    window_size = 8

    model = Locoformer(
        embedder = dict(dim = 32, dim_state = dim_state),
        unembedder = dict(
            dim = 32,
            num_continuous = dim_action,
            continuous_dist_type = 'beta',
            selectors = [list(range(dim_action))]
        ),
        transformer = dict(dim = 32, dim_head = 16, heads = 2, depth = 1, window_size = window_size),
        policy_network = MLP(32, 32),
        value_network = MLP(32, 32),
        dim_value_input = 32,
        reward_range = (-100., 100.),
        per_action_dim = per_action_dim,
        value_loss_weight = 100., # large critic loss should not throttle the actor
        calc_gae_kwargs = dict(use_accelerated = False)
    )

    state = torch.randn(b, n, dim_state)
    action = torch.zeros(b, n, dim_action)
    action_log_prob = torch.zeros(b, n, dim_action)
    reward = torch.randn(b, n)
    value = torch.zeros(b, n)
    done = torch.zeros(b, n, dtype = torch.bool)
    episode_lens = torch.tensor([8, 6, 5, 3]) # second window is fully padded and should be skipped

    max_grad_norm = 1e-3

    optims = [
        SpySGD(model.base_parameters(), lr = 1e-3),
        SpySGD(model.actor_parameters(), lr = 1e-3),
        SpySGD(model.critic_parameters(), lr = 1e-3)
    ]

    actor_loss, critic_loss = model.ppo(
        state, None, action, action_log_prob, reward, value, done, episode_lens,
        optims = optims,
        state_embed_kwargs = dict(state_type = 'raw'),
        action_select_kwargs = dict(selector_index = 0),
        compute_state_pred_loss = False,
        max_grad_norm = max_grad_norm
    )

    assert torch.isfinite(actor_loss) and torch.isfinite(critic_loss)

    # each optimizer is clipped independently, so the actor is not throttled by the critic

    assert optims[1].grad_norm > max_grad_norm * 0.5
    assert optims[2].grad_norm <= max_grad_norm * 1.01

def test_vectorized_gather_experience_early_stop():
    import numpy as np
    import shutil
    from memmap_replay_buffer import ReplayBuffer
    from locoformer.locoformer import calc_gae

    shutil.rmtree('./replay_test_vec_stop', ignore_errors = True)

    model = Locoformer(
        embedder = dict(dim = 32, dim_state = 4),
        unembedder = dict(dim = 32, num_continuous = 2, continuous_dist_type = 'beta', selectors = [[0, 1]]),
        transformer = dict(dim = 32, dim_head = 16, heads = 2, depth = 1, window_size = 8),
        policy_network = nn.Identity(),
        value_network = nn.Identity(),
        dim_value_input = 32,
        reward_range = (-100., 100.),
        calc_gae_kwargs = dict(use_accelerated = False)
    ).eval()

    class MockVecEnv:
        def __init__(self):
            self.t = 0

        def reset(self):
            self.t = 0
            return np.zeros((2, 4), dtype = np.float32), {}

        def step(self, action):
            self.t += 1
            obs = np.full((2, 4), float(self.t), dtype = np.float32)
            # env 0 terminates at step 2 with reward 50; env 1 truncates at step 4 with reward 2
            rew = np.array([50. if self.t == 2 else 10., 2.], dtype = np.float32)
            term = np.array([self.t == 2, False], dtype = bool)
            trunc = np.array([False, self.t == 4], dtype = bool)
            return obs, rew, term, trunc, {}

    replay = ReplayBuffer(
        './replay_test_vec_stop',
        max_episodes = 2,
        max_timesteps = 8,
        fields = dict(
            state = ('float', 4),
            action = ('float', 2),
            action_log_prob = ('float', 2),
            reward = 'float',
            value = 'float',
            done = 'bool',
            cond_mask = 'bool'
        ),
        meta_fields = dict(cum_rewards = 'float'),
        overwrite = True
    )

    mean_cum_reward, steps = model.gather_experience_from_env_(
        MockVecEnv(),
        replay,
        num_envs = 2,
        max_timesteps = 7,
        state_embed_kwargs = dict(state_type = 'raw'),
        action_select_kwargs = dict(selector_index = 0),
        state_id_kwarg = dict(state_id = 0)
    )

    # env 0 accumulated 10 + 50 = 60 over 2 steps; env 1 accumulated 2 * 4 = 8 over 4 steps

    assert steps == 4
    assert np.isclose(mean_cum_reward, (60. + 8.) / 2.)

    batch = next(iter(replay.dataloader(batch_size = 2, shuffle = False)))

    # env 0 has 2 learnable steps and done=True for the padding steps after its terminal transition

    assert batch['done'][0].tolist()[:5] == [False, False, True, True, True]
    assert batch['reward'][0, 2].item() == 0. and batch['value'][0, 2].item() == 0.

    # env 1 has 4 learnable steps and one bootstrap node at step 4

    assert batch['done'][1].tolist()[:5] == [False, False, False, False, True]
    assert torch.allclose(batch['reward'][1, 4], batch['value'][1, 4])

    # batched GAE matches per-episode GAE

    masks = ~batch['done'][:, :5]
    gae_batch, ret_batch = calc_gae(batch['reward'][:, :5], batch['value'][:, :5], masks = masks, use_accelerated = False)
    gae_0, ret_0 = calc_gae(batch['reward'][0:1, :3], batch['value'][0:1, :3], masks = masks[0:1, :3], use_accelerated = False)
    gae_1, ret_1 = calc_gae(batch['reward'][1:2, :5], batch['value'][1:2, :5], masks = masks[1:2, :5], use_accelerated = False)

    assert torch.allclose(gae_batch[0, :2], gae_0[0, :2], atol = 1e-5)
    assert torch.allclose(ret_batch[0, :2], ret_0[0, :2], atol = 1e-5)
    assert torch.allclose(gae_batch[1, :4], gae_1[0, :4], atol = 1e-5)
    assert torch.allclose(ret_batch[1, :4], ret_1[0, :4], atol = 1e-5)

    shutil.rmtree('./replay_test_vec_stop', ignore_errors = True)
