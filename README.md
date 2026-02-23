<img src="./fig3.png" width="400px"></img>

## LocoFormer (wip)

[LocoFormer - Generalist Locomotion via Long-Context Adaptation](https://generalist-locomotion.github.io/)

The gist is they trained a simple Transformer-XL in simulation on robots with many different bodies (cross-embodiment) and extreme domain randomization. When transferring to the real-world, they noticed the robot now gains the ability to adapt to insults. The XL memories span across multiple trials, which allowed the robot to learn in-context adaptation.

## Install

```bash
$ pip install locoformer
```

## Usage

```python
import torch
from locoformer.locoformer import Locoformer

# mock robot embodied with some state dimensions and action dimensions

locoformer = Locoformer(
    embedder = dict(
        dim = 512,
        dim_state = [32, 16], # support multiple bodies / robots
    ),
    unembedder = dict(
        num_continuous = 12 + 6,
        selectors = [
            list(range(12)),
            list(range(12, 12 + 6))
        ]
    ),
    transformer = dict(
        dim = 512,
        depth = 6,
        heads = 8,
        window_size = 32
    )
)

# mock state from one of the robots (0th one)

state = torch.randn(1, 1, 32)

# forward to get action logits

action_logits, _ = locoformer(
    state,
    state_embed_kwargs = dict(state_type = 'raw'),
    state_id_kwarg = dict(state_id = 0),
    action_select_kwargs = dict(selector_index = 0)
)

# sample action using the internal distribution

action = locoformer.unembedder.sample(action_logits, selector_index = 0) # (1, 1, 12)
```

## Sponsors

This open sourced work is sponsored by [Safe Sentinel](https://www.safesentinels.com/)

## Citations

```bibtex
@article{liu2025locoformer,
    title   = {LocoFormer: Generalist Locomotion via Long-Context Adaptation},
    author  = {Liu, Min and Pathak, Deepak and Agarwal, Ananye},
    journal = {Conference on Robot Learning ({CoRL})},
    year    = {2025}
}
```

```bibtex
@inproceedings{anonymous2025flow,
    title   = {Flow Policy Gradients for Legged Robots},
    author  = {Anonymous},
    booktitle = {Submitted to The Fourteenth International Conference on Learning Representations},
    year    = {2025},
    url     = {https://openreview.net/forum?id=BA6n0nmagi},
    note    = {under review}
}
```

```bibtex
@misc{ashlag2025stateentropyregularizationrobust,
    title   = {State Entropy Regularization for Robust Reinforcement Learning}, 
    author  = {Yonatan Ashlag and Uri Koren and Mirco Mutti and Esther Derman and Pierre-Luc Bacon and Shie Mannor},
    year    = {2025},
    eprint  = {2506.07085},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2506.07085}, 
}
```
