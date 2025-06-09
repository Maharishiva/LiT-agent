# (LiT-agent) TransformerXL PPO with Thinking Tokens

<p align="center">
 <img width="80%" src="https://raw.githubusercontent.com/Maharishiva/LiT-agent/assets/LiT_agent.png" />
</p>

An experimental implementation of TransformerXL PPO that adds thinking tokens to the agent's action space, allowing the model to perform internal reasoning steps before committing to environment actions.

## Acknowledgment

This repo is built on top of [Reytuag's TransformerXL PPO JAX implementation](https://github.com/Reytuag/transformerXL_PPO_JAX). Base transformer architecture code and PPO implementation comes from that excellent repository - we've extended further to introduce thinking tokens.

## What's This About?

This project experiments with adding explicit reasoning steps to RL agents, inspired by:
- effectiveness of Chain-of-Thought (CoT) in LLMs
- recent breakthroughs in LLM reasoning capabilities with RL
- decoupling computeation from environment interaction and model size

## Implementation

- We extend the action space with special thinking tokens with randomly initialized embeddings
- The transformer's memory tracks reasoning traces
- The reward structure should encourage efficient, meaningful thinking patterns
- Uses combined token embeddings that concatenate state observations, previous actions, and previous rewards

## Features

- **Combined Token Embeddings**: Each token processed by the transformer is a combination of:
  - Current observation state embedding
  - Previous action embedding
  - Previous reward embedding
- This enriched sequential representation helps the model capture temporal dependencies in state transitions, agent behavior, and received rewards

## Roadmap

- [X] Add combined token embeddings (observation, previous action, previous reward)
- [X] Adding thinking tokens to the action space and action embedding
- [X] Run tests on creaftax and gymnax environments
- [ ] Investigate the effect of different reward functions and discounting strategies
- [ ] Add world model based learning (museli) style



---

This is just the beginning of blending LLM-style reasoning with classical RL - please star our repo and join us for the ride!

## Citation

When using this codebase, please cite both this repository and the original implementation:

```
@software{LiT-agent,
  author = {Fedor Kurdov},
  title = {LiT-agent: TransformerXL PPO with Thinking Tokens},
  year = {2025},
  keywords = {Transformer ; Reinforcement Learning ; Emergent Reasoning ; Latent Reasoning},
  url = {https://github.com/Maharishiva/LiT-agent/tree/main}
}

@softwareversion{hamon:hal-04659863v1,
  TITLE = {{transformerXL_PPO_JAX}},
  AUTHOR = {Hamon, Gautier},
  URL = {https://inria.hal.science/hal-04659863},
  YEAR = {2024},
  MONTH = Jul,
  REPOSITORY = {https://github.com/Reytuag/transformerXL_PPO_JAX},
  LICENSE = {MIT License},
  KEYWORDS = {Transformer ; Reinforcement Learning ; JAX},
  HAL_ID = {hal-04659863},
  HAL_VERSION = {v1},
}
``` 