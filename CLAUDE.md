# CLAUDE.md - TransformerXL PPO with Thinking Tokens

## Commands
- Training: `python train_PPO_trXL.py`
- Interactive gameplay: `python play_game.py [--god_mode] [--debug] [--save_trajectories] [--fps <value>]`
- Experiment tracking: Uses WandB (configure in train_PPO_trXL.py)

## Code Style Guidelines
- **Naming**: Classes in PascalCase, functions/variables in snake_case, constants in UPPERCASE
- **Imports**: System imports first, third-party packages second, local modules last
- **Formatting**: 4-space indentation
- **Types**: Type annotations for function signatures and class attributes
- **Error handling**: Use try/except blocks for directory creation and environment setup
- **Documentation**: Docstrings for scripts and major functions
- **Constants**: Define configuration parameters at module level
- **JAX patterns**: Prefer pure functions, use jit when appropriate, seed RNGs properly

## Project Structure
- TransformerXL architecture with PPO implementation in JAX/Flax
- Integration with craftax/gymnax environments
- Extension with "thinking tokens" for internal reasoning steps
