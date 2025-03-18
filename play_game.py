#!/usr/bin/env python3

"""
This script runs the interactive Craftax player.

Controls:
w: up
a: left
s: down
d: right
space: do (interact)
Many other keys for different actions (see output when run)

Run with:
python play_game.py

Optional arguments:
--god_mode: Invincibility
--debug: Run without JIT compilation
--save_trajectories: Save your gameplay
--fps: Set the frames per second (default 60)

Example: python play_game.py --god_mode --fps 30
"""

import sys
from craftax.craftax.play_craftax import entry_point

if __name__ == "__main__":
    # Forward command line arguments to the craftax play script
    sys.argv = [sys.argv[0]] + sys.argv[1:]
    entry_point() 