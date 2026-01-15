# SnakeAI Reinforcement Learning Agents

This repository contains several Snake AI variants built with PyTorch and Pygame. The two actively maintained entries are **SNAKEAI_MASTER_V3** (single agent with visualizer) and **SNAKEAI_MULTI_V2** (multi-agent arena). The instructions below focus on setting up, training, and demoing those two versions.

## Quick Start

- Install Python 3.10+ and pip.
- (Recommended) Create a virtual environment: `python -m venv .venv` then activate it.
- Install dependencies from the project root:
  - `pip install -r requirements.txt`
- Place any pre-trained model at `./models/model.pth` inside each variant folder if you want to resume or run demos with learned weights.

## Running SNAKEAI_MASTER_V3

- **Train** (shows Pygame window and NN visualizer by default):
  1. `cd SNAKEAI_MASTER_V3`
  2. `python main.py`
- **Demo / Inference only** (no training, uses saved model):
  1. `cd SNAKEAI_MASTER_V3`
  2. `python demo.py`
- **Key toggles** in [SNAKEAI_MASTER_V3/settings.py](SNAKEAI_MASTER_V3/settings.py):
  - `RENDER` (enable/disable Pygame window to speed training).
  - `SHOW_VISUALIZER` and `VISUALIZE_INPUT_SLICE` (neural net panel alongside the game window).
  - `LOAD_MODEL` and `MODEL_PATH` (resume training or run demos from saved weights).
  - `AGENTS`, `NUM_FOOD`, and learning hyperparameters such as `LR`, `GAMMA`, `BATCH_SIZE`, and `TARGET_UPDATE_SIZE`.
- Notes:
  - Training loop saves to `models/model.pth` when a new record is reached.
  - Use Ctrl+C to stop; plots are emitted via matplotlib in real time.

## Running SNAKEAI_MULTI_V2

- **Train** (headless by default for speed):
  1. `cd SNAKEAI_MULTI_V2`
  2. `python main.py`
- **Demo / Inference only** (renders the arena and uses the saved model):
  1. `cd SNAKEAI_MULTI_V2`
  2. `python demo.py`
- **Key toggles** in [SNAKEAI_MULTI_V2/settings.py](SNAKEAI_MULTI_V2/settings.py):
  - `RENDER` (set True to watch training; False is fastest).
  - `AGENTS` roster defines the snakes (names and colors) that spawn in the arena.
  - `LOAD_MODEL`, `MODEL_PATH`, and `EPSILON_*` control resume behavior and exploration.
  - Reward shaping, starvation limits, board size, and network dimensions are centralized here.
- Notes:
  - Multi-agent loop trains and logs per-match scores; models save to `models/model.pth` when records are met.
  - Ctrl+C cleanly stops the run.

## Tips and Troubleshooting

- If Pygame windows do not appear, ensure `RENDER` is True and a display is available.
- For faster experimentation, keep `RENDER` False during training and enable it only for demos.
- If a demo reports that the model did not load, verify that `models/model.pth` exists and `LOAD_MODEL` is True in the corresponding settings file.
- GPU acceleration is configurable via `DEVICE` in each settings module (defaults to CPU).
