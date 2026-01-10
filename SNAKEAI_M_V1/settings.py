import torch

# ==============================================================================
#                               MASTER CONFIGURATION
# ==============================================================================

# --- SYSTEM & FILES ---
LOAD_MODEL = False             # Set to True to resume training from a saved file
MODEL_PATH = './models/model.pth'
RENDER = True                 # Set to False to speed up training (no GUI)
SPEED = 60                    # Game speed (frames per sec). Use 0 for max speed.

# --- DIMENSIONS & GRAPHICS ---
BLOCK_SIZE = 20
WIDTH = 960                   # Map Width (Must be multiple of BLOCK_SIZE)
HEIGHT = 720                  # Map Height (Must be multiple of BLOCK_SIZE)
COLOR_FOOD = (200, 0, 0)  # Red

# Colors (R, G, B)
COLOR_BG = (0, 0, 0)          # Black Background
COLOR_TEXT = (255, 255, 255)  # White Text

# --- MULTI-AGENT SETUP ---
# Defining the roster of agents. 
# For now, keep it to 1 until you refactor game.py completely.
AGENTS = [
    {
        "name": "Viper", 
        "color_head": (0, 255, 0),    # Green Head
        "color_body": (0, 200, 0)     # Darker Green Body
    },
    # Future Agents (Uncomment when multi-agent logic is ready):
    # {
    #     "name": "Python", 
    #     "color_head": (0, 0, 255),  # Blue Head
    #     "color_body": (0, 0, 200)   # Darker Blue Body
    # },
]

NUM_FOOD = 1                  # How many apples exist on screen at once?

# --- HYPERPARAMETERS ---
MAX_MEMORY = 500_000          # Experience Replay Buffer Size
BATCH_SIZE = 1000             # How many memories to train on per game
LR = 0.0001                   # Learning Rate (Stepsize for the brain)
GAMMA = 0.9                   # Discount Factor (0.9 = cares about future, 0.1 = short sighted)
TARGET_UPDATE_SIZE = 50
# ... (rest of your settings file) ...

# --- GRAPHS & LOGS ---
PLOT_TITLE = "Training Performance: Snake AI V4"
PLOT_DESCRIPTION = "Solid Line = Score | Dotted Line = Mean Score"

# --- EXPLORATION (Epsilon) ---
# Randomness logic: Epsilon = START - (n_games // DECAY)
EPSILON_START = 100            # Initial randomness % (e.g. 80%)
EPSILON_MIN = 2               # The "Floor" (Never go below 2% random)
EPSILON_DECAY = 25            # Higher number = Slower decay (Longer exploration phase)

# --- REWARDS (The "Definition of Bad") ---
REWARD_FOOD = 10
REWARD_STEP = -0.01
REWARD_COLLISION = -15
# Formula: final_penalty = REWARD_COLLISION - (score * REWARD_STARVE_MULTIPLIER)
# Example at Score 50: -15 - (50 * 0.5) = -40 penalty
REWARD_STARVE_MULTIPLIER = 0.5 

# --- MODEL ARCHITECTURE ---
# You can now change the brain size here without touching model.py
INPUT_SIZE = 30               # 24 Ray + 4 Orientation + 2 Food Direction
HIDDEN_SIZE_1 = 256           # First Hidden Layer
HIDDEN_SIZE_2 = 128           # Second Hidden Layer
HIDDEN_SIZE_3 = 64            # Third Hidden Layer
OUTPUT_SIZE = 3               # [Straight, Right, Left]

# --- DEVICE CONFIG ---
# Automatically picks GPU (cuda/mps) if available, otherwise CPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'