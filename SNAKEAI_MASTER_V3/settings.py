import torch

# ==============================================================================
#                               MASTER CONFIGURATION
# ==============================================================================

# --- SYSTEM & FILES ---
RENDER = True
                 # NOTE: Set to False to speed up training (no GUI)

MODEL_PATH = './models/model.pth'
STARVE_LIMIT = 80             # Steps allowed per body length before starvation
SPEED = 30 if RENDER else 0    # Game speed (frames per sec). Use 0 for max speed.

LOAD_MODEL = True              # NOTE: This will Reduce randomness

# --- GRAPHS & LOGS ---
PLOT_TITLE = "MASTER SNAKE V3 / FOOD SMELL / FRAMESTACKING / 38 x 3" # NOTE: Name
PLOT_DESCRIPTION = "Solid Line = Score | Dotted Line = Mean Score"

# --- DIMENSIONS & GRAPHICS ---
BLOCK_SIZE = 30
WIDTH = 1200                   # Map Width (Must be multiple of BLOCK_SIZE)
HEIGHT = 1200                  # Map Height (Must be multiple of BLOCK_SIZE)
COLOR_FOOD = (200, 0, 0)  # Red

# Colors (R, G, B)
COLOR_BG = (0, 0, 0)          # Black Background
COLOR_TEXT = (255, 255, 255)  # White Text
GREEN = (0, 200, 0)
BLUE = (0, 0, 200)
RED = (200, 0, 0)
DARK_RED = (128, 0, 0)
# --- MULTI-AGENT SETUP ---
# Defining the roster of agents. 
# For now, keep it to 1 until you refactor game.py completely.
AGENTS = [
    {
        "name": "Viper", 
        "color": BLUE   
    },
    # Future Agents (Uncomment when multi-agent logic is ready):
    # {
    #     "name": "Python", 
    #     "color": (0, 0, 200)   # Darker Blue Body
    # },
]

NUM_FOOD = 2                  # How many apples exist on screen at once?

# --- HYPERPARAMETERS ---
MAX_MEMORY = 500_000          # Experience Replay Buffer Size
BATCH_SIZE = 1500             # How many memories to train on per game
LR = 0.0005                    # Learning Rate (Stepsize for the brain)
GAMMA = 0.95                   # Discount Factor (0.9 = cares about future, 0.1 = short sighted)
TARGET_UPDATE_SIZE = 100      # Number of games take to update the Target NN

# --- EXPLORATION (Epsilon) ---
# Randomness logic: Epsilon = START - (n_games // DECAY)
EPSILON_START = 100            # Initial randomness % (e.g. 100%)
EPSILON_MIN = 0               # The "Floor" (Never go below n% random)
EPSILON_DECAY = 200            # Higher number = Slower decay (Longer exploration phase)
EPSILON_MEMORY_LOAD = EPSILON_START * EPSILON_DECAY # Trick to reduce epsilon immediately when loading memory
# EPSILON_MEMORY_LOAD = 17000

# --- REWARDS (The "Definition of Bad") ---
REWARD_FOOD = 30
REWARD_STEP = -0.01
REWARD_KILL = 50  # Big bonus for taking out an enemy
REWARD_COLLISION = -40
# Formula: final_penalty = REWARD_COLLISION - (score * REWARD_STARVE_MULTIPLIER)
# Example at Score 50: -15 - (50 * 0.5) = -40 penalty
REWARD_STARVE_MULTIPLIER = 0.5 

# --- MODEL ARCHITECTURE ---
# You can now change the brain size here without touching model.py
STACK_SIZE = 3               # How many frames to remember
INPUT_SIZE = 38 * STACK_SIZE  
HIDDEN_SIZE_0 = 512
HIDDEN_SIZE_1 = 256           # First Hidden Layer
# HIDDEN_SIZE_2 = 128           # Second Hidden Layer
# HIDDEN_SIZE_3 = 64            # Third Hidden Layer
OUTPUT_SIZE = 3               # [Straight, Right, Left]

# --- NN VISUALIZER CONFIG ---
SHOW_VISUALIZER = True         # Master toggle
VISUALIZE_INPUT_SLICE = True   # If True, only shows the latest 38 inputs (Live frame)
                               # If False, shows all 114 inputs (Stacked frames)
VIS_WIDTH = 1200
VIS_HEIGHT = 1200

# Now calculate the Visualizer's Position dynamically
# It starts exactly where the Game Width ends (+ some padding)
VIS_X_START = WIDTH + 40 
VIS_Y_START = ((HEIGHT - VIS_HEIGHT) // 2) + 30

VIS_CONFIG = {
    "X": VIS_X_START,         # <--- Dynamic X Position
    "Y": VIS_Y_START,         
    "WIDTH": VIS_WIDTH,              
    "HEIGHT": VIS_HEIGHT,             
    "MAX_NODES": 38,
    "NODE_RADIUS": 10,
    "LAYER_SPACING": 100,      
    "COLORS": {
        "BG": (0, 0, 0),
        "NODE_OFF": (0, 0, 0),
        "NODE_ON": (0, 255, 0),
        "NODE_NEG": (255, 0, 0),
        "WEIGHT_POS": (0, 180, 0),
        "WEIGHT_NEG": (180, 0, 0),
        "TEXT": (200, 200, 200)
    }
}
# --- DEVICE CONFIG ---
# Automatically picks GPU (cuda/mps) if available, otherwise CPU
DEVICE = 'cpu'
# DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'