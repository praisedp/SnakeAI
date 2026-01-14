import torch

# ==============================================================================
#                               MASTER CONFIGURATION
# ==============================================================================

# --- SYSTEM & FILES ---
RENDER = False                 # NOTE: Set to False to speed up training (no GUI)

MODEL_PATH = './models/model.pth'
STARVE_LIMIT = 80             # Steps allowed per body length before starvation
SPEED = 20 if RENDER else 0    # Game speed (frames per sec). Use 0 for max speed.

LOAD_MODEL = True              # NOTE: This will Reduce randomness

# --- GRAPHS & LOGS ---
PLOT_TITLE = "MULTI V2 / FOOD SMELL / FRAMESTACKING / 46 x 3" # NOTE: Name
PLOT_DESCRIPTION = "Solid Line = Score | Dotted Line = Mean Score"

# --- DIMENSIONS & GRAPHICS ---
BLOCK_SIZE = 20
WIDTH = 800                   # Map Width (Must be multiple of BLOCK_SIZE)
HEIGHT = 800                  # Map Height (Must be multiple of BLOCK_SIZE)
COLOR_FOOD = (200, 0, 0)  # Red

# Colors (R, G, B)
COLOR_BG = (0, 0, 0)          # Black Background
COLOR_TEXT = (255, 255, 255)  # White Text

GREEN = (0, 200, 0)
BLUE = (0, 0, 200)
RED = (200, 0, 0)
YELLOW =(255, 255, 0)
# --- MULTI-AGENT SETUP ---
# Defining the roster of agents. 
# For now, keep it to 1 until you refactor game.py completely.
AGENTS = [
    {
        "name": "Viper", 
        "color": RED   
    },
    {
        "name": "Python", 
        "color": BLUE  
    },
    # {
    #     "name": "Cobra", 
    #     "color": GREEN  
    # },
    # {
    #     "name": "Rat Snake", 
    #     "color": YELLOW  
    # }
]

NUM_FOOD = 3                  # How many apples exist on screen at once?

# --- HYPERPARAMETERS ---
MAX_MEMORY = 100_000          # Experience Replay Buffer Size
BATCH_SIZE = 1000             # How many memories to train on per game
LR = 0.0001                    # Learning Rate (Stepsize for the brain)
GAMMA = 0.95                   # Discount Factor (0.9 = cares about future, 0.1 = short sighted)
TARGET_UPDATE_SIZE = 100      # Number of games take to update the Target NN

# --- EXPLORATION (Epsilon) ---
# Randomness logic: Epsilon = START - (n_games // DECAY)
EPSILON_START = 100            # Initial randomness % (e.g. 100%)
EPSILON_MIN = 5               # The "Floor" (Never go below n% random)
EPSILON_DECAY = 100            # Higher number = Slower decay (Longer exploration phase)
EPSILON_MEMORY_LOAD = EPSILON_START * EPSILON_DECAY # Trick to reduce epsilon immediately when loading memory
EPSILON_MEMORY_LOAD = 8500

# --- REWARDS (The "Definition of Bad") ---
REWARD_FOOD = 30
REWARD_STEP = -0.01
REWARD_KILL = 40  # Big bonus for taking out an enemy
REWARD_COLLISION = -100
# Formula: final_penalty = REWARD_COLLISION - (score * REWARD_STARVE_MULTIPLIER)
# Example at Score 50: -15 - (50 * 0.5) = -40 penalty
REWARD_STARVE_MULTIPLIER = 0.5 

# --- MODEL ARCHITECTURE ---
# You can now change the brain size here without touching model.py
STACK_SIZE = 3               # How many frames to remember
INPUT_SIZE = 46 * STACK_SIZE  # 24 Ray + 4 Orientation + 2 Food Direction
HIDDEN_SIZE_0 = 512
HIDDEN_SIZE_1 = 256           
HIDDEN_SIZE_2 = 128           
HIDDEN_SIZE_3 = 64            
OUTPUT_SIZE = 3               # [Straight, Right, Left]

# TODO --- VISUALIZATION SETTINGS ---
VISUALIZE_NN = True      # Set to False to save FPS
NN_X = 660               # Draw NN starting at x=660 (You might need to increase WIDTH)
NN_Y = 50
NN_WIDTH = 300
NN_HEIGHT = 400

# --- DEVICE CONFIG ---
# Automatically picks GPU (cuda/mps) if available, otherwise CPU
DEVICE = 'cpu'
# DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'