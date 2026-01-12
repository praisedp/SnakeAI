import os
import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
import math
from model import Linear_QNet, QTrainer
from helper import plot
import settings 

class Agent:

    def __init__(self):
        self._init_hyperparameters()
        self._init_memory()
        self._init_network()
        self._attempt_resume()

    # State representation
    def get_state(self, game, snake_entity):
        # 1. Collect Raw Data Components
        vision_state = self._get_vision_state(game, snake_entity)
        smell_state  = self._get_food_smell(game, snake_entity)
        context_state = self._get_context_data(game, snake_entity)
        
        # 2. Combine into 'Current Frame' (Size: 32)
        current_frame = np.concatenate([vision_state, smell_state, context_state])
        
        # 3. Frame Stacking (Memory)
        # We append this frame to the snake's individual history
        # (Assuming you added 'self.state_history = deque(...)' to SnakeEntity in game.py)
        snake_entity.state_history.append(current_frame)
        
        # 4. Flatten the stack (Size: 32 * 3 = 96)
        stacked_state = np.concatenate(list(snake_entity.state_history))
        
        return np.array(stacked_state, dtype=float)

    # --------------------------------------------------
    # Standard Interface
    # --------------------------------------------------
    def get_action(self, state):
        self.epsilon = settings.EPSILON_START - self.n_games // settings.EPSILON_DECAY
        if self.epsilon < settings.EPSILON_MIN:
             self.epsilon = settings.EPSILON_MIN

        final_move = [0, 0, 0]

        if random.randint(0, 100) < self.epsilon:
            move = random.randint(0, 2)
        else:
            # We predict based on the Stacked State (96 inputs)
            state_t = torch.tensor(state, dtype=torch.float).to(settings.DEVICE).unsqueeze(0)
            prediction = self.model(state_t)
            move = torch.argmax(prediction).item()

        final_move[move] = 1
        return final_move

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done, self.target_model)

    def train_long_memory(self):
        if len(self.memory) > settings.BATCH_SIZE:
            batch = random.sample(self.memory, settings.BATCH_SIZE)
        else:
            batch = self.memory

        states, actions, rewards, next_states, dones = zip(*batch)
        self.trainer.train_step(states, actions, rewards, next_states, dones, self.target_model)
   
    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------
    def _init_hyperparameters(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = settings.GAMMA

    def _init_memory(self):
        self.memory = deque(maxlen=settings.MAX_MEMORY)

    def _init_network(self):
        self.model = Linear_QNet()
        self.target_model = Linear_QNet()
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.trainer = QTrainer(self.model, lr=settings.LR, gamma=self.gamma)

    def _attempt_resume(self):
        if settings.LOAD_MODEL and os.path.exists(settings.MODEL_PATH):
            print(">> Loading Model...")
            saved_state = torch.load(settings.MODEL_PATH, map_location=settings.DEVICE)
            self.model.load_state_dict(saved_state)
            self.target_model.load_state_dict(saved_state)
            self.model.eval()
            self.n_games = settings.EPSILON_MEMORY_LOAD
            print(f">> Success! Resuming from approx game {self.n_games}")

    # --- VISION SYSTEM (Raycasts) ---
    def _get_vision_state(self, game, snake):
        head = snake.head
        
        # Define Relative Directions based on current facing
        # Order: Front, Front-Right, Right, Back-Right, Back, Back-Left, Left, Front-Left
        directions = self._get_relative_directions(snake.direction)
        
        vision_data = []
        for i, direction in enumerate(directions):
            # cast_ray returns [wall, body, enemy, food]
            result = self._cast_ray(game, snake, head, direction)
            
            # Optimization: 
            # Cardinal rays (Front, Right, Back, Left) -> Get FULL data
            # Diagonal rays -> Get BODY only (Walls are less useful diagonally)
            if i % 2 == 0: 
                vision_data.extend([result[0], result[1], result[2], result[3]]) # Add result[2] Enemy Body
            else:
                vision_data.append(result[1]) # My Body
                vision_data.append(result[2]) # Enemy Body
                vision_data.append(result[3]) # Food
                
        # Add Orientation One-Hot
        vision_data.extend([
            int(snake.direction == Direction.LEFT),
            int(snake.direction == Direction.RIGHT),
            int(snake.direction == Direction.UP),
            int(snake.direction == Direction.DOWN)
        ])
        
        return np.array(vision_data)

    def _cast_ray(self, game, snake, start, direction):
        current = start
        distance = 0
        
        # Signals
        wall_signal = 0.0
        my_body_signal = 0.0
        enemy_signal = 0.0
        food_signal = 0.0
        
        max_dist = max(game.w // settings.BLOCK_SIZE, game.h // settings.BLOCK_SIZE)
        log_max = np.log(max_dist + 1)

        while True:
            current = Point(
                current.x + direction[0] * settings.BLOCK_SIZE,
                current.y + direction[1] * settings.BLOCK_SIZE
            )

            # 1. Wall Check
            if (current.x < 0 or current.x >= game.w or 
                current.y < 0 or current.y >= game.h):
                break

            distance += 1
            
            # Logarithmic Drop-off
            signal = max(0.0, 1 - (np.log(distance + 1) / log_max))

            # 2. Body Check (Split into My Body vs Enemy)
            for s in game.snakes:
                if s.is_alive and current in s.body:
                    if s == snake:
                        # It's ME (Don't hit yourself, but you can loop)
                        my_body_signal = max(my_body_signal, signal)
                    else:
                        # It's ENEMY (Target for kills, or avoid head)
                        enemy_signal = max(enemy_signal, signal)

            # 3. Food Check
            if current in game.food_list:
                food_signal = max(food_signal, signal)

        wall_signal = max(0.0, 1 - (np.log(distance + 1) / log_max))
        
        # Return 4 values now
        return [wall_signal, my_body_signal, enemy_signal, food_signal]
    
    def _get_relative_directions(self, direction):
        """Returns list of 8 direction vectors relative to the snake's facing"""
        # Standard Compass
        dirs = [
            (0, -1), (1, -1), (1, 0), (1, 1), 
            (0, 1), (-1, 1), (-1, 0), (-1, -1)
        ] # Up, UR, Right, DR, Down, DL, Left, UL
        
        # Rotate logic
        if direction == Direction.UP:    idx = 0
        elif direction == Direction.RIGHT: idx = 2
        elif direction == Direction.DOWN:  idx = 4
        elif direction == Direction.LEFT:  idx = 6
        
        # Shift the list so 'Front' is always first
        # We need 8 directions. Slicing with modulo is tricky, 
        # easier to just manually reorder or use deque rotation.
        # Simple manual map for speed:
        if direction == Direction.UP:
            return dirs # Front is Up (0,-1)
        elif direction == Direction.RIGHT:
            # Front is (1,0) -> Index 2
            return dirs[2:] + dirs[:2]
        elif direction == Direction.DOWN:
            return dirs[4:] + dirs[:4]
        elif direction == Direction.LEFT:
            return dirs[6:] + dirs[:6]
    
    # --- NEW: SMELL SYSTEM (Sector Based) ---
    def _get_food_smell(self, game, snake):
        # 1. Initialize 8 Sectors (0.0 means no food)
        # 0=Front, 1=Front-Right, 2=Right, 3=Back-Right
        # 4=Back,  5=Back-Left,   6=Left,  7=Front-Left
        sectors = [0.0] * 8
        
        # 2. Loop through ALL food (Variable number doesn't matter!)
        max_diag = np.sqrt(game.w**2 + game.h**2)
        
        for food in game.food_list:
            raw_dx = food.x - snake.head.x
            raw_dy = food.y - snake.head.y
            
            # A. Rotate to be Relative to Head (Ego-centric)
            # After this: +X is Front, +Y is Right
            rel_front, rel_right = self._rotate_coordinates(snake.direction, raw_dx, raw_dy)
            
            # B. Calculate Distance & Signal
            dist = np.sqrt(rel_front**2 + rel_right**2)
            signal = 1.0 - (dist / max_diag) # 1.0 = Touching, 0.0 = Far
            
            # C. Calculate Angle (0 radians is Front)
            # atan2(y, x) -> result is between -pi and +pi
            angle = math.atan2(rel_right, rel_front) 
            
            # D. Map Angle to Sector Index (0-7)
            # Convert radians to degrees (-180 to 180) -> shift to (0 to 360)
            degree = math.degrees(angle)
            
            # Shift so that 'Front' (0 degrees) is in the middle of a slice
            # Slice 0 covers -22.5 to +22.5 degrees
            idx = int(((degree + 22.5) % 360) // 45)
            
            # E. Update the Sector (Keep the Strongest Signal)
            sectors[idx] = max(sectors[idx], signal)

        # 3. Total Food Count (Normalized)
        # Using softsign normalization (Count 10 = 0.5)
        count_norm = len(game.food_list) / (len(game.food_list) + 10.0)
        
        # Combine: 8 Directional Inputs + 1 Count Input
        return np.array(sectors + [count_norm])

    def _get_target_gps(self, game, snake, targets):
        """Calculates distance and relative angle to closest targets"""
        head = snake.head
        state = []
        
        # Sort targets by distance
        distances = []
        for t in targets:
            dx = t.x - head.x
            dy = t.y - head.y
            dist = np.sqrt(dx**2 + dy**2)
            distances.append((dist, dx, dy))
        distances.sort(key=lambda x: x[0])
        
        # Get Closest Target (K=1)
        if distances:
            dist, raw_dx, raw_dy = distances[0]
            rel_front, rel_right = self._rotate_coordinates(snake.direction, raw_dx, raw_dy)
            
            max_diag = np.sqrt(game.w**2 + game.h**2)
            state = [
                rel_front / game.h,
                rel_right / game.w,
                1 - (dist / max_diag)
            ]
        else:
            state = [0, 0, 0]
            
        return np.array(state)

    def _rotate_coordinates(self, direction, dx, dy):
        """Rotates global dx/dy into snake's local 'Front/Right' space"""
        if direction == Direction.UP:
            return -dy, dx # Front is -y, Right is +x
        elif direction == Direction.RIGHT:
            return dx, dy  # Front is +x, Right is +y
        elif direction == Direction.DOWN:
            return dy, -dx # Front is +y, Right is -x
        elif direction == Direction.LEFT:
            return -dx, -dy # Front is -x, Right is -y
        return 0, 0

    # --- CONTEXT SYSTEM ---
    def _get_context_data(self, game, snake):
        # 1. Length
        norm_len = len(snake.body) / (len(snake.body) + 20.0)
        
        # 2. Hunger
        limit = settings.STARVE_LIMIT * len(snake.body)
        norm_hunger = snake.frame_iteration / (limit if limit > 0 else 100)
        
        # 3. Tail GPS (Use the helper!)
        tail_gps = self._get_target_gps(game, snake, [snake.body[-1]])
        
        return np.concatenate([[norm_len, norm_hunger], tail_gps])
