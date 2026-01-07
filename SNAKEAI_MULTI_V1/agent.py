import torch
import random
import numpy as np
from collections import deque
from game import MultiSnakeGame, Direction, Point 
from model import Linear_QNet, QTrainer
from helper import plot
import os

# --------------------
# Hyperparameters
# --------------------
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9

        self.memory = deque(maxlen=MAX_MEMORY)

        # 30 INPUTS:
        # 8 Rays * 3 Signals (Wall, Body/Enemy, Food) = 24
        # 4 Direction flags = 4
        # 2 Food Relative flags = 2
        # TOTAL = 30
        self.model = Linear_QNet(30, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        
        # Load existing model if available (for continuing training)
        if os.path.exists('./models/model.pth'):
            try:
                self.model.load_state_dict(torch.load('./models/model.pth'))
                self.model.eval()
                print(">> MODEL LOADED! Continuing training...")
            except:
                print(">> Model structure mismatch or error. Starting fresh.")

    # --------------------------------------------------
    # Ray Casting (Modified for Multi-Agent)
    # --------------------------------------------------
    def cast_ray(self, game, start_point, direction, player_id):
        # 1. Identify "Self" and "Enemy" based on ID
        if player_id == 1:
            my_snake = game.snake1
            enemy_snake = game.snake2
        else:
            my_snake = game.snake2
            enemy_snake = game.snake1

        current = start_point
        distance = 0
        
        # Signals to return
        food_signal = 0.0
        body_signal = 0.0 # Represents 'Lethal Obstacle' (My Body OR Enemy Body)

        # Logarithmic Scale Setup
        max_dist = max(game.w // BLOCK_SIZE, game.h // BLOCK_SIZE)
        log_max = np.log(max_dist + 1)

        while True:
            # Move the ray forward
            next_point = Point(
                current.x + direction[0] * BLOCK_SIZE,
                current.y + direction[1] * BLOCK_SIZE
            )

            # A. Check Wall Collision
            if next_point.x < 0 or next_point.x >= game.w or next_point.y < 0 or next_point.y >= game.h:
                break

            current = next_point
            distance += 1

            # Calculate "Proximity Score" (1.0 = Close, 0.0 = Far)
            log_proximity = 1 - (np.log(distance + 1) / log_max)
            log_proximity = max(0.0, log_proximity)

            # B. Check Body Collision (ME OR ENEMY)
            # We treat the enemy body exactly like our own - a deadly obstacle.
            if current in my_snake or current in enemy_snake:
                body_signal = max(body_signal, log_proximity)

            # C. Check Food
            if current == game.food:
                food_signal = max(food_signal, log_proximity)

        # Wall Signal based on final distance
        wall_signal = 1 - (np.log(distance + 1) / log_max)
        wall_signal = max(0.0, wall_signal)

        return [wall_signal, body_signal, food_signal]

    # --------------------------------------------------
    # Get State (Relative to specific Player ID)
    # --------------------------------------------------
    def get_state(self, game, player_id):
        # Identify Head & Direction for the specific player
        if player_id == 1:
            head = game.head1
            direction = game.direction1
        else:
            head = game.head2
            direction = game.direction2

        # ----------------------------------------------
        # 1. Relative Directions (Compass Rotation)
        # ----------------------------------------------
        p_up    = (0, -1)
        p_down  = (0, 1)
        p_left  = (-1, 0)
        p_right = (1, 0)
        p_ul    = (-1, -1)
        p_ur    = (1, -1)
        p_dl    = (-1, 1)
        p_dr    = (1, 1)

        # Rotate rays based on current facing direction
        # Order: [Front, Front-Right, Right, Back-Right, Back, Back-Left, Left, Front-Left]
        if direction == Direction.UP:
            rays = [p_up, p_ur, p_right, p_dr, p_down, p_dl, p_left, p_ul]
        elif direction == Direction.RIGHT:
            rays = [p_right, p_dr, p_down, p_dl, p_left, p_ul, p_up, p_ur]
        elif direction == Direction.DOWN:
            rays = [p_down, p_dl, p_left, p_ul, p_up, p_ur, p_right, p_dr]
        elif direction == Direction.LEFT:
            rays = [p_left, p_ul, p_up, p_ur, p_right, p_dr, p_down, p_dl]

        state = []

        # ----------------------------------------------
        # 2. Vision Input (24 values)
        # ----------------------------------------------
        for d in rays:
            state.extend(self.cast_ray(game, head, d, player_id))

        # ----------------------------------------------
        # 3. Orientation Input (4 values)
        # ----------------------------------------------
        state.extend([
            int(direction == Direction.LEFT),
            int(direction == Direction.RIGHT),
            int(direction == Direction.UP),
            int(direction == Direction.DOWN)
        ])

        # ----------------------------------------------
        # 4. Relative Food Direction (2 values)
        # ----------------------------------------------
        food_dx = game.food.x - head.x
        food_dy = game.food.y - head.y
        
        # Rotate food coordinates to be "Head-Centric"
        rel_front = 0
        rel_right = 0
        
        if direction == Direction.UP:
            rel_front, rel_right = -food_dy, food_dx
        elif direction == Direction.RIGHT:
            rel_front, rel_right = food_dx, food_dy
        elif direction == Direction.DOWN:
            rel_front, rel_right = food_dy, -food_dx
        elif direction == Direction.LEFT:
            rel_front, rel_right = -food_dx, -food_dy
            
        state.extend([np.sign(rel_front), np.sign(rel_right)])

        return np.array(state, dtype=float)

    # --------------------------------------------------
    # Standard Memory & Training Functions
    # --------------------------------------------------
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            batch = random.sample(self.memory, BATCH_SIZE)
        else:
            batch = self.memory

        states, actions, rewards, next_states, dones = zip(*batch)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def get_action(self, state):
        # Epsilon Decay (Randomness)
        self.epsilon = 15 - self.n_games 
        
        # Floor epsilon at 5 so it never stops experimenting completely
        if self.epsilon < 5: 
            self.epsilon = 5
            
        final_move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()

        final_move[move] = 1
        return final_move

# --------------------------------------------------
# Multi-Agent Training Loop
# --------------------------------------------------
def train():
    scores = []
    mean_scores = []
    total_score = 0
    record = 0 # Track combined max score or just max turns

    # CONSTANTS needed for Agent class reference
    global BLOCK_SIZE
    BLOCK_SIZE = 20

    agent = Agent()
    game = MultiSnakeGame()

    while True:
        # 1. Get Old States (for both snakes)
        state_old_1 = agent.get_state(game, player_id=1)
        state_old_2 = agent.get_state(game, player_id=2)

        # 2. Get Moves (Agent plays as both!)
        action_1 = agent.get_action(state_old_1)
        action_2 = agent.get_action(state_old_2)

        # 3. Perform Moves
        r1, r2, done, score1, score2 = game.play_step(action_1, action_2)

        # 4. Get New States
        state_new_1 = agent.get_state(game, player_id=1)
        state_new_2 = agent.get_state(game, player_id=2)

        # 5. Train Short Term (Teach BOTH experiences)
        agent.train_short_memory(state_old_1, action_1, r1, state_new_1, done)
        agent.train_short_memory(state_old_2, action_2, r2, state_new_2, done)

        # 6. Remember (Store BOTH experiences)
        agent.remember(state_old_1, action_1, r1, state_new_1, done)
        agent.remember(state_old_2, action_2, r2, state_new_2, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            # Optional: Save if the sum of scores is high
            combined_score = score1 + score2
            if combined_score > record:
                record = combined_score
                agent.model.save()

            print(f"Game {agent.n_games}  Blue: {score1} Red: {score2}  Record: {record}")

            # Plotting (Optional - plotting combined score)
            scores.append(combined_score)
            total_score += combined_score
            mean_scores.append(total_score / agent.n_games)
            plot(scores, mean_scores)

if __name__ == "__main__":
    train()