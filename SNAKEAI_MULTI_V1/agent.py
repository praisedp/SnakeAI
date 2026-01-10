import os
import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

# --------------------
# Hyperparameters
# --------------------
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

# NOTE: consider number of self.n_games in init and self.epsilon

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9

        self.memory = deque(maxlen=MAX_MEMORY)

        # 30 inputs: 24 ray vision + 4 direction + 2 food direction
        self.model = Linear_QNet(38, 256, 3)
        self.target_model = Linear_QNet(38, 256, 3)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

          # Resume training rather than start form scratch
        if os.path.exists('./models/model.pth'):
            self.model.load_state_dict(torch.load('./models/model.pth'))
            self.model.eval() # Set to evaluation mode
            self.target_model.load_state_dict(self.model.state_dict())
            self.n_games = 2500 # Trick to disable random exploration immediately
            print(">> MODEL LOADED! Resuming with smart brain.")

   # --------------------------------------------------
    # Ray casting for vision (Updated with Log Scale)
    # --------------------------------------------------
    def cast_ray(self, game, start_point, direction, my_snake, enemy_snake):
        current = start_point
        distance = 0

        food_signal = 0.0
        my_body_signal = 0.0
        enemy_body_signal = 0.0

        # Correct max distance (grid-based, not diagonal)
        max_dist = max(
            game.w // game.blockSize,
            game.h // game.blockSize
        )
        log_max = np.log(max_dist + 1)

        while True:
            next_point = Point(
                current.x + direction[0] * game.blockSize,
                current.y + direction[1] * game.blockSize
            )

            # Wall check BEFORE stepping
            if next_point.x < 0 or next_point.x >= game.w or next_point.y < 0 or next_point.y >= game.h:
                break

            current = next_point
            distance += 1

            log_proximity = 1 - (np.log(distance + 1) / log_max)
            log_proximity = max(0.0, log_proximity)

            # Body detection (closest wins)
            if current in my_snake.body:
                my_body_signal = max(my_body_signal, log_proximity)

            if current in enemy_snake.body:
                enemy_body_signal = max(enemy_body_signal, log_proximity)

            # Food detection (closest wins)
            if current == game.food:
                food_signal = max(food_signal, log_proximity)

        # Wall signal based on true distance
        wall_signal = 1 - (np.log(distance + 1) / log_max)
        wall_signal = max(0.0, wall_signal)

        return [wall_signal, my_body_signal, enemy_body_signal, food_signal]


    # --------------------------------------------------
    # State representation
    # --------------------------------------------------
    # --------------------------------------------------
    # State representation (Updated to be Relative)
    # --------------------------------------------------
    # --------------------------------------------------
    # State representation (Updated to be Relative)
    # --------------------------------------------------
    def get_state(self, game, me, enemy):
        # ERROR WAS HERE: head = game.snake[0] 
        # FIX: Use 'me.head' because 'me' is the specific snake playing right now
        head = me.head 

        # Define the 8 compass points
        p_up    = (0, -1)
        p_down  = (0, 1)
        p_left  = (-1, 0)
        p_right = (1, 0)
        
        # Diagonals
        p_ul    = (-1, -1) 
        p_ur    = (1, -1)
        p_dl    = (-1, 1)
        p_dr    = (1, 1)

        # Rotate rays based on MY direction (not game.direction)
        if me.direction == Direction.UP:
            rays = [p_up, p_ur, p_right, p_dr, p_down, p_dl, p_left, p_ul]
        elif me.direction == Direction.RIGHT:
            rays = [p_right, p_dr, p_down, p_dl, p_left, p_ul, p_up, p_ur]
        elif me.direction == Direction.DOWN:
            rays = [p_down, p_dl, p_left, p_ul, p_up, p_ur, p_right, p_dr]
        elif me.direction == Direction.LEFT:
            rays = [p_left, p_ul, p_up, p_ur, p_right, p_dr, p_down, p_dl]

        state = []

        # 1. Vision (Relative Rays)
        for d in rays:
            state.extend(self.cast_ray(game, head, d, me, enemy))

        # 2. Orientation (One-Hot)
        # FIX: Use 'me.direction' instead of 'game.direction'
        state.extend([
            int(me.direction == Direction.LEFT),
            int(me.direction == Direction.RIGHT),
            int(me.direction == Direction.UP),
            int(me.direction == Direction.DOWN)
        ])

        # 3. Food Direction (Relative to Head)
        food_dx = game.food.x - head.x
        food_dy = game.food.y - head.y
        
        # FIX: Use 'me.direction' for relative rotation
        if me.direction == Direction.UP:
            rel_food_front = -food_dy 
            rel_food_right = food_dx
        elif me.direction == Direction.RIGHT:
            rel_food_front = food_dx
            rel_food_right = food_dy
        elif me.direction == Direction.DOWN:
            rel_food_front = food_dy
            rel_food_right = -food_dx
        elif me.direction == Direction.LEFT:
            rel_food_front = -food_dx
            rel_food_right = -food_dy

        state.extend([
            np.sign(rel_food_front),
            np.sign(rel_food_right)
        ])

        return np.array(state, dtype=float)
    # --------------------------------------------------
    # Memory
    # --------------------------------------------------
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # --------------------------------------------------
    # Training
    # --------------------------------------------------
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done, self.target_model)

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            batch = random.sample(self.memory, BATCH_SIZE)
        else:
            batch = self.memory

        states, actions, rewards, next_states, dones = zip(*batch)
        self.trainer.train_step(states, actions, rewards, next_states, dones, self.target_model)

    # --------------------------------------------------
    # Action selection (ε-greedy)
    # --------------------------------------------------
    def get_action(self, state):
        self.epsilon = 100 - self.n_games // 25
        if self.epsilon < 10:
             self.epsilon = 2

        final_move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
        else:
            state_t = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_t)
            move = torch.argmax(prediction).item()

        final_move[move] = 1
        return final_move


# --------------------------------------------------
# Training loop
# --------------------------------------------------
def train():
    scores = []
    mean_scores = []
    total_score = 0
    record = 0

    agent = Agent()
    game = SnakeGameAI()

    while True:
        # 1. Get States (Perspective Flip!)
        # P1 sees: "I am P1, Enemy is P2"
        state_p1 = agent.get_state(game, game.p1, game.p2)
        
        # P2 sees: "I am P2, Enemy is P1"
        state_p2 = agent.get_state(game, game.p2, game.p1)

        # 2. Get Actions (Same Brain)
        action_p1 = agent.get_action(state_p1)
        action_p2 = agent.get_action(state_p2)

        # 3. Play Step (Simultaneous)
        render_flag = (agent.n_games % 100 == 0) # Speed up
        
        # returns [r1, r2], [done1, done2], [score1, score2]
        rewards, dones, scores_game = game.play_step([action_p1, action_p2], render=True)
        
        # 4. Get New States
        state_new_p1 = agent.get_state(game, game.p1, game.p2)
        state_new_p2 = agent.get_state(game, game.p2, game.p1)

        # 5. Train Short Memory (Train BOTH perspectives)
        # The agent learns from P1's experience
        agent.train_short_memory(state_p1, action_p1, rewards[0], state_new_p1, dones[0])
        agent.remember(state_p1, action_p1, rewards[0], state_new_p1, dones[0])
        
        # The agent ALSO learns from P2's experience
        agent.train_short_memory(state_p2, action_p2, rewards[1], state_new_p2, dones[1])
        agent.remember(state_p2, action_p2, rewards[1], state_new_p2, dones[1])

        if any(dones): # If either died, reset
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if agent.n_games % 10 == 0: # Fast Target Update
                agent.target_model.load_state_dict(agent.model.state_dict())

            # Track P1's score for the graph (or max of both)
            current_score = max(scores_game) 
            
            if current_score > record:
                record = current_score
                agent.model.save()

            print(f"Game {agent.n_games}  MaxScore {current_score}  Record {record}")

            scores.append(current_score)
            total_score += current_score
            mean_scores.append(total_score / agent.n_games)
            plot(scores, mean_scores)

if __name__ == "__main__":
    train()