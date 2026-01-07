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
BLOCK_SIZE = 20

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9

        self.memory = deque(maxlen=MAX_MEMORY)

        # 30 inputs: 24 ray vision + 4 direction + 2 food direction
        self.model = Linear_QNet(30, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

          # Resume training rather than start form scratch
        if os.path.exists('./models/model.pth'):
            self.model.load_state_dict(torch.load('./models/model.pth'))
            self.model.eval() # Set to evaluation mode
            self.n_games = 200 # Trick to disable random exploration immediately
            print(">> MODEL LOADED! Resuming with smart brain.")

   # --------------------------------------------------
    # Ray casting for vision (Updated with Log Scale)
    # --------------------------------------------------
    def cast_ray(self, game, start_point, direction):
        current = start_point
        distance = 0

        food_signal = 0.0
        body_signal = 0.0

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
            if current in game.snake:
                body_signal = max(body_signal, log_proximity)

            # Food detection (closest wins)
            if current == game.food:
                food_signal = max(food_signal, log_proximity)

        # Wall signal based on true distance
        wall_signal = 1 - (np.log(distance + 1) / log_max)
        wall_signal = max(0.0, wall_signal)

        return [wall_signal, body_signal, food_signal]


    # --------------------------------------------------
    # State representation
    # --------------------------------------------------
    def get_state(self, game):
        head = game.snake[0]

        # 8 directions (ray casting)
        directions = [
            (0, -1),   # up
            (1, -1),   # up-right
            (1, 0),    # right
            (1, 1),    # down-right
            (0, 1),    # down
            (-1, 1),   # down-left
            (-1, 0),   # left
            (-1, -1)   # up-left
        ]

        state = []

        # Ray vision (24 values)
        for d in directions:
            state.extend(self.cast_ray(game, head, d))

        # Direction one-hot (4)
        state.extend([
            int(game.direction == Direction.LEFT),
            int(game.direction == Direction.RIGHT),
            int(game.direction == Direction.UP),
            int(game.direction == Direction.DOWN)
        ])

        # Food direction fallback (2)
        food_dx = game.food.x - head.x
        food_dy = game.food.y - head.y

        state.extend([
            np.sign(food_dx),
            np.sign(food_dy)
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
        self.trainer.train_step(state, action, reward, next_state, done)

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            batch = random.sample(self.memory, BATCH_SIZE)
        else:
            batch = self.memory

        states, actions, rewards, next_states, dones = zip(*batch)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    # --------------------------------------------------
    # Action selection (ε-greedy)
    # --------------------------------------------------
    def get_action(self, state):
        self.epsilon = 200 - self.n_games
        final_move = [0, 0, 0]

        # The Random factor
        if self.epsilon < 10: 
            self.epsilon = 0 # Keep it fixed at like 10

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
        state_old = agent.get_state(game)
        action = agent.get_action(state_old)

        reward, done, score = game.play_step(action)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, action, reward, state_new, done)
        agent.remember(state_old, action, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print(f"Game {agent.n_games}  Score {score}  Record {record}")

            scores.append(score)
            total_score += score
            mean_scores.append(total_score / agent.n_games)
            plot(scores, mean_scores)


if __name__ == "__main__":
    train()
