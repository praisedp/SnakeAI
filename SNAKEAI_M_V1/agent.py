import os
import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
import settings 

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = settings.GAMMA 

        self.memory = deque(maxlen=settings.MAX_MEMORY)

        # Dynamic Model Size from Settings
        # (Input=30, Hidden1=256, Hidden2=128, Hidden3=64, Output=3)
        self.model = Linear_QNet() 
        self.target_model = Linear_QNet()
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        
        self.trainer = QTrainer(self.model, lr=settings.LR, gamma=self.gamma)

        # Resume training logic
        if settings.LOAD_MODEL and os.path.exists(settings.MODEL_PATH):
            self.model.load_state_dict(torch.load(settings.MODEL_PATH))
            self.model.eval()
            self.target_model.load_state_dict(self.model.state_dict())
            self.n_games = 2500  # Trick to reduce epsilon immediately
            print(">> MODEL LOADED! Resuming with smart brain.")

    # --------------------------------------------------
    # Ray casting for vision (Your Original Logic)
    # --------------------------------------------------
    def cast_ray(self, game, snake_entity, start_point, direction):
        # NOTE: Added 'snake_entity' arg so it knows which body to check
        current = start_point
        distance = 0

        food_signal = 0.0
        body_signal = 0.0

        # Use game dimensions from settings
        max_dist = max(
            game.w // settings.BLOCK_SIZE,
            game.h // settings.BLOCK_SIZE
        )
        log_max = np.log(max_dist + 1)

        while True:
            next_point = Point(
                current.x + direction[0] * settings.BLOCK_SIZE,
                current.y + direction[1] * settings.BLOCK_SIZE
            )

            # Wall check
            if next_point.x < 0 or next_point.x >= game.w or next_point.y < 0 or next_point.y >= game.h:
                break

            current = next_point
            distance += 1

            log_proximity = 1 - (np.log(distance + 1) / log_max)
            log_proximity = max(0.0, log_proximity)

            # Body detection (Check THIS snake's body)
            if current in snake_entity.body:
                body_signal = max(body_signal, log_proximity)

            # Food detection (Check GLOBAL food list)
            if current in game.food_list:
                food_signal = max(food_signal, log_proximity)

        # Wall signal
        wall_signal = 1 - (np.log(distance + 1) / log_max)
        wall_signal = max(0.0, wall_signal)

        return [wall_signal, body_signal, food_signal]

    # --------------------------------------------------
    # State representation
    # --------------------------------------------------
    def get_state(self, game, snake_entity):
        # NOTE: Added 'snake_entity' arg
        head = snake_entity.head

        # Define 8 compass points
        p_up    = (0, -1)
        p_down  = (0, 1)
        p_left  = (-1, 0)
        p_right = (1, 0)
        p_ul    = (-1, -1)
        p_ur    = (1, -1)
        p_dl    = (-1, 1)
        p_dr    = (1, 1)

        # Relative Rays based on direction
        if snake_entity.direction == Direction.UP:
            rays = [p_up, p_ur, p_right, p_dr, p_down, p_dl, p_left, p_ul]
        elif snake_entity.direction == Direction.RIGHT:
            rays = [p_right, p_dr, p_down, p_dl, p_left, p_ul, p_up, p_ur]
        elif snake_entity.direction == Direction.DOWN:
            rays = [p_down, p_dl, p_left, p_ul, p_up, p_ur, p_right, p_dr]
        elif snake_entity.direction == Direction.LEFT:
            rays = [p_left, p_ul, p_up, p_ur, p_right, p_dr, p_down, p_dl]

        state = []

        # NOTE: 1. Vision (Pass snake_entity to cast_ray)
        for d in rays:
            state.extend(self.cast_ray(game, snake_entity, head, d))

        # NOTE: 2. Orientation (One-Hot)
        state.extend([
            int(snake_entity.direction == Direction.LEFT),
            int(snake_entity.direction == Direction.RIGHT),
            int(snake_entity.direction == Direction.UP),
            int(snake_entity.direction == Direction.DOWN)
        ])

        # NOTE: 3. Food Direction (Relative to Head)
        # Find closest food if multiple exist
        closest_food = None
        min_dist = float('inf')
        
        if len(game.food_list) > 0:
            for food in game.food_list:
                dist = abs(food.x - head.x) + abs(food.y - head.y)
                if dist < min_dist:
                    min_dist = dist
                    closest_food = food
        else:
            closest_food = Point(0,0) # Should not happen

        food_dx = closest_food.x - head.x
        food_dy = closest_food.y - head.y
        
        # Rotate food coords
        if snake_entity.direction == Direction.UP:
            rel_food_front = -food_dy 
            rel_food_right = food_dx
        elif snake_entity.direction == Direction.RIGHT:
            rel_food_front = food_dx
            rel_food_right = food_dy
        elif snake_entity.direction == Direction.DOWN:
            rel_food_front = food_dy
            rel_food_right = -food_dx
        elif snake_entity.direction == Direction.LEFT:
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
        if len(self.memory) > settings.BATCH_SIZE:
            batch = random.sample(self.memory, settings.BATCH_SIZE)
        else:
            batch = self.memory

        states, actions, rewards, next_states, dones = zip(*batch)
        self.trainer.train_step(states, actions, rewards, next_states, dones, self.target_model)

    # --------------------------------------------------
    # Action selection
    # --------------------------------------------------
    def get_action(self, state):
        self.epsilon = settings.EPSILON_START - self.n_games // settings.EPSILON_DECAY
        if self.epsilon < settings.EPSILON_MIN:
             self.epsilon = settings.EPSILON_MIN

        final_move = [0, 0, 0]

        # Use settings.EPSILON_START (80) as the scale for randint(0, 200) logic?
        # Keeping your exact original logic: randint(0, 200) < epsilon
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
        else:
            state_t = torch.tensor(state, dtype=torch.float).to(settings.DEVICE).unsqueeze(0)
            prediction = self.model(state_t)
            move = torch.argmax(prediction).item()

        final_move[move] = 1
        return final_move


# --------------------------------------------------
# Training Loop (Updated for Multi-Agent Lists)
# --------------------------------------------------
def train():
    scores = []
    mean_scores = []
    total_score = 0
    record = 0

    agent = Agent()
    game = SnakeGameAI()

    while True:
        # -----------------------------------------
        # 1. Get State (For the FIRST snake only)
        # -----------------------------------------
        # Since we are testing single agent, we grab game.snakes[0]
        snake_entity = game.snakes[0] 
        state_old = agent.get_state(game, snake_entity)
        
        # 2. Get Action
        action = agent.get_action(state_old)
        
        # 3. Play Step (Pass Action as a LIST)
        # Game expects a list of actions: [action_for_snake_0]
        rewards, dones, current_scores = game.play_step([action])
        
        # Extract data for the first snake
        reward = rewards[0]
        done = dones[0]      # This is specifically if snake[0] died
        score = current_scores[0]
        
        # 4. Get New State
        state_new = agent.get_state(game, snake_entity)

        # 5. Train Short Memory
        agent.train_short_memory(state_old, action, reward, state_new, done)
        agent.remember(state_old, action, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if agent.n_games % settings.TARGET_UPDATE_SIZE == 0:
                agent.target_model.load_state_dict(agent.model.state_dict())
                print(">>> Target Network Updated!")

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