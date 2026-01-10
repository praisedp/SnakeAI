import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple
import settings

pygame.init()
font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT, LEFT, UP, DOWN = 1, 2, 3, 4

Point = namedtuple('Point', 'x, y')

# --- Class to handle ONE snake's physics ---
class SnakeEntity:
    def __init__(self, name, color_head, color_body, start_x, start_y):
        self.name = name
        self.head_color = color_head
        self.body_color = color_body
        self.direction = Direction.RIGHT
        
        # Init body
        self.head = Point(start_x, start_y)
        self.body = [
            self.head,
            Point(self.head.x - settings.BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * settings.BLOCK_SIZE), self.head.y)
        ]
        self.score = 0
        self.is_alive = True

    def move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += settings.BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= settings.BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += settings.BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= settings.BLOCK_SIZE

        self.head = Point(x, y)

# --- THE GAME ENGINE ---
class SnakeGameAI:
    def __init__(self):
        self.w = settings.WIDTH
        self.h = settings.HEIGHT
        # Only start display if RENDER is True
        if settings.RENDER:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake AI')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.snakes = []
        # 1. Create Agents dynamically
        for i, agent_conf in enumerate(settings.AGENTS):

            mid_x = self.w // 2
            mid_y = self.h // 2
            
            # 2. SNAP TO GRID
            # We divide by block size, drop the remainder, then multiply back
            start_x = (mid_x // settings.BLOCK_SIZE) * settings.BLOCK_SIZE
            start_y = (mid_y // settings.BLOCK_SIZE) * settings.BLOCK_SIZE
            
            # 3. Apply Offset for multiple snakes
            offset = i * 2 * settings.BLOCK_SIZE
            start_y += offset

            new_snake = SnakeEntity(
                name=agent_conf["name"], 
                color_head=agent_conf["color_head"],
                color_body=agent_conf["color_body"],
                start_x=start_x, 
                start_y=start_y
            )
            self.snakes.append(new_snake)

        # 2. Spawn Multiple Foods
        self.food_list = []
        for _ in range(settings.NUM_FOOD):
            self._place_food()
            
        self.frame_iteration = 0

    def _place_food(self):
        while True:
            x = random.randint(0, (self.w - settings.BLOCK_SIZE) // settings.BLOCK_SIZE) * settings.BLOCK_SIZE
            y = random.randint(0, (self.h - settings.BLOCK_SIZE) // settings.BLOCK_SIZE) * settings.BLOCK_SIZE
            food = Point(x, y)
            
            # Check if food is inside ANY snake body
            occupied = False
            for snake in self.snakes:
                if food in snake.body:
                    occupied = True
                    break
            # Also check if food is on top of other food
            if food in self.food_list:
                occupied = True
                
            if not occupied:
                self.food_list.append(food)
                break

    def play_step(self, actions, render=True):
        self.frame_iteration += 1
        
        # 1. Handle Quit Event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. Move ALL Snakes
        # Note: 'actions' must be a list of moves, one for each snake
        for i, snake in enumerate(self.snakes):
            if snake.is_alive:
                snake.move(actions[i]) # Calls the fixed .move() method
                snake.body.insert(0, snake.head)

        # 3. Check Collisions & Rewards
        rewards = []
        dones = []
        
        for snake in self.snakes:
            # If already dead, give 0 reward and mark done
            if not snake.is_alive:
                rewards.append(0) 
                dones.append(True)
                continue

            reward = settings.REWARD_STEP
            game_over = False

            # Check collision 
            if self.is_collision(snake):
                game_over = True
                reward = settings.REWARD_COLLISION - (snake.score * settings.REWARD_STARVE_MULTIPLIER)
                snake.is_alive = False
            
            # Check Food (Modified for multi-food)
            elif self.check_food_collision(snake):
                snake.score += 1
                reward = settings.REWARD_FOOD
                # No pop() here, so snake grows
            else:
                snake.body.pop() # Move normally

            rewards.append(reward)
            dones.append(game_over)

        # 4. Update UI
        if settings.RENDER and render:
            self._update_ui()
            self.clock.tick(settings.SPEED)

        # 5. Global Done (Game over if ALL snakes are dead)
        all_done = all(dones)
        
        return rewards, dones, [s.score for s in self.snakes]

    # --- NEW: Helper to check if a specific snake hit food ---
    def check_food_collision(self, snake):
        if snake.head in self.food_list:
            self.food_list.remove(snake.head) # Eat the apple
            self._place_food() # Spawn a new one
            return True
        return False

    # --- NEW: Helper to check wall or body collisions ---
    def is_collision(self, snake):
        pt = snake.head
        # 1. Hits boundary
        if pt.x > self.w - settings.BLOCK_SIZE or pt.x < 0 or pt.y > self.h - settings.BLOCK_SIZE or pt.y < 0:
            return True
        
        # 2. Hits ITSELF
        if pt in snake.body[1:]:
            return True
            
        # 3. Hits OTHER snakes (Multi-Agent Logic)
        for other_snake in self.snakes:
            if other_snake != snake and other_snake.is_alive:
                if pt in other_snake.body:
                    return True
                    
        return False

    # --- NEW: Renders all snakes and food ---
    def _update_ui(self):
        self.display.fill(settings.COLOR_BG)

        # Draw all snakes
        for snake in self.snakes:
            if not snake.is_alive: continue # Optional: Don't draw dead snakes
            
            for pt in snake.body:
                pygame.draw.rect(self.display, snake.body_color, pygame.Rect(pt.x, pt.y, settings.BLOCK_SIZE, settings.BLOCK_SIZE))
                pygame.draw.rect(self.display, snake.head_color, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        # Draw all food
        for food in self.food_list:
            pygame.draw.rect(self.display, settings.COLOR_FOOD, pygame.Rect(food.x, food.y, settings.BLOCK_SIZE, settings.BLOCK_SIZE))

        # Draw Scores
        text_str = " | ".join([f"{s.name}: {s.score}" for s in self.snakes])
        text = font.render(text_str, True, settings.COLOR_TEXT)
        self.display.blit(text, [0, 0])
        
        pygame.display.flip()