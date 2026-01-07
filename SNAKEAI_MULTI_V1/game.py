import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.SysFont('arial', 25)
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# ---------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------
BLOCK_SIZE = 20
SPEED = 3000  # High speed for training
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)

class MultiSnakeGame:

    def __init__(self, w=880, h=800):
        self.w = w
        self.h = h
        # Init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake AI Multi-Agent')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # -----------------------------------------------------
        # PLAYER 1 (Blue) - Starts Left side, moving Right
        # -----------------------------------------------------
        self.direction1 = Direction.RIGHT
        x1 = (self.w // 4 // BLOCK_SIZE) * BLOCK_SIZE
        y1 = (self.h // 2 // BLOCK_SIZE) * BLOCK_SIZE
        self.head1 = Point(x1, y1)
        self.snake1 = [
            self.head1,
            Point(self.head1.x - BLOCK_SIZE, self.head1.y),
            Point(self.head1.x - (2 * BLOCK_SIZE), self.head1.y)
        ]
        self.score1 = 0

        # -----------------------------------------------------
        # PLAYER 2 (Red) - Starts Right side, moving Left
        # -----------------------------------------------------
        self.direction2 = Direction.LEFT
        x2 = (3 * self.w // 4 // BLOCK_SIZE) * BLOCK_SIZE
        y2 = (self.h // 2 // BLOCK_SIZE) * BLOCK_SIZE
        self.head2 = Point(x2, y2)
        self.snake2 = [
            self.head2,
            Point(self.head2.x + BLOCK_SIZE, self.head2.y),
            Point(self.head2.x + (2 * BLOCK_SIZE), self.head2.y)
        ]
        self.score2 = 0

        # Shared Game State
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        # Randomly place food
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE - 1) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE - 1) * BLOCK_SIZE
        self.food = Point(x, y)
        
        # Recursive check: Ensure food doesn't spawn on top of ANY snake
        if self.food in self.snake1 or self.food in self.snake2:
            self._place_food()

    def play_step(self, action1, action2):
        self.frame_iteration += 1
        
        # -----------------------------------------------------
        # 1. MOVEMENT PHASE
        # -----------------------------------------------------
        # Move Snake 1
        self._move(action1, 1) 
        self.snake1.insert(0, self.head1)
        
        # Move Snake 2
        self._move(action2, 2) 
        self.snake2.insert(0, self.head2)
        
        # Default Rewards (Cost of Living)
        reward1 = -0.01
        reward2 = -0.01
        game_over = False

        # -----------------------------------------------------
        # 2. COLLISION PHASE (Death Checks)
        # -----------------------------------------------------
        # Check collision for Snake 1 (vs Wall, Own Body, Enemy Body)
        # We pass 'True' for is_snake1 to distinguish self vs enemy
        collision1 = self.is_collision(self.head1, self.snake1, self.snake2)
        
        # Check collision for Snake 2
        collision2 = self.is_collision(self.head2, self.snake2, self.snake1)

        # Timeout Check (Prevent infinite loops)
        # If they just spin in circles for too long without eating, kill both.
        timeout = self.frame_iteration > 100 * len(self.snake1)

        if collision1 or collision2 or timeout:
            game_over = True
            
            # SCENARIO: Both die (Head-on-Head or Simultaneous crash)
            if (collision1 and collision2) or timeout:
                reward1 = -10
                reward2 = -10
            
            # SCENARIO: Snake 1 Died (Hit Wall/Self/Enemy)
            elif collision1:
                reward1 = -10
                reward2 = +20 # Snake 2 Wins (Survival Bonus)
                
            # SCENARIO: Snake 2 Died
            elif collision2:
                reward2 = -10
                reward1 = +20 # Snake 1 Wins (Survival Bonus)
                
            return reward1, reward2, game_over, self.score1, self.score2

        # -----------------------------------------------------
        # 3. EATING PHASE
        # -----------------------------------------------------
        
        # Did Snake 1 eat?
        if self.head1 == self.food:
            self.score1 += 1
            reward1 = 10        # Reward for eating
            reward2 = -5        # PUNISHMENT for letting opponent eat
            self._place_food()
            # Do NOT pop tail (Snake 1 grows)
            # Snake 2 did NOT eat, so pop its tail (maintain length)
            self.snake2.pop()

        # Did Snake 2 eat?
        elif self.head2 == self.food:
            self.score2 += 1
            reward2 = 10        # Reward for eating
            reward1 = -5        # PUNISHMENT for letting opponent eat
            self._place_food()
            # Snake 2 grows (don't pop)
            # Snake 1 did NOT eat, so pop its tail
            self.snake1.pop()
            
        else:
            # Nobody ate: Both move normally (remove tails)
            self.snake1.pop()
            self.snake2.pop()

        # -----------------------------------------------------
        # 4. UI UPDATE
        # -----------------------------------------------------
        self._update_ui()
        self.clock.tick(SPEED)

        return reward1, reward2, game_over, self.score1, self.score2

    def is_collision(self, head, my_body, enemy_body):
        # 1. Hits Boundary
        if head.x > self.w - BLOCK_SIZE or head.x < 0 or head.y > self.h - BLOCK_SIZE or head.y < 0:
            return True
        
        # 2. Hits Own Body
        # (Start from index 1 because head is at index 0)
        if head in my_body[1:]:
            return True
            
        # 3. Hits Enemy Body
        # Hitting ANY part of the enemy is death
        if head in enemy_body:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        # Draw Snake 1 (Blue)
        for pt in self.snake1:
            pygame.draw.rect(self.display, BLUE, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, WHITE, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        # Draw Snake 2 (Red)
        for pt in self.snake2:
            pygame.draw.rect(self.display, RED, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            # Different inner color to distinguish
            pygame.draw.rect(self.display, (255, 100, 100), pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        # Draw Food
        pygame.draw.rect(self.display, GREEN, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # Draw Scores
        text = font.render(f"Blue: {self.score1}  Red: {self.score2}", True, WHITE)
        self.display.blit(text, [0, 0])
        
        pygame.display.flip()

    def _move(self, action, snake_id):
        # [Straight, Right, Left]
        
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        
        # Determine current direction
        if snake_id == 1:
            current_dir = self.direction1
        else:
            current_dir = self.direction2
            
        idx = clock_wise.index(current_dir)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn

        # Update direction state
        if snake_id == 1:
            self.direction1 = new_dir
        else:
            self.direction2 = new_dir

        # Calculate new head position
        x = self.head1.x if snake_id == 1 else self.head2.x
        y = self.head1.y if snake_id == 1 else self.head2.y

        if new_dir == Direction.RIGHT:
            x += BLOCK_SIZE
        elif new_dir == Direction.LEFT:
            x -= BLOCK_SIZE
        elif new_dir == Direction.DOWN:
            y += BLOCK_SIZE
        elif new_dir == Direction.UP:
            y -= BLOCK_SIZE

        # Update head state
        if snake_id == 1:
            self.head1 = Point(x, y)
        else:
            self.head2 = Point(x, y)