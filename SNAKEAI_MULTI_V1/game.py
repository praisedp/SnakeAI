import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.SysFont('arial', 25)

# Enum for directions to keep code readable
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# Named tuple for coordinates (makes it easier to access x and y)
Point = namedtuple('Point', 'x, y')

# RGB Colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
RED1 = (255, 0, 0)
RED2 = (255, 100, 100)
GREEN = (0, 255, 0) # Food
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 20  # Keep arround 20 for human viewing

class SnakeEntity:
    def __init__(self, w, h, is_player_one=True):
        self.w = w
        self.h = h
        self.is_player_one = is_player_one
        self.color1 = BLUE1 if is_player_one else RED1
        self.color2 = BLUE2 if is_player_one else RED2
        self.reset()

    def reset(self):
        # P1 starts Left, P2 starts Right
        if self.is_player_one:
            x = (self.w // 4 // BLOCK_SIZE) * BLOCK_SIZE
            self.direction = Direction.RIGHT
        else:
            x = (3 * self.w // 4 // BLOCK_SIZE) * BLOCK_SIZE
            self.direction = Direction.LEFT
            
        y = (self.h // 2 // BLOCK_SIZE) * BLOCK_SIZE
        self.head = Point(x, y)
        
        # Create body (3 blocks long)
        self.body = [
            self.head,
            Point(self.head.x - (BLOCK_SIZE if self.is_player_one else -BLOCK_SIZE), y),
            Point(self.head.x - (2 * BLOCK_SIZE if self.is_player_one else -2 * BLOCK_SIZE), y)
        ]
        
        self.score = 0
        self.dead = False

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
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

# ---------------------------------------------------------
# GAME ENGINE (The Referee)
# ---------------------------------------------------------
class SnakeGameAI:

    def __init__(self, w=900, h=880):
        self.w = w
        self.h = h
        self.blockSize = BLOCK_SIZE
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake Battle: Blue vs Red')
        self.clock = pygame.time.Clock()
        
        # Create Two Snakes
        self.p1 = SnakeEntity(self.w, self.h, is_player_one=True)
        self.p2 = SnakeEntity(self.w, self.h, is_player_one=False)
        
        self.reset()

    def reset(self):
        self.p1.reset()
        self.p2.reset()
        self.frame_iteration = 0
        self._place_food()

    def _place_food(self):
        while True:
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            self.food = Point(x, y)
            
            # Ensure food doesn't spawn on ANY snake
            if self.food not in self.p1.body and self.food not in self.p2.body:
                break

    def play_step(self, actions, render=True):
        # actions is now a list: [action_p1, action_p2]
        self.frame_iteration += 1
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 1. Move Both Snakes
        # Note: We move head but don't insert to body yet (we check collision first)
        self.p1.move(actions[0])
        self.p2.move(actions[1])
        
        self.p1.body.insert(0, self.p1.head)
        self.p2.body.insert(0, self.p2.head)

        # 2. Check Collisions
        reward1 = 0
        reward2 = 0
        done = False # Game ends if ANYONE dies (for now, simpler training)
        
        # Check P1 Collision (Walls, Own Body, Enemy Body)
        if self.is_collision(self.p1, self.p2):
            self.p1.dead = True
            reward1 = -15
            reward2 = 50 # P2 wins!
            done = True
            
        # Check P2 Collision
        if self.is_collision(self.p2, self.p1):
            self.p2.dead = True
            reward2 = -15
            if not self.p1.dead:
                reward1 = 50 # P1 wins!
            else:
                # Head-to-Head collision: Draw
                reward1 = -15 
                reward2 = -15
            done = True
        
        # Starvation (If game takes too long)
        if self.frame_iteration > 100 * max(len(self.p1.body), len(self.p2.body)):
            done = True
            reward1 = -10
            reward2 = -10

        if done:
            return [reward1, reward2], [done, done], [self.p1.score, self.p2.score]

        # 3. Check Food
        # P1 Eats
        if self.p1.head == self.food:
            self.p1.score += 1
            reward1 = 30
            self._place_food()
        else:
            self.p1.body.pop()

        # P2 Eats
        # (Handling rare case where food spawns under P2 head immediately after P1 eats?)
        # Or if both hit food same time (rare). For now, simple logic:
        if self.p2.head == self.food:
            self.p2.score += 1
            reward2 = 30
            self._place_food()
        else:
            self.p2.body.pop()

        # 4. Update UI
        if render:
            self._update_ui()
            self.clock.tick(SPEED)

        return [reward1, reward2], [done, done], [self.p1.score, self.p2.score]

    def is_collision(self, player, enemy):
        pt = player.head
        # 1. Hits Wall
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # 2. Hits Self
        if pt in player.body[1:]:
            return True
        # 3. Hits Enemy
        if pt in enemy.body:
            return True
        
        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        # Draw P1 (Blue)
        for pt in self.p1.body:
            pygame.draw.rect(self.display, self.p1.color1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, self.p1.color2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        # Draw P2 (Red)
        for pt in self.p2.body:
            pygame.draw.rect(self.display, self.p2.color1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, self.p2.color2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        # Draw Food
        pygame.draw.rect(self.display, GREEN, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render(f"Blue: {self.p1.score}  Red: {self.p2.score}", True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()