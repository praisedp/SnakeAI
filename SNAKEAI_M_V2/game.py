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
    def __init__(self, name, color, start_x, start_y):
        self.name = name
        self.color = color
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
        self.frame_iteration = 0

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
            start_y = ((start_y + offset) % self.h // settings.BLOCK_SIZE) * settings.BLOCK_SIZE

            new_snake = SnakeEntity(
                name=agent_conf["name"], 
                color=agent_conf["color"],
                start_x=start_x, 
                start_y=start_y
            )
            self.snakes.append(new_snake)

        # 2. Spawn Multiple Foods
        self.food_list = []
        for _ in range(settings.NUM_FOOD):
            if not self._place_food():
                break
            
        self.frame_iteration = 0

    def _place_food(self):
        occupied = {(pt.x, pt.y) for pt in self.food_list}
        for snake in self.snakes:
            occupied.update((pt.x, pt.y) for pt in snake.body)

        candidates = [
            (x, y)
            for x in range(0, self.w, settings.BLOCK_SIZE)
            for y in range(0, self.h, settings.BLOCK_SIZE)
            if (x, y) not in occupied
        ]

        if not candidates:
            return False

        x, y = random.choice(candidates)
        self.food_list.append(Point(x, y))
        return True

    def play_step(self, actions, render=True):
        self.frame_iteration += 1
        # Ensure we have one action per snake; pad with "straight" if missing
        if len(actions) < len(self.snakes):
            missing = len(self.snakes) - len(actions)
            actions = list(actions) + [[1, 0, 0] for _ in range(missing)]
        
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

            snake.frame_iteration += 1
            reward = settings.REWARD_STEP
            game_over = False

            # Check collision and starvation
            if self.is_collision(snake) or snake.frame_iteration > settings.STARVE_LIMIT * len(snake.body):
                game_over = True
                reward = settings.REWARD_COLLISION - (snake.score * settings.REWARD_STARVE_MULTIPLIER)
                snake.is_alive = False
            
            # Check Food (Modified for multi-food)
            elif self.check_food_collision(snake):
                snake.score += 1
                reward = settings.REWARD_FOOD
                snake.frame_iteration = 0
                # No pop() here, so snake grows
            else:
                snake.body.pop() # Move normally

            rewards.append(reward)
            dones.append(game_over)

        # Keep apples from disappearing due to an unexpected removal
        while len(self.food_list) < settings.NUM_FOOD:
            if not self._place_food():
                break

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
# ------------------------------------------------------------------
    # NEW: Helper for Color Gradients
    # ------------------------------------------------------------------
    def _get_gradient_color(self, start_rgb, end_rgb, position, total_length):
        """
        Calculates the color of a specific body segment.
        - position 0 (Head) -> start_rgb (White)
        - position Last (Tail) -> end_rgb (Settings Body Color)
        - positions in between -> Smooth blend
        """
        # Edge case: If snake is length 1, just return the start color
        if total_length < 2:
            return start_rgb
            
        # 't' is a value from 0.0 to 1.0 representing progress down the body
        t = position / (total_length - 1)
        
        # Linear Interpolation (Lerp) for Red, Green, and Blue channels
        r = start_rgb[0] + (end_rgb[0] - start_rgb[0]) * t
        g = start_rgb[1] + (end_rgb[1] - start_rgb[1]) * t
        b = start_rgb[2] + (end_rgb[2] - start_rgb[2]) * t
        
        return (int(r), int(g), int(b))

    # ------------------------------------------------------------------
    # NEW: UI Update with Gradients
    # ------------------------------------------------------------------
    def _update_ui(self):
        self.display.fill(settings.COLOR_BG)

        # 1. Draw All Snakes
        for snake in self.snakes:
            if not snake.is_alive: continue # Optional: Skip dead snakes
            
            snake_len = len(snake.body)
            
            # Target Colors: Head is always White, Tail is the settings color
            head_color_target = (255, 255, 255) 
            tail_color_target = snake.color 

            for i, pt in enumerate(snake.body):
                
                # Calculate the specific color for THIS segment
                current_color = self._get_gradient_color(head_color_target, tail_color_target, i, snake_len)
                
                # Draw the block (Full block size for smooth gradient look)
                pygame.draw.rect(
                    self.display, 
                    current_color, 
                    pygame.Rect(pt.x, pt.y, settings.BLOCK_SIZE, settings.BLOCK_SIZE)
                )
                
                # OPTIONAL: Draw Eyes on the Head (since head is white)
                # This helps see direction clearly
                if i == 0:
                    center = settings.BLOCK_SIZE // 2
                    radius = 3
                    eye_color = (0, 0, 0) # Black eyes on White head
                    
                    # Draw a simple 'eye' in the center of the head block
                    # (You can make this more complex based on direction if you want)
                    pygame.draw.circle(
                        self.display, 
                        eye_color, 
                        (int(pt.x + center), int(pt.y + center)), 
                        radius
                    )

        # 2. Draw Food
        for food in self.food_list:
            pygame.draw.rect(
                self.display, 
                settings.COLOR_FOOD, 
                pygame.Rect(food.x, food.y, settings.BLOCK_SIZE, settings.BLOCK_SIZE)
            )

        # 3. Draw Scores
        # Join names and scores with a separator
        text_str = " | ".join([f"{s.name}: {s.score}" for s in self.snakes])
        text = font.render(text_str, True, settings.COLOR_TEXT)
        self.display.blit(text, [0, 0])
        
        # 4. Refresh Screen
        pygame.display.flip()