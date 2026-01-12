import pygame
import random
import numpy as np
from enum import Enum
from collections import deque
from collections import namedtuple
import settings

pygame.init()
font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT, LEFT, UP, DOWN = 1, 2, 3, 4

Point = namedtuple('Point', 'x, y')

# --- Class to handle ONE snake's physics ---
class SnakeEntity:
    """
    Represents a single snake agent in the game.
    Handles physical state, movement logic, and body segments.
    """

    def __init__(self, name, color, start_x, start_y):
        self.name = name
        self.color = color
        
        # Physics State
        self.direction = Direction.RIGHT
        self.head = Point(start_x, start_y)
        self.body = self._init_body(start_x, start_y)
        
        # Frame Stacking Memory
        self.state_history = deque(maxlen=settings.STACK_SIZE)

        single_frame_size = settings.INPUT_SIZE // settings.STACK_SIZE
        
        for _ in range(settings.STACK_SIZE):
            self.state_history.append(np.zeros(single_frame_size))
            
        # Game State
        self.score = 0
        self.is_alive = True
        self.frame_iteration = 0
        
        # Memory (For Frame Stacking Later)
        # self.state_history = deque(maxlen=settings.STACK_SIZE)

    def move(self, action):
        """
        Updates the snake's direction and head position based on the action.
        Action format: [Straight, Right Turn, Left Turn]
        """
        self._update_direction(action) # 1. Update Facing Direction
        self._move_head()              # 2. Update Head Coordinate

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------

    def _init_body(self, x, y):
        """Initializes the body segments (Head + 2 Tail segments moving left)"""
        block = settings.BLOCK_SIZE
        return [
            Point(x, y),
            Point(x - block, y),
            Point(x - 2 * block, y)
        ]

    def _update_direction(self, action):
        """Calculates the new direction based on the RL action [Straight, Right, Left]"""
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        current_idx = clock_wise.index(self.direction)

        new_dir_idx = current_idx # Default: Go Straight [1, 0, 0]

        if np.array_equal(action, [0, 1, 0]):   # [0, 1, 0] -> Turn Right (Clockwise)
            new_dir_idx = (current_idx + 1) % 4
            
        elif np.array_equal(action, [0, 0, 1]): # [0, 0, 1] -> Turn Left (Counter-Clockwise)
            new_dir_idx = (current_idx - 1) % 4

        self.direction = clock_wise[new_dir_idx]

    def _move_head(self):
        """Calculates the new head coordinate based on current direction"""
        x = self.head.x
        y = self.head.y
        block = settings.BLOCK_SIZE

        if self.direction == Direction.RIGHT:
            x += block
        elif self.direction == Direction.LEFT:
            x -= block
        elif self.direction == Direction.DOWN:
            y += block
        elif self.direction == Direction.UP:
            y -= block

        self.head = Point(x, y)

# --- THE GAME ENGINE ---
class SnakeGameAI:
    """
    The Main Game Engine. 
    Manages the game loop, rendering, collision detection, and multi-agent rules.
    """

    # ------------------------------------------------------------------
    # 1. INITIALIZATION & SETUP
    # ------------------------------------------------------------------
    def __init__(self):
        self._init_display()
        self.clock = pygame.time.Clock()
        self.reset()

    def _init_display(self):
        """Initializes the Pygame window if rendering is enabled."""
        self.w = settings.WIDTH
        self.h = settings.HEIGHT
        if settings.RENDER:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake AI - Battle Arena')

    def reset(self):
        """Resets the game state: respawns snakes and food."""
        self._create_agents()
        self._respawn_food()
        self.frame_iteration = 0

    def _create_agents(self):
        """Instantiates SnakeEntity objects at random, non-overlapping positions."""
        self.snakes = []
        
        # Calculate grid limits
        rows = (self.h // settings.BLOCK_SIZE) - 1
        cols = (self.w // settings.BLOCK_SIZE) - 1

        for agent_conf in settings.AGENTS:
            
            # Loop until we find a valid, safe spot
            while True:
                # 1. Pick a Random Spot
                # range(2, cols): Ensures x is at least 2 blocks from left (for tail room)
                # range(1, rows): Ensures y is not touching the very edge
                x = random.randint(2, cols - 1) * settings.BLOCK_SIZE
                y = random.randint(1, rows - 1) * settings.BLOCK_SIZE
                
                # 2. Check overlap with existing snakes
                valid_spot = True
                for other in self.snakes:
                    # Manhattan Distance check
                    # If the new head is within 5 blocks of another snake, try again
                    dist = abs(x - other.head.x) + abs(y - other.head.y)
                    if dist < 5 * settings.BLOCK_SIZE:
                        valid_spot = False
                        break
                
                if valid_spot:
                    # Found a good spot! Break the while loop
                    start_x, start_y = x, y
                    break 

            # Create the snake at the found coordinate
            new_snake = SnakeEntity(
                name=agent_conf["name"], 
                color=agent_conf["color"],
                start_x=start_x, 
                start_y=start_y
            )
            self.snakes.append(new_snake)

    def _respawn_food(self):
        """Clears old food and spawns new apples up to the limit."""
        self.food_list = []
        for _ in range(settings.NUM_FOOD):
            if not self._place_single_food():
                break

    # ------------------------------------------------------------------
    # 2. MAIN GAME LOOP (The API)
    # ------------------------------------------------------------------
    def play_step(self, actions, render=True, model=None):
        """
        Executes one frame of the game.
        Args:
            actions (list): List of moves [[1,0,0], ...] for each snake.
            render (bool): Whether to draw the frame.
            model (torch.nn.Module): Optional, for visualizing NN internals.
        Returns:
            rewards (list), dones (list), scores (list)
        """
        self.frame_iteration += 1
        
        self._handle_user_events() # 1. Event Handling (Quit Game)
        actions = self._validate_actions(actions) # 2. Input Validation (Pad actions if fewer actions than snakes)
        self._move_all_snakes(actions) # 3. Physics Step (Move snakes)
        rewards, dones = self._process_game_rules() # 4. Game Logic (Collisions, Eating, Rewards)

        # 5. Rendering
        if settings.RENDER and render:
            self._update_ui(model)
            self.clock.tick(settings.SPEED)

        return rewards, dones, [s.score for s in self.snakes]

    # ------------------------------------------------------------------
    # 3. PRIVATE LOGIC HELPERS
    # ------------------------------------------------------------------
    def _handle_user_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

    def _validate_actions(self, actions):
        """Ensures we have a valid move for every snake."""
        if len(actions) < len(self.snakes):
            missing = len(self.snakes) - len(actions)
            # Default to "Straight" [1, 0, 0] for missing actions
            actions = list(actions) + [[1, 0, 0] for _ in range(missing)]
        return actions

    def _move_all_snakes(self, actions):
        for i, snake in enumerate(self.snakes):
            if snake.is_alive:
                snake.move(actions[i])
                snake.body.insert(0, snake.head) # Add new head
                # Note: We don't pop tail here; we do it in _process_game_rules depending on food

    def _process_game_rules(self):
        rewards = []
        dones = []
        
        # Buffer for kill bonuses (e.g. +50 for killing an enemy)
        kill_bonuses = [0] * len(self.snakes)

        # FIX 1: Use enumerate() to get the index 'i'
        for i, snake in enumerate(self.snakes):
            
            # Skip dead snakes
            if not snake.is_alive:
                rewards.append(0)
                dones.append(True)
                continue

            snake.frame_iteration += 1
            reward = settings.REWARD_STEP
            game_over = False

            # --- Rule A: Collision (Death) ---
            if self._is_collision(snake) or self._is_starving(snake):
                game_over = True
                reward = settings.REWARD_COLLISION - (snake.score * settings.REWARD_STARVE_MULTIPLIER)
                snake.is_alive = False

                # [NEW] KILL REWARD LOGIC
                # Only give credit for kills, not starvation
                if not self._is_starving(snake): 
                    # FIX 3: Removed victim_idx, just pass the snake object
                    self._handle_kill_credit(snake, kill_bonuses)
                    
            # --- Rule B: Eating Food (Growth) ---
            elif self._check_food_eaten(snake):
                snake.score += 1
                reward = settings.REWARD_FOOD
                snake.frame_iteration = 0
            
            # --- Rule C: Normal Movement ---
            else:
                snake.body.pop() # Remove tail

            rewards.append(reward)
            dones.append(game_over)

        # Merge the standard rewards with the kill bonuses
        final_rewards = []
        for i in range(len(rewards)):
            total = rewards[i] + kill_bonuses[i]
            final_rewards.append(total)

        # Maintain food count
        while len(self.food_list) < settings.NUM_FOOD:
            if not self._place_single_food():
                break

        # FIX 2: Return 'final_rewards', not 'rewards'
        return final_rewards, dones
    
    # ------------------------------------------------------------------
    # 4. COLLISION & PHYSICS HELPERS
    # ------------------------------------------------------------------
    def _is_starving(self, snake):
        return snake.frame_iteration > settings.STARVE_LIMIT * len(snake.body)

    def _check_food_eaten(self, snake):
        if snake.head in self.food_list:
            self.food_list.remove(snake.head)
            self._place_single_food()
            return True
        return False

    def _is_collision(self, snake):
        pt = snake.head
        # 1. Wall Collision
        if pt.x > self.w - settings.BLOCK_SIZE or pt.x < 0 or pt.y > self.h - settings.BLOCK_SIZE or pt.y < 0:
            return True
        
        # 2. Self Collision
        if pt in snake.body[1:]:
            return True

        # 3. Enemy Collision
        for other in self.snakes:
            if other != snake:
                # Treat dead snakes as solid to ensure simultaneous head-on hits kill both
                if pt in other.body:
                    return True
        return False

    def _handle_kill_credit(self, victim, bonuses):
        """
        Checks if the victim hit another snake. 
        If so, gives REWARD_KILL to that snake.
        """
        head = victim.head
        
        for i, other in enumerate(self.snakes):
            # Don't check yourself or dead snakes
            if other != victim and other.is_alive:
                # Did the victim's head hit this snake's body?
                if head in other.body:
                    bonuses[i] += settings.REWARD_KILL
                    # Optional: Print for debugging
                    # print(f"KILL! {other.name} destroyed {victim.name}")
                    break

    def _place_single_food(self):
        """Attempts to place a single apple in a valid spot."""
        # Create a set of all occupied coordinates for fast lookup
        occupied = {(pt.x, pt.y) for pt in self.food_list}
        for snake in self.snakes:
            occupied.update((pt.x, pt.y) for pt in snake.body)

        # Generate list of ALL valid candidates
        candidates = []
        for x in range(0, self.w, settings.BLOCK_SIZE):
            for y in range(0, self.h, settings.BLOCK_SIZE):
                # FIX: Avoid Scoreboard Area (Top-Left 300x50)
                if x < 300 and y < 50:
                    continue
                
                if (x, y) not in occupied:
                    candidates.append((x, y))

        if not candidates:
            return False

        x, y = random.choice(candidates)
        self.food_list.append(Point(x, y))
        return True

    # ------------------------------------------------------------------
    # 5. RENDERING HELPERS
    # ------------------------------------------------------------------
    def _update_ui(self, model=None):
        self.display.fill(settings.COLOR_BG)

        # Draw Snakes
        for snake in self.snakes:
            if not snake.is_alive: continue
            self._draw_snake(snake)

        # Draw Food
        for food in self.food_list:
            pygame.draw.rect(
                self.display, 
                settings.COLOR_FOOD, 
                pygame.Rect(food.x, food.y, settings.BLOCK_SIZE, settings.BLOCK_SIZE)
            )

        # Draw Text
        self._draw_score()
        
        # (Future Place for NN Visualizer)
        if model and settings.VISUALIZE_NN:
            # self._draw_nn(model) 
            pass

        pygame.display.flip()

    def _draw_snake(self, snake):
        snake_len = len(snake.body)
        head_color = (255, 255, 255)
        tail_color = snake.color

        for i, pt in enumerate(snake.body):
            color = self._get_gradient_color(head_color, tail_color, i, snake_len)
            pygame.draw.rect(self.display, color, pygame.Rect(pt.x, pt.y, settings.BLOCK_SIZE, settings.BLOCK_SIZE))
            
            # Optional: Draw Eyes
            if i == 0:
                center = settings.BLOCK_SIZE // 2
                pygame.draw.circle(self.display, (0,0,0), (int(pt.x+center), int(pt.y+center)), 3)

    def _draw_score(self):
        text_str = " | ".join([f"{s.name}: {s.score}" for s in self.snakes])
        text = font.render(text_str, True, settings.COLOR_TEXT)
        self.display.blit(text, [0, 0])

    def _get_gradient_color(self, start_rgb, end_rgb, position, total_length):
        if total_length < 2: return start_rgb
        t = position / (total_length - 1)
        r = start_rgb[0] + (end_rgb[0] - start_rgb[0]) * t
        g = start_rgb[1] + (end_rgb[1] - start_rgb[1]) * t
        b = start_rgb[2] + (end_rgb[2] - start_rgb[2]) * t
        return (int(r), int(g), int(b))