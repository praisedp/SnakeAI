import sys
import settings

# --- OVERRIDE SETTINGS FOR DEMO MODE ---
# We modify these before importing the game so they take effect immediately.
settings.RENDER = True          # Must see the game
settings.SPEED = 20             # Comfortable watch speed (change to 0 for max speed)
settings.LOAD_MODEL = True      # Force loading the trained brain
settings.EPSILON_START = 0      # 0% Randomness (Pure AI)
settings.EPSILON_MIN = 0        # Ensure it stays 0%


from agent import Agent
from game import SnakeGameAI

def demo():
    """
    Runs the game in visual mode using the pre-trained model.
    No training happens here.
    """
    print(">>> DEMO MODE: Loading AI Brain...")
    
    game = SnakeGameAI()
    agent = Agent()

    # Safety Check
    if agent.n_games == 0:
        print("\n[WARNING] It looks like the model didn't load (n_games=0).")
        print("Make sure 'model.pth' exists in the ./models/ folder.")
        print("Or set LOAD_MODEL = True in settings.py if it wasn't overridden.\n")

    print(">>> RUNNING! Press Ctrl+C to exit.")

    while True:
        # -------------------------------------------------------
        # 1. GET PURE AI ACTIONS
        # -------------------------------------------------------
        final_moves = []
        
        for snake in game.snakes:
            if snake.is_alive:
                # Get the current state
                state = agent.get_state(game, snake)
                
                # Get action (With Epsilon=0, this is always the 'Best' move)
                action = agent.get_action(state)
                final_moves.append(action)
            else:
                final_moves.append(None)

        # -------------------------------------------------------
        # 2. PLAY STEP
        # -------------------------------------------------------
        # We don't need 'rewards' for the demo, just 'dones' to know when to reset
        _, dones, scores = game.play_step(final_moves)

        # -------------------------------------------------------
        # 3. RESET IF EVERYONE IS DEAD
        # -------------------------------------------------------
        if all(dones):
            game.reset()
            # Print the final score of that match
            winner_score = max(scores) if scores else 0
            print(f"Match Finished. Winner Score: {winner_score}")

if __name__ == "__main__":
    try:
        demo()
    except KeyboardInterrupt:
        print("\n>>> Demo Closed.")
        sys.exit()