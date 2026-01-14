import sys
import settings
from agent import Agent
from game import SnakeGameAI
from helper import plot

def train():
    """
    The Main Multi-Agent Training Loop.
    Coordinates the 'Brain' (Agent) and the 'World' (Game).
    """
    
    # 1. Metrics Setup
    scores = []
    mean_scores = []
    total_score = 0
    
    # 2. Initialize Components
    agent = Agent()
    game = SnakeGameAI()
    
    record = getattr(agent, 'resume_record', 0) # <--- IMPORTANT
    print(f"Current Record to Beat: {record}")

    print(">>> Training Started. Press Ctrl+C to stop.")

    while True:
        # -------------------------------------------------------
        # STEP 1: GATHER ACTIONS (For Alive Snakes Only)
        # -------------------------------------------------------
        final_moves = []
        old_states = {} # Map {snake_index: state} to track who played this frame

        for i, snake in enumerate(game.snakes):
            if snake.is_alive:
                # A. See the world (Frame Stacked)
                state_old = agent.get_state(game, snake)
                
                # B. Decide move
                action = agent.get_action(state_old)
                
                # C. Store for training later
                old_states[i] = state_old
                final_moves.append(action)
            else:
                # Dead snakes do nothing (Maintain list alignment)
                final_moves.append(None)

        # -------------------------------------------------------
        # STEP 2: PHYSICS STEP
        # -------------------------------------------------------
        # Execute all moves simultaneously
        rewards, dones, current_scores = game.play_step(final_moves)

        # -------------------------------------------------------
        # STEP 3: TRAIN ON RESULTS (Short Term Memory)
        # -------------------------------------------------------
        for i, snake in enumerate(game.snakes):
            # Only train if the snake WAS alive at the start of the frame
            if i in old_states:
                state_old = old_states[i]
                action = final_moves[i]
                reward = rewards[i]
                done = dones[i] # Did I die in this specific step?
                
                # Get the new state (result of the move)
                # Note: Even if dead, we get the 'Terminal State' for the memory
                state_new = agent.get_state(game, snake)

                # Train immediate step
                agent.train_short_memory(state_old, action, reward, state_new, done)
                
                # Store in Replay Buffer
                agent.remember(state_old, action, reward, state_new, done)

        # -------------------------------------------------------
        # STEP 4: GAME OVER / RESET LOGIC
        # -------------------------------------------------------
        # Reset only when EVERYONE is dead
        if all(dones):
            game.reset()
            agent.n_games += 1
            
            # A. Experience Replay (Long Term Memory)
            agent.train_long_memory()

            # B. Update Target Network (Stability)
            if agent.n_games % settings.TARGET_UPDATE_SIZE == 0:
                agent.target_model.load_state_dict(agent.model.state_dict())
                print(f">>> Target Network Synced (Game {agent.n_games})")

            # C. Track Records (Using the Winner's score)
            match_high_score = max(current_scores) if current_scores else 0
            
            if match_high_score > record:
                record = match_high_score
                # Pass the NEW record to the save function
                agent.model.save(record=record) 
                print(f"NEW RECORD! {record}")

            # D. Logging & Plotting
            print(f"Game {agent.n_games} | High Score: {match_high_score} | Record: {record}")

            scores.append(match_high_score)
            total_score += match_high_score
            mean_scores.append(total_score / agent.n_games)
            
            # Plotting (Optional: Run in separate thread if slow)
            plot(scores, mean_scores)


if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\n>>> Training Interrupted. Exiting...")
        sys.exit()