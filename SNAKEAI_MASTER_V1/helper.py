import matplotlib.pyplot as plt
from IPython import display
import settings
import numpy as np

plt.ion() # Turn on interactive mode

def plot(scores, mean_scores):
    """
    Plots training data.
    Supports both Single Agent (list of ints) and Multi-Agent (list of lists of ints).
    """
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    
    # 1. Setup Title and Labels from Settings
    plt.title(getattr(settings, 'PLOT_TITLE', 'Training...'))
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    
    # Add Description as a subtitle (smaller font)
    desc = getattr(settings, 'PLOT_DESCRIPTION', '')
    if desc:
        plt.figtext(0.5, 0.01, desc, wrap=True, horizontalalignment='center', fontsize=8)

    # 2. Normalize Data Format
    # If we receive a simple list [1, 2, 3], wrap it [[1, 2, 3]] so logic handles it like 1 agent.
    # This ensures backwards compatibility with your current agent.py
    if not isinstance(scores[0], list) and not isinstance(scores[0], np.ndarray):
        all_scores = [scores]
        all_means = [mean_scores]
    else:
        all_scores = scores
        all_means = mean_scores

    # 3. Plot Data for Each Agent
    # We iterate through the agents defined in settings to get their colors/names
    for i, agent_data in enumerate(all_scores):
        
        # Safety check: If we have more score lists than agents defined in settings
        if i >= len(settings.AGENTS):
            name = f"Unknown Agent {i}"
            color = (0, 0, 0) # Default Black
        else:
            conf = settings.AGENTS[i]
            name = conf["name"]
            # Convert RGB (0-255) to Matplotlib format (0.0-1.0)
            c = conf["color"] # Use head color for the graph line
            color = (c[0]/255, c[1]/255, c[2]/255)

        # Plot Score (Solid Line)
        plt.plot(agent_data, label=f"{name} Score", color=color, linewidth=1.5)
        
        # Plot Mean (Dotted Line) - slightly transparent
        mean_data = all_means[i]
        plt.plot(mean_data, color=(0, 0, 0), alpha=0.6, linewidth=1.5)

        # Text Annotation at the end of the line (Current Score)
        if len(agent_data) > 0:
            last_val = agent_data[-1]
            last_mean = mean_data[-1]
            plt.text(len(agent_data)-1, last_val, str(last_val), color=(0, 0, 0), fontsize=9, fontweight='bold')
            plt.text(len(agent_data)-1, last_mean, f"avg:{last_mean:.4f}", color=(0, 0, 0), fontsize=8)

    # 4. Finalize
    plt.legend(loc='upper left')
    plt.ylim(ymin=0)
    plt.subplots_adjust(bottom=0.15) # Make room for the description text
    plt.show(block=False)
    plt.pause(0.1)