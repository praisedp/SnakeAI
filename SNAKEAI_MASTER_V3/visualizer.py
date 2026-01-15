import pygame
import torch
import numpy as np
import settings

class NNVisualizer:
    def __init__(self, model):
        self.model = model
        self.conf = settings.VIS_CONFIG
        self.font = pygame.font.SysFont('arial', 12)
        
        # 1. Detect Layers
        # We manually define the architecture because extracting it 
        # perfectly from a dynamic PyTorch model is tricky without running it.
        # Format: (Layer Name, Number of Nodes, Is Input?)
        if settings.VISUALIZE_INPUT_SLICE:
             input_size = 38 # Show only 'Live' frame
        else:
             input_size = settings.INPUT_SIZE # Show all stacked frames

        self.layers = [
            ("Input (114 neurons)", input_size, True),
            ("Hidden (512 neurons)", settings.HIDDEN_SIZE_0, False),
            ("Hidden (256 neurons)", settings.HIDDEN_SIZE_1, False),
            ("Output (3 neurons)", settings.OUTPUT_SIZE, False)
        ]

        # 2. Setup Node Positions
        self.node_positions = []  # List of lists of (x, y)
        self.sampled_indices = [] # List of lists of real indices [0, 50, 100...]
        
        self._calculate_layout()
        
        # 3. Storage for live data
        self.activations = {} # Key: Layer Index, Value: Tensor/Array
        self.hooks = []
        self._register_hooks()

    def _calculate_layout(self):
        """Calculates screen coordinates for nodes, handling downsampling."""
        margin = 20
        start_x = self.conf["X"] + margin
        center_y = self.conf["Y"] + (self.conf["HEIGHT"] // 2)

        # Distribute layers across available width so changing VIS_CONFIG width resizes the layout
        usable_width = max(self.conf["WIDTH"] - (margin * 2), margin)
        layer_spacing = usable_width // max(1, len(self.layers) - 1)
        
        for i, (name, size, is_input) in enumerate(self.layers):
            layer_positions = []
            layer_indices = []
            
            # Decide how many nodes to draw
            if size > self.conf["MAX_NODES"]:
                num_to_draw = self.conf["MAX_NODES"]
                # Create evenly spaced indices (e.g., 0, 50, 100... 511)
                indices = np.linspace(0, size - 1, num_to_draw, dtype=int)
            else:
                num_to_draw = size
                indices = np.arange(size)

            # Calculate Y positions to center them
            total_h = num_to_draw * (self.conf["NODE_RADIUS"] * 3)
            start_y = center_y - (total_h // 2)
            
            for j in range(num_to_draw):
                x = start_x + (i * layer_spacing)
                y = start_y + (j * (self.conf["NODE_RADIUS"] * 3))
                layer_positions.append((x, y))
                layer_indices.append(indices[j])
            
            self.node_positions.append(layer_positions)
            self.sampled_indices.append(layer_indices)

    def _register_hooks(self):
        """Attaches 'spies' to the Linear layers to capture output values."""
        # We hook into the Linear layers. 
        # Note: This captures Pre-ReLU values. We will clamp them visually.
        
        # Map visual layers to model layers
        # Visual Layer 0 is Input (No hook needed, we get it from update())
        # Visual Layer 1 is output of self.model.linear1
        # Visual Layer 2 is output of self.model.linear3
        # Visual Layer 3 is output of self.model.output
        
        target_modules = [self.model.linear1, self.model.linear3, self.model.output]
        
        for i, module in enumerate(target_modules):
            # layer_idx maps to self.layers (0=Input, so +1)
            layer_idx = i + 1 
            
            # Define the hook function (needs default arg to capture layer_idx)
            def hook_fn(module, input, output, idx=layer_idx):
                self.activations[idx] = output.detach().cpu().numpy()
            
            handle = module.register_forward_hook(hook_fn)
            self.hooks.append(handle)

    def update(self, state_input):
        """Called every frame with the raw input state."""
        # Store input activations (Layer 0)
        # Handle the slicing logic (Full stack vs Last Frame)
        if settings.VISUALIZE_INPUT_SLICE:
            # Assuming stack is [old, mid, new], we take the last chunk
            single_frame_size = 38
            sliced = state_input[-single_frame_size:] 
            self.activations[0] = sliced
        else:
            self.activations[0] = state_input

    def draw(self, surface):
        """Main draw loop."""
        # 1. Draw Background
        bg_rect = (self.conf["X"], self.conf["Y"], self.conf["WIDTH"], self.conf["HEIGHT"])
        pygame.draw.rect(surface, self.conf["COLORS"]["BG"], bg_rect)
        pygame.draw.rect(surface, (100, 100, 100), bg_rect, 2) # Border

        # 2. Draw Connections (Weights)
        # We look at Layer i and Layer i+1
        for i in range(len(self.layers) - 1):
            curr_layer_indices = self.sampled_indices[i]
            next_layer_indices = self.sampled_indices[i+1]
            
            curr_pos = self.node_positions[i]
            next_pos = self.node_positions[i+1]

            # Get weight matrix from the model
            # layer 0->1 is linear1, 1->2 is linear3, 2->3 is output
            if i == 0: weights = self.model.linear1.weight.detach().cpu().numpy()
            elif i == 1: weights = self.model.linear3.weight.detach().cpu().numpy()
            elif i == 2: weights = self.model.output.weight.detach().cpu().numpy()

            # Iterate through our SAMPLED nodes only
            for c_vis_idx, c_real_idx in enumerate(curr_layer_indices):
                for n_vis_idx, n_real_idx in enumerate(next_layer_indices):
                    
                    # Weight shape is (Output_Size, Input_Size) -> (Next, Current)
                    # We need to handle the input slicing offset if visualizing slice
                    weight_val = 0
                    try:
                        if i == 0 and settings.VISUALIZE_INPUT_SLICE:
                            # Use offset to point to the latest frame in the weights
                            # Last frame starts at index 76 (38*2)
                            offset = settings.INPUT_SIZE - 38
                            weight_val = weights[n_real_idx][c_real_idx + offset]
                        else:
                            weight_val = weights[n_real_idx][c_real_idx]
                    except IndexError:
                        continue # Safety skip

                    # Visualization Threshold (Don't draw weak connections)
                    if abs(weight_val) > 0.1:
                        color = self.conf["COLORS"]["WEIGHT_POS"] if weight_val > 0 else self.conf["COLORS"]["WEIGHT_NEG"]
                        # Scale alpha/width by strength
                        intensity = min(255, int(abs(weight_val) * 255))
                        width = 1 if intensity < 100 else 2
                        
                        start = curr_pos[c_vis_idx]
                        end = next_pos[n_vis_idx]
                        
                        # Draw Line
                        pygame.draw.line(surface, color, start, end, width)

        # 3. Draw Nodes (Activations)
        for i in range(len(self.layers)):
            positions = self.node_positions[i]
            real_indices = self.sampled_indices[i]
            
            # Get activation data if available
            layer_data = self.activations.get(i, None)
            layer_vec = None
            max_abs = 1.0
            if layer_data is not None:
                layer_vec = layer_data[0] if len(layer_data.shape) > 1 else layer_data
                max_abs = float(np.max(np.abs(layer_vec))) + 1e-6
            
            for vis_idx, pos in enumerate(positions):
                val = 0.0
                if layer_vec is not None:
                    # Normalize per-layer so small activations stay visible
                    real_idx = real_indices[vis_idx]
                    val = float(layer_vec[real_idx]) / max_abs

                # Grayscale color logic: black when idle, brighter for positive, dim gray for negative
                if val > 0:
                    shade = min(255, int(40 + (abs(val) * 215)))
                elif val < 0:
                    shade = max(20, int(abs(val) * 140))
                else:
                    shade = 0  # inactive = black
                color = (shade, shade, shade)
                
                pygame.draw.circle(surface, color, pos, self.conf["NODE_RADIUS"])
                pygame.draw.circle(surface, (200,200,200), pos, self.conf["NODE_RADIUS"], 1) # Outline

            # Draw Layer Label
            label = self.font.render(self.layers[i][0], True, self.conf["COLORS"]["TEXT"])
            surface.blit(label, (positions[0][0] - 10, positions[0][1] - 30))

    def close(self):
        """Removes hooks to prevent memory leaks."""
        for h in self.hooks:
            h.remove()