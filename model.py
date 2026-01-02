import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Layer 1: Input -> Hidden
        self.linear1 = nn.Linear(input_size, hidden_size)
        # Layer 2: Hidden -> Output
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Activation Function: ReLU (Rectified Linear Unit)
        # It turns negative numbers to 0. It introduces "non-linearity".
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        # Create a folder to save the model if it doesn't exist
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr        # Learning Rate (how fast we learn)
        self.gamma = gamma  # Discount Rate (how much we care about future)
        self.model = model
        # Optimizer: Adam is the standard "smart" gradient descent
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        # Loss Function: Mean Squared Error (Target - Prediction)^2
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        # 1. Convert numpy arrays to PyTorch tensors
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        # Handle formatting if we only get 1 data point (Short Memory)
        if len(state.shape) == 1:
            # (1, x) -> Reshape to look like a batch of 1
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 2. Get the prediction (Q values) for current state
        # Example output: [[0.8, 0.1, -0.5]] (Batch size, 3 actions)
        pred = self.model(state)

        # 3. Apply the Bellman Equation:
        # Q_new = R + gamma * max(next_predicted_Q_value)
        # But only do this if not done. If done, Q_new = Reward (Death = -10).
        
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                # The 'Magic' formula
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            # Update the target Q value for the specific action we took
            # We use torch.argmax(action) to find which move we made (0, 1, or 2)
            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 4. Calculate Loss and Backpropagate
        self.optimizer.zero_grad()      # Reset gradients
        loss = self.criterion(target, pred) # Calculate error
        loss.backward()                 # Calculate gradients
        self.optimizer.step()           # Update weights