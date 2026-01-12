import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import settings  # <--- IMPORT SETTINGS

class Linear_QNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize layers using sizes from settings.py
        # Layer 1: Input -> Hidden 1
        self.linear1 = nn.Linear(settings.INPUT_SIZE, settings.HIDDEN_SIZE_0)
        
        # Layer 2: Hidden -> Hidden
        self.linear2 = nn.Linear(settings.HIDDEN_SIZE_0, settings.HIDDEN_SIZE_1)
        
        # Layer 3: Hidden -> Hidden
        self.linear3 = nn.Linear(settings.HIDDEN_SIZE_1, settings.HIDDEN_SIZE_2)
        
        # Layer 4: Hidden -> Output (Action)
        self.output = nn.Linear(settings.HIDDEN_SIZE_2, settings.OUTPUT_SIZE)

        # Move model to the correct device (GPU/CPU) immediately
        self.to(settings.DEVICE)

    def forward(self, x):
        # 1. Input Layer + Relu
        x = F.relu(self.linear1(x))
        
        # 2. First Hidden Layer + Relu
        x = F.relu(self.linear2(x))
        
        # 3. Second Hidden Layer + Relu
        x = F.relu(self.linear3(x))
        
        # 4. Output Layer (Raw Q-Values, no activation)
        x = self.output(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './models'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done, target_model=None):
        # 1. Convert numpy arrays to PyTorch tensors AND move to DEVICE
        state = torch.tensor(state, dtype=torch.float).to(settings.DEVICE)
        next_state = torch.tensor(next_state, dtype=torch.float).to(settings.DEVICE)
        action = torch.tensor(action, dtype=torch.long).to(settings.DEVICE)
        reward = torch.tensor(reward, dtype=torch.float).to(settings.DEVICE)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x) -> Reshape to look like a batch of 1
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 2. Prediction
        pred = self.model(state)

        # 3. Bellman Equation
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                # Use target network if provided
                if target_model:
                    next_pred = target_model(next_state[idx])
                else:
                    next_pred = self.model(next_state[idx])
                
                Q_new = reward[idx] + self.gamma * torch.max(next_pred)

            # Update the target Q value
            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 4. Backpropagation
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()