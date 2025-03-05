import torch
from torch import nn
from torchviz import make_dot

import torch
from torch import nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(DQN, self).__init__()


        self.layer1 = nn.Linear(state_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, action_dim)



    def forward(self, x):

        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)

        return x



# Define and instantiate the model
state_dim = 6
action_dim = 8
model = DQN(state_dim, action_dim)

# Create dummy input
dummy_input = torch.randn(1, state_dim)

# Forward pass
output = model(dummy_input)

# Generate visualization
dot = make_dot(output, params=dict(model.named_parameters()))
dot.render("dqn_model_graph", format="png")  # Save as a PNG file