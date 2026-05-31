import torch.nn as nn


class DQN(nn.Module):
    """
    Deep Q-Network.

    Input:
        5 state values

    Output:
        2 Q-values

    output[0] = predicted value of doing nothing
    output[1] = predicted value of flapping
    """

    def __init__(self, input_size=5, output_size=2):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
        )

    def forward(self, x):
        return self.network(x)