import torch
import torch.nn.functional as F
class MSC_Policy(torch.nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self):
        super(MSC_Policy, self).__init__()
        self.affine1 = torch.nn.Linear(1, 128)

        # actor's layer
        self.action_head = torch.nn.Linear(128, 2)

        # critic's layer
        self.value_head = torch.nn.Linear(128, 1)

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(x))
        action_prob = F.softmax(self.action_head(x), dim=-1)
        state_values = self.value_head(x)
        return action_prob, state_values
    