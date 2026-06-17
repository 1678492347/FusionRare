import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

torch.set_default_device(device)


class ResidualNetworkClassifier(nn.Module):

    def __init__(self, lr_prior_matrix, residual_for='lr'):

        super().__init__()

        self.residual_for = residual_for
        self.num_hpo = lr_prior_matrix.size(1)
        self.num_dise = lr_prior_matrix.size(0)
        self.lr_knowledge_weight = nn.Parameter(
            lr_prior_matrix.clone().detach(),
            requires_grad=False
        )

        self.residual_net = nn.Sequential(
            nn.Linear(self.num_hpo, RESIDUAL_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(RESIDUAL_HIDDEN_SIZE, self.num_dise)
        )
        nn.init.zeros_(self.residual_net[-1].weight)
        nn.init.zeros_(self.residual_net[-1].bias)

        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x_1, x_2):
        kb_logits = F.linear(x_1, self.lr_knowledge_weight)
        res_logits = self.residual_net(x_2)
        final_logits = kb_logits + self.alpha * res_logits
        return final_logits

    def lr_predict(self, multi_hot_hpos):
        return torch.matmul(multi_hot_hpos, self.lr_knowledge_weight.T)
