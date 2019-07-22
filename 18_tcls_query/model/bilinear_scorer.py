
import torch
import torch.nn as nn


class BilinearScorer(nn.Module):
    def __init__(
            self,
            input_dim,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.W = nn.Linear(input_dim, input_dim)

    def forward(self, a_embeds, b_embeds):
        return torch.sum(a_embeds * self.W(b_embeds), dim=-1)
