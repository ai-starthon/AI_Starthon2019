
import torch
import torch.nn as nn
import torch.nn.functional as F


class CharCNN(nn.Module):
    def __init__(
            self,
            vocab_size,
            embed_size,
            filter_sizes,
            out_dim,
            dropout,
            activation,
            pad_ind=0
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.out_dim = out_dim
        self.filter_sizes = sorted(filter_sizes)
        self.activation = {
            "relu": F.relu,
            "tanh": torch.tanh,
            "elu": nn.functional.elu,
            "none": lambda x: x,
        }[activation]
        self.dropout = nn.Dropout(p=dropout)

        assert self.out_dim % len(self.filter_sizes) == 0

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size,
                                      padding_idx=pad_ind)

        self.convs = nn.ModuleList([nn.Conv1d(self.embed_size, self.out_dim // len(self.filter_sizes), width) \
                                    for width in self.filter_sizes])
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x, seq_len=None, word_lens=None):
        embeds = self.embedding(x)
        embeds = self.dropout(embeds)  # [B, L, D]

        if embeds.size(1) < max(self.filter_sizes):
            embeds = torch.cat(
                [
                    embeds,
                    torch.zeros(embeds.size(0), max(self.filter_sizes) - embeds.size(1), embeds.size(2)).to(embeds.device)
                ], dim=1)  # pad

        if seq_len is not None:
            L = max(seq_len.max().int(), max(self.filter_sizes))
            embeds = embeds[:, :L, :] # [B, L, D] with L cut

        embeds = embeds.transpose(1, 2)   # [B, D, L]
        conv_outputs = []
        for conv in self.convs:
            x = self.pool(self.activation(conv(embeds))).squeeze(-1)
            conv_outputs.append(x)
        ret = torch.cat(conv_outputs, dim=-1)   # [B, num_all_filters]
        return ret
