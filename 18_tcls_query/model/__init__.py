
import torch.nn as nn

from model.char_cnn import CharCNN
from model.bilinear_scorer import BilinearScorer


class CharCNNScorer(nn.Module):
    def __init__(
            self,
            vocab_size,
            char_embed_size,
            filter_sizes,
            sentence_embed_size,
            dropout,
            activation,
            pad_ind,
    ):
        super().__init__()

        self.sentence_embedder = CharCNN(
            vocab_size=vocab_size,
            embed_size=char_embed_size,
            filter_sizes=filter_sizes,
            out_dim=sentence_embed_size,
            dropout=dropout,
            activation=activation,
            pad_ind=pad_ind,
        )
        self.scorer = BilinearScorer(
            sentence_embed_size,
        )

    def forward(
            self,
            a_seqs,
            b_seqs,
            len_a_seqs=None,
            len_b_seqs=None,
    ):
        a_embeds = self.sentence_embedder(a_seqs, len_a_seqs)
        b_embeds = self.sentence_embedder(b_seqs, len_b_seqs)

        logits = self.scorer(a_embeds, b_embeds).squeeze(-1)

        return logits
