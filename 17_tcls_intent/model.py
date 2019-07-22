import torch
import torch.nn as nn

from  torch.autograd import Variable


def expand(tensor, target):
    return tensor.expand_as(target)


def masking(tensor, mask):
    return torch.mul(tensor, expand(mask.unsqueeze(-1), tensor).float())


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout=.0):
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(input_dim,
                          hidden_dim,
                          batch_first=True,
                          num_layers=n_layers,
                          bidirectional=True,
                          dropout=dropout)

    def forward(self, embedded_inputs, input_length, hidden_state=None):
        batch_size = embedded_inputs.size(0)
        total_length = embedded_inputs.size(1)

        if hidden_state is None:
            h_size = (self.n_layers * 2, batch_size, self.hidden_dim)
            enc_h_0 = Variable(embedded_inputs.data.new(*h_size).zero_(), requires_grad=False)

        packed_input = nn.utils.rnn.pack_padded_sequence(embedded_inputs, input_length.tolist(), batch_first=True)
        packed_output, enc_h_t = self.gru(packed_input, enc_h_0)
        _, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, total_length=total_length)
        return enc_h_t[-1]


class LSTMClassifier(nn.Module):
    def __init__(self, config, vocab_size, n_label):
        super(LSTMClassifier, self).__init__()

        self.embed = nn.Embedding(vocab_size, config.embed_dim)

        self.encoder = Encoder(config.embed_dim,
                               config.hidden_dim,
                               config.n_layer,
                               dropout=config.dropout)

        self.classifier = nn.Linear(config.hidden_dim, n_label)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids, input_length):
        embedded_layer = self.embed(input_ids)
        embedded_layer = self.dropout(embedded_layer)
        encoded_layer = self.encoder(embedded_layer, input_length)
        logits = self.classifier(encoded_layer)

        outputs = {
            "logits": logits,
            "predicted_intents": torch.topk(logits, 1)[1],
        }
        return outputs