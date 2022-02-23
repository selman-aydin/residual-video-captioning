import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.optim import SGD
from pathlib import Path
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

DEVICE = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class EncoderRNN(nn.Module):
    def __init__(self, output_size, hidden_size=2048, num_layers=4, dropout_p=0.2, init_weight=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedder = nn.Embedding(output_size, hidden_size)
        init.uniform_(self.embedder.weight.data, -init_weight, init_weight)

        self.rnn_layers = nn.ModuleList()
        self.rnn_layers.append(
            nn.LSTM(hidden_size, hidden_size, num_layers=1, bias=True, bidirectional=True, batch_first=True))

        # 2nd LSTM layer, with 2x larger input_size
        self.rnn_layers.append(
            nn.LSTM((2 * hidden_size), hidden_size, num_layers=1, bias=True, batch_first=True))

        # Remaining LSTM layers
        for _ in range(num_layers - 2):
            self.rnn_layers.append(
                nn.LSTM(hidden_size, hidden_size, num_layers=1, bias=True, batch_first=True))

        self.classifier = nn.Linear(hidden_size, output_size)
        init.uniform_(self.classifier.weight.data, -init_weight, init_weight)
        init.uniform_(self.classifier.bias.data, -init_weight, init_weight)
        self.dropout = nn.Dropout(p=dropout_p)
        self.softmax = nn.LogSoftmax(dim=1)

    def initialize_hidden(self):
        # Initalizing hidden state
        hiddens = [torch.zeros(2, 64, self.hidden_size) if i == 0 else torch.zeros(1, 64, 2048) for i in
                   range(self.num_layers)]
        return hiddens

    def forward(self, x, first_step=False):
        # bidirectional layer
        hiddens = self.initialize_hidden()
        carry = self.initialize_hidden()


        x = self.dropout(x)

        x, _ = self.rnn_layers[0](x, (hiddens[0], carry[0]))

        # 1st unidirectional layer
        x = self.dropout(x)
        x, _ = self.rnn_layers[1](x, (hiddens[1], carry[1]))

        # the rest of unidirectional layers,
        # with residual connections starting from 3rd layer
        for i in range(2, len(self.rnn_layers)):
            residual = x
            x = self.dropout(x)
            x, _ = self.rnn_layers[i](x, (hiddens[i], carry[i]))
            x = x + residual

        return x



class VisualAttention(nn.Module):
    """Some Information about VisualAttention"""

    def __init__(self, units):
        super(VisualAttention, self).__init__()
        self.W1 = nn.Linear(units, units)
        self.W2 = nn.Linear(units, units)
        self.V = nn.Linear(units, 1)

    def forward(self, x, hidden):
        # x shape == (N, 8, 2048)

        # hidden shape == (1, N, hidden_size)

        # hidden shape == (N, 1, hidden_size)
        hidden = hidden.squeeze().unsqueeze(1)

        # attention hidden layer shape == (N, 8, units)
        attention_hidden_layer = torch.tanh(self.W1(x) + self.W2(hidden))
        # score shape == (N, 8, 1)
        score = self.V(attention_hidden_layer)
        # attention weights shape == (N, 1, 8)
        attention_weights = torch.softmax(score, axis=1).squeeze().unsqueeze(1)
        # context vector shape == (N, 2048)
        context_vector = torch.bmm(attention_weights, x).squeeze().unsqueeze(0)

        return context_vector, attention_weights


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, dropout_p=0.1):
        super(DecoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_p = nn.Dropout(dropout_p)
        self.embedding = nn.Embedding(output_size, hidden_size)

        self.rnn_layers = nn.ModuleList()
        self.rnn_layers.append(
            nn.LSTM(hidden_size, hidden_size, num_layers=1, bias=True))

        # Remaining LSTM layers
        for _ in range(num_layers - 1):
            self.rnn_layers.append(
                nn.LSTM(hidden_size * 2, hidden_size, num_layers=1, bias=True))

        self.classifier = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.attention = VisualAttention(self.hidden_size)

    def initialize_hidden(self):
        # Initalizing hidden state
        hiddens = [torch.zeros(1, 64, self.hidden_size) for i in range(self.num_layers)]
        return hiddens

    def forward(self, x, features,hidden_states,carry_states):

        hidden_outputs = []
        carry_outputs = []

        x = x.unsqueeze(0)
        x = self.embedding(x)
        x = self.dropout_p(x)

        x, h0 = self.rnn_layers[0](x, (hidden_states[0], carry_states[0]))
        hidden_outputs.append(h0[0])
        carry_outputs.append(h0[1])
        context_vector, attention_weigths = self.attention(features, h0[0])

        # x = torch.cat((x, context_vector), dim=2)
        # x = self.dropout_p(x)

        for i in range(1, len(self.rnn_layers)):
            residual = x
            x = torch.cat((x, context_vector), dim=2)
            x = self.dropout_p(x)
            x, h_i = self.rnn_layers[i](x, (hidden_states[i], carry_states[i]))
            hidden_outputs.append(h_i[0])
            carry_outputs.append(h_i[1])
            x = x + residual

        x = self.softmax(self.classifier(x[0]))
        hiddens = hidden_outputs
        carrys = carry_outputs

        return x



a = torch.rand(64, 8, 2048)
t = torch.randint(10, (64,)).long()
output_size = 1270  # vocab_size = output_size
hidden_Size = 2048
num_layers_encoder = 8
num_layers_decoder = 5

encoder = EncoderRNN(output_size, hidden_Size, num_layers_encoder)
decoder = DecoderRNN(hidden_Size, output_size, num_layers_decoder)

output = encoder.forward(a)

hidden_states = decoder.initialize_hidden()
carry_states  = decoder.initialize_hidden()
output_decoder = decoder.forward(t, output,hidden_states,carry_states)
print("")
