import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.optim import SGD
from pathlib import Path
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
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

    def initialize_hidden(self, batch_size):
        # Initalizing hidden state
        hiddens = [torch.zeros(2, batch_size, self.hidden_size).to(DEVICE) if i == 0
                   else torch.zeros(1, batch_size, self.hidden_size).to(DEVICE)
                   for i in range(self.num_layers)]
        return hiddens

    def forward(self, x, batch_size, first_step=False):
        # bidirectional layer
        hiddens = self.initialize_hidden(batch_size)
        carry = self.initialize_hidden(batch_size)

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


class BahdanauAttention(nn.Module):
    """Some Information about VisualAttention"""

    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
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

import numpy as np
class VideoCaptioning(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, token, hidden_states, carry_states):

        x = self.encoder.forward(x,x.shape[0])
        return self.decoder.forward(token, x, hidden_states, carry_states)


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
        self.attention = BahdanauAttention(self.hidden_size)

    def initialize_hidden(self, batch_size):
        # Initalizing hidden state
        hiddens = [torch.zeros(1, batch_size, self.hidden_size).to(DEVICE) for i in range(self.num_layers)]
        return hiddens

    def forward(self, x, features, hidden_states, carry_states):

        hidden_outputs = []
        carry_outputs = []

        # x = x.unsqueeze(0)
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

        # x = torch.argmax(x, dim=-1)

        return x, hiddens, carrys


from torch.utils.tensorboard import SummaryWriter
from prepare_msvd_dataset import MSVDDataset
from prepare_msrvtt_dataset import MSRVTTDataset
from data_utils import get_loader_and_vocab
from tqdm import tqdm
import json
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


def get_datasets():
    dts = []
    # dt = MSCOCODataset()
    # dts.append(dt)
    dt = MSVDDataset()
    dts.append(dt)
    dt = MSRVTTDataset()
    dts.append(dt)
    return dts


DATASETS = get_datasets()


def train(dt, num_layer_encoder, num_layer_decoder, seed_number, dropout_p):
    torch.manual_seed(seed_number)
    torch.cuda.manual_seed(seed_number)
    torch.cuda.manual_seed_all(seed_number)

    project_folder = Path('.')

    annotation_file = dt.val_captions
    annotation_name = str(annotation_file.parts[-1][:-5])
    coco = COCO(str(annotation_file))

    train_loader, val_loader, test_loader, vocab = get_loader_and_vocab(dt)
    VOCABULARY_SIZE = vocab.__len__()
    features, tokens = next(iter(train_loader))
    BATCH_SIZE, FRAME_SIZE, HIDDEN_SIZE = features.shape
    BATCH_SIZE, CAPTION_LENGTH = tokens.shape

    COMMENT = f"_LSTM_16FRAME__{dt.name}_dataset_{num_layer_encoder}_Encoderlayer_{num_layer_decoder}_Decoderlayer_{seed_number}_seed_{dropout_p}_dropout"
    ANNOTATION_NAME = f"LSTM_16FRAME_{dt.name}_dataset_{num_layer_encoder}_Encoderlayer_{num_layer_decoder}_Decoderlayer_{seed_number}_seed_{dropout_p}_dropout"
    result_folder = project_folder / 'resultsSelman'
    if not result_folder.exists():
        result_folder.mkdir()
    annotation_folder = project_folder / ANNOTATION_NAME
    if not annotation_folder.exists():
        annotation_folder.mkdir()

    writer = SummaryWriter(comment=COMMENT)

    encoder = EncoderRNN(output_size=VOCABULARY_SIZE, hidden_size=HIDDEN_SIZE, num_layers=num_layer_encoder,
                         dropout_p=dropout_p)
    decoder = DecoderRNN(hidden_size=HIDDEN_SIZE, output_size=VOCABULARY_SIZE, num_layers=num_layer_decoder,
                         dropout_p=dropout_p)
    videoCaptioning = VideoCaptioning(encoder,decoder)

    encoder = encoder.to(DEVICE)
    decoder = decoder.to(DEVICE)
    videoCaptioning = videoCaptioning.to(DEVICE)

    features = features.to(DEVICE)
    tokens = tokens.to(DEVICE)
    hidden = decoder.initialize_hidden(BATCH_SIZE)
    carry = decoder.initialize_hidden(BATCH_SIZE)

    writer.add_graph(videoCaptioning, (features,(tokens[:, 0].unsqueeze(0)),hidden,carry))

    encoder_optimizer = SGD(encoder.parameters(), lr=0.01)
    decoder_optimizer = SGD(decoder.parameters(), lr=0.01)
    criterion = nn.NLLLoss()

    best_score = 0.0
    for epoch in tqdm(range(50)):
        encoder, decoder, epoch_loss = train_epoch(encoder, decoder, train_loader, num_layer_encoder, num_layer_decoder,
                                                   encoder_optimizer, decoder_optimizer, criterion)
        writer.add_scalar('Train loss', epoch_loss, epoch)
        best_score, epoch_val_scores = test_epoch(encoder, decoder, vocab, num_layer_encoder, num_layer_decoder, val_loader, test_loader, coco,
                                                  annotation_folder, result_folder, ANNOTATION_NAME, CAPTION_LENGTH, best_score)
        for metric, score in epoch_val_scores:
            writer.add_scalar(f'{metric}', score, epoch)

    encoder = None
    decoder = None

    print("")


def train_epoch(encoder, decoder, train_loader, num_layer_encoder, num_layer_decoder, encoder_optimizer,
                decoder_optimizer, criterion):
    encoder.train()
    decoder.train()

    running_loss = 0.0

    for ids, (X, y) in enumerate(train_loader):
        loss = 0.0
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        batch_size, caption_length = y.shape
        batch_size, frame_size, feature_size = X.shape
        y = y.to(DEVICE)
        X = X.to(DEVICE)

        h = decoder.initialize_hidden(batch_size)
        c = decoder.initialize_hidden(batch_size)

        encoder_output = encoder.forward(X, batch_size)

        for i in range(caption_length - 1):
            output, h, c = decoder.forward(y[:, i].unsqueeze(0), encoder_output, h, c)
            loss += criterion(output, y[:, i + 1])


        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        running_loss += loss.item() / caption_length
    epoch_loss = running_loss / (ids + 1)
    print(f"Train loss: {epoch_loss}")

    return encoder, decoder, epoch_loss


def test_epoch(encoder, decoder, vocab, num_layer_encoder, num_layer_decoder, val_loader, test_loader, coco, annotation_folder, result_folder, annotation_name,
               caption_length, best_cider_val_score):
    encoder.eval()
    decoder.eval()

    val_data = []
    val_name = []
    for X, y in val_loader:
        with torch.no_grad():
            captions, ids = test_step(encoder, decoder, num_layer_encoder, num_layer_decoder, X, y, vocab, caption_length)
        for caption, id in zip(captions, ids):
            if not id in val_name:
                val_name.append(id)
                val_data.append({
                    "image_id": id,
                    "caption": caption})


    json_file = f"{str(result_folder)}/{annotation_name}_result.json"
    with open(json_file, "w") as file:
        json.dump(val_data, file)

    coco_result = coco.loadRes(json_file)
    coco_eval = COCOEvalCap(coco, coco_result)
    coco_eval.params['image_id'] = coco_result.getImgIds()
    coco_eval.evaluate()
    val_scores = coco_eval.eval.items()

    for metric, score in val_scores:
        print(f'{metric}: {score:.3f}')

        if metric == "CIDEr":
            cider_score = score

    test_data = []
    test_name = []
    if cider_score > best_cider_val_score:
        best_cider_val_score = cider_score
        json_file = f"{str(annotation_folder)}/{annotation_name}_val_data_result.json"
        with open(json_file, "w") as file:
            json.dump(val_data, file)
        for X, y in test_loader:
            with torch.no_grad():
                captions, ids = test_step(encoder, decoder, num_layer_encoder, num_layer_decoder, X, y, vocab, caption_length)

            for caption, id in zip(captions, ids):
                if not id in test_name:
                    test_name.append(id)
                    test_data.append({
                        "image_id": id,
                        "caption": caption})

        json_file = f"{str(annotation_folder)}/{annotation_name}_test_data_result.json"
        with open(json_file, "w") as file:
            json.dump(test_data, file)
    return best_cider_val_score, val_scores

def test_step(encoder, decoder, num_layer_encoder, num_layer_decoder, X, y, vocab, caption_length):
    batch_size, frame_size, feature_size = X.shape
    X = X.to(DEVICE)

    h = decoder.initialize_hidden(batch_size)
    c = decoder.initialize_hidden(batch_size)

    stoi = vocab.get_stoi()
    start_token = stoi['boc']
    end_token = stoi['eoc']
    result = torch.zeros((batch_size, caption_length), dtype=torch.long, device=DEVICE)
    result[:, 0] = start_token

    encoder_output = encoder.forward(X, batch_size)
    for i in range(caption_length-1):
        output, h, c = decoder.forward(result[:, i].unsqueeze(0), encoder_output, h, c)
        result[:, i+1] = torch.argmax(output, dim=1)

    captions = []
    itos = vocab.get_itos()
    for i in range(batch_size):
        caption = ""
        tokens = result[i, :]
        for token in tokens[1:]:
            if token == end_token:
                break
            if itos[token][0] == "'":
                caption = caption[:-1] + itos[token] + " "
            else:
                caption += itos[token] + " "
        captions.append(caption)
    return captions, y

NUM_LAYERS_ENCODER = [3, 4, 5, 6]
NUM_LAYERS_DECODER = [3, 4, 5, 6, 7, 8, 9, 10]
SEED_NUMBERS = [0]
DROPOUT_P = [0.5]

[train(DATASETS[0], num_layer_encoder, num_layer_decoder, seed_number, dropout_p) for num_layer_encoder in NUM_LAYERS_ENCODER for num_layer_decoder in NUM_LAYERS_DECODER for seed_number in SEED_NUMBERS for dropout_p in DROPOUT_P]



