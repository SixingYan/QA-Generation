# -*- coding: utf-8 -*-

# 取消 < EOS > 结尾，防止早停
import random
import time
import math
import pickle
from typing import Dict, List
import numpy as np
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1

MAX_LENGTH = 100
EMBEDDING_DIM = 50
DATA_PATH = '/Users/alfonso/workplace/qasystem/movieqa/data_mqa/'
SEED = 1
FOLD_NUM = 4
TRAIN_EPOCHS = 4
LEARN_RATE = 0.05
DROPOUT = 0.01
SAVE_EP_PATH = '/content/drive/Colab/checkpoint/mqa/state_epoch.ck'
SAVE_KF_PATH = '/content/drive/Colab/checkpoint/mqa/state_fold.ck'
local = True
if local:
    PATH = '/Users/alfonso/workplace/qasystem/generateQA/'
    DATA_PATH = '/Users/alfonso/workplace/qasystem/movieqa/data_mqa/'
    GLOVE_PATH = PATH + 'embedding/glove.6B.50d.txt'
    FOLD_NUM = 2
else:
    PATH = '/content/drive/'
    DATA_PATH = PATH + 'Colab/data_mqa/'
    GLOVE_PATH = PATH + 'embedding/glove.6B.50d.txt'


def savep(var, vname: str, path='temp/'):
    with open(path + vname + '.pickle', 'wb') as f:
        pickle.dump(var, f)


def loadp(vname: str, path='temp/'):
    with open(path + vname + '.pickle', 'rb') as f:
        var = pickle.load(f)
    return var


def index_to_item(data, to_item):
    return [to_item[d] for d in data]


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def tensorsFromPair(pair, name: str):
    #print(pair)
    input_tensor = torch.tensor(
        pair[0][name], dtype=torch.long, device=device).view(-1, 1)
    target_tensor = torch.tensor(
        pair[1][name], dtype=torch.long, device=device).view(-1, 1)
    return (input_tensor, target_tensor)


def load_wordEmbed(word_index: Dict,
                   embed_path: str,
                   max_features: int=-1):
    # read
    def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(embed_path))
    # init
    all_embs = np.stack(list(embeddings_index.values()))
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]
    nb_words = min(max_features, len(word_index)
                   ) if max_features != -1 else len(word_index)
    embedding_matrix = np.random.normal(
        emb_mean, emb_std, (nb_words, embed_size))
    # mapping
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix, embed_size


class EncoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size, pre_word_embeds=None):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        if pre_word_embeds is not None:
            self.pre_word_embeds = True
            self.embedding.weight = nn.Parameter(
                torch.FloatTensor(pre_word_embeds))
            self.embedding.weight.requires_grad = False
        else:
            self.pre_word_embeds = False

        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):

    def __init__(self, hidden_size, output_size, dropout_p=DROPOUT, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        # embedding 的size是 1 by 1 by hidden_size
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)

        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights  # 这里把attn也返回了，应该是作用于后面的可视化

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


teacher_forcing_ratio = 0.1


def train(input_tensor, target_tensor,
          encoder, decoder,
          encoder_optimizer, decoder_optimizer,
          criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(
        max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]
    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):  # let it generate
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            # if decoder_input.item() == EOS_token:
            #    break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def trainIters(pairs, name, encoder, decoder,
               encoder_optimizer, decoder_optimizer,
               print_every=10):

    start = time.time()
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    training_pairs = [tensorsFromPair(p, name) for p in pairs]

    criterion = nn.NLLLoss()

    for iter in range(1, len(pairs) + 1):  # 这里从1开始是为了print的时候方便一点
        input_tensor, target_tensor = training_pairs[iter - 1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / len(pairs)),
                                         iter, iter / len(pairs) * 100, print_loss_avg))


torch.manual_seed(SEED)

dataset = loadp('mqa_dataset', path=DATA_PATH)[:50]
_to_ix = loadp('word_to_ix', path=DATA_PATH)
_to_item = loadp('ix_to_word', path=DATA_PATH)
name = 'word'

pre_word_embeds = None
start_time = time.time()
pre_word_embeds, word_embed_dim = load_wordEmbed(_to_ix, GLOVE_PATH)
print('Embedding Size={} \t Time={:.2f} mins \n'.format(
    word_embed_dim, (time.time() - start_time) / 60))

hidden_size = EMBEDDING_DIM
encoder1 = EncoderRNN(len(_to_ix), hidden_size, pre_word_embeds).to(device)
attn_decoder1 = AttnDecoderRNN(
    hidden_size, len(_to_ix), dropout_p=0.1).to(device)

encoder_optimizer = optim.SGD(encoder1.parameters(), lr=LEARN_RATE)
decoder_optimizer = optim.SGD(attn_decoder1.parameters(), lr=LEARN_RATE)

for idx in range(TRAIN_EPOCHS):
    print('***************************')
    print('Epoch {}'.format(idx+1))
    random.shuffle(dataset)
    trainIters(dataset, name, encoder1, attn_decoder1,
           encoder_optimizer, decoder_optimizer)

    checkpoint = {
        'encoder': encoder1.state_dict(),
        'decoder': attn_decoder1.state_dict(),
        'encoder_optimizer': encoder_optimizer.state_dict(),
        'decoder_optimizer': decoder_optimizer.state_dict(),
    }

#   torch.save(checkpoint, SAVE_KF_PATH)

"""
for fold_idx, subset in enumerate(kfolds):
    print('Fold {} ***********************************************'.format(fold_idx + 1))
    s_time = time.time()
    trainIters(subset, name, encoder1, attn_decoder1,
               encoder_optimizer, decoder_optimizer)
    print('*********************************************** Time {:.2f} mins \n'.format(
        (time.time() - s_time) / 60))
    checkpoint = {
        'encoder': encoder1.state_dict(),
        'decoder': attn_decoder1.state_dict(),
        'encoder_optimizer': encoder_optimizer.state_dict(),
        'decoder_optimizer': decoder_optimizer.state_dict(),
    }

    #torch.save(checkpoint, SAVE_KF_PATH)
"""


def evaluate(encoder, decoder, indexes, max_length=MAX_LENGTH):
    with torch.no_grad():

        input_tensor = torch.tensor(
            indexes, dtype=torch.long, device=device).view(-1, 1)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(
            max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)

            # if topi.item() == EOS_token:
            #    decoded_words.append('<EOS>')
            #    break
            # else:
            if True:
                decoded_words.append(_to_item[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(pairs, name, encoder, decoder, nums=5):
    for i in range(nums):
        pair = random.choice(pairs)
        print('>', index_to_item(pair[0][name], _to_item))
        print('=', index_to_item(pair[1][name], _to_item))
        output_words, attentions = evaluate(encoder, decoder, pair[0][name])
        #output_sentence = ' '.join(output_words)
        #print('<', index_to_item(output_words, _to_item))
        print('<', output_words)
        print('')


evaluateRandomly(dataset, name, encoder1, attn_decoder1)
