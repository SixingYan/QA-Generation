"""
模型：
输入用户检索词（词语，语义特征），输出数据库检索词（词语，语义特征）

训练
输入问题，target：对应答案，和标注类型

评价：
已标注知识点的数据集
看能不能搜索到这些知识点的索引
"""
from typing import List, Dict
import numpy as np
import time
import pickle
import random
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import KFold

#####################################################################

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1
START_TAG = 0  # '<START>'
STOP_TAG = 1  # '<STOP>'

MAX_LENGTH = 256
TRAIN_EPOCHS = 3
HIDDEN_DIM = 5  # 100
NUM_UNITS = 2
SEED = 1
FOLD_NUM = 2
EMBEDDING_DIM = 5
LEARN_RATE = 0.05
DROPOUT = 0.01
POS_DIM = 5
NER_DIM = 5
#####################################################################
local = True
if local:
    PATH = '/Users/alfonso/workplace/qasystem/movieqa/'
    DATA_PATH = PATH + 'data/'
    GLOVE_PATH = '/Users/alfonso/workplace/qasystem/generateQA/' + \
        'embedding/glove.6B.50d.txt'
    SAVE_EP_PATH = PATH + 'data/state_epoch.ck'
    SAVE_KF_PATH = PATH + 'data/state_fold.ck'

else:
    PATH = '/content/drive/'
    DATA_PATH = PATH + 'Colab/tmpdata/'
    GLOVE_PATH = PATH + 'embedding/glove.6B.50d.txt'
    SAVE_EP_PATH = '/content/drive/Colab/checkpoint/state_epoch.ck'
    SAVE_KF_PATH = '/content/drive/Colab/checkpoint/state_fold.ck'

#####################################################################


def to_tensor(indexes):
    return torch.tensor(indexes, dtype=torch.long, device=DEVICE).view(-1, 1)


def savep(var, vname: str, path='temp/'):
    with open(path + vname + '.pickle', 'wb') as f:
        pickle.dump(var, f)


def loadp(vname: str, path='temp/'):
    with open(path + vname + '.pickle', 'rb') as f:
        var = pickle.load(f)
    return var


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





class Encoder(nn.Module):

    def __init__(self,
                 vocab_size,
                 word_embed_dim,
                 hidden_dim,
                 num_layers=1,
                 pos_embed_dim=None,
                 pos_size=None,
                 ner_embed_dim=None,
                 ner_size=None,
                 pre_word_embeds=None
                 ):
        super(Encoder, self).__init__()
        # encoder: BiGRU
        self.pos_embed_dim = pos_embed_dim
        self.ner_embed_dim = ner_embed_dim
        self.hidden_dim = hidden_dim
        self.word_embeds = nn.Embedding(vocab_size, word_embed_dim)
        if pre_word_embeds is not None:
            self.pre_word_embeds = True
            self.word_embeds.weight = nn.Parameter(
                torch.FloatTensor(pre_word_embeds))
            self.word_embeds.weight.requires_grad = False
        else:
            self.pre_word_embeds = False

        embed_dim = word_embed_dim

        if pos_embed_dim is not None and pos_size is not None:
            self.pos_embeds = nn.Embedding(pos_size, pos_embed_dim)
            embed_dim += pos_embed_dim

        if ner_embed_dim is not None and ner_size is not None:
            self.ner_embeds = nn.Embedding(ner_size, ner_embed_dim)
            embed_dim += ner_embed_dim

        # 里面训练两个GRU用于把其他特征映射后，拼在结尾

        self.nn_word = nn.GRU(embed_dim, hidden_dim,
                              num_layers=num_layers,
                              bidirectional=True)

        if pos_embed_dim is not None and pos_size is not None:
            self.nn_pos = nn.GRU(embed_dim, hidden_dim,
                                 num_layers=num_layers,
                                 bidirectional=True)

        if ner_embed_dim is not None and ner_size is not None:
            self.nn_ner = nn.GRU(embed_dim, hidden_dim,
                                 num_layers=num_layers,
                                 bidirectional=True)

    def _get_feature(self, words, ners=None, poss=None):
        embeds = self.word_embeds(words).view(
            len(words), 1, -1)

        if self.pos_embed_dim is not None and poss is not None:
            pos_embeds = self.word_embeds(poss).view(
                len(poss), 1, -1)
            embeds = torch.cat((embeds, pos_embeds), 2)

        if self.ner_embed_dim is not None and ners is not None:
            ner_embeds = self.word_embeds(ners).view(
                len(ners), 1, -1)
            embeds = torch.cat((embeds, ner_embeds), 2)

        return embeds

    def forward(self, words, ners=None, poss=None):
        # 注意它的输入输出size，这里直接读的一整个词语tensor
        embeds = self._get_feature(words, ners, poss)

        # 这直接传入一整个序列，不需要额外的hidden，但是输出的hidden要作为decode的输入之一
        output_word, hidden_word = self.nn_word(embeds)
        output, hidden = output_word, hidden_word

        if self.pos_embed_dim is not None and poss is not None:
            output_ner, hidden_ner = self.nn_ner(embeds)
            output = torch.cat((output, output_ner), 2)
            hidden = torch.cat((hidden, hidden_ner), 2)

        if self.ner_embed_dim is not None and ners is not None:
            output_pos, hidden_pos = self.nn_pos(embeds)
            output = torch.cat((output, output_pos), 2)
            hidden = torch.cat((hidden, hidden_pos), 2)

        return output, hidden


class Attention(nn.Module):

    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.weight = None  # encoder_outputs
        self.attn = nn.Linear(hidden_size * 2, MAX_LENGTH)
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, embedded, hidden):
        print(embedded.size())
        print(hidden.size())
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        # attn_weights 1 by self.max_length tensor
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 self.weight.unsqueeze(0))
        # output size 1 by (hidden_size + hidden_size)
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        # attn_combine 线性映射 hidden_size*2 -> hidden_size
        # output size 1 by 1 by hidden_size
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)

        return output, attn_weights

    def initWeight(self, encoder_outputs):
        self.weight = encoder_outputs


class Decoder(nn.Module):

    def __init__(self,
                 hidden_dim,  # 这里是 encoder.h_dim * 3
                 output_dim,
                 num_layers=1,
                 dropout_rate=0.1,
                 has_ner=True,
                 has_pos=True):
        # 这里还是需要max_length作为转接，因为attention的矩阵的size必须是固定的
        super(Decoder, self).__init__()
        self.has_ner = has_ner
        self.has_pos = has_pos

        self.word_embeds = nn.Embedding(vocab_size, word_embed_dim)
        #if pre_word_embeds is not None:
        #    self.pre_word_embeds = True
        #    self.word_embeds.weight = nn.Parameter(
        #        torch.FloatTensor(pre_word_embeds))
        #    self.word_embeds.weight.requires_grad = False
        #else:
        #    self.pre_word_embeds = False

        embed_dim = word_embed_dim

        if pos_embed_dim is not None and pos_size is not None:
            self.pos_embeds = nn.Embedding(pos_size, pos_embed_dim)
            embed_dim += pos_embed_dim

        if ner_embed_dim is not None and ner_size is not None:
            self.ner_embeds = nn.Embedding(ner_size, ner_embed_dim)
            embed_dim += ner_embed_dim


        #self.embedding = nn.Embedding(output_dim, hidden_dim)

        self.gru = nn.GRU(hidden_dim, hidden_dim,
                          num_layers=num_layers, bidirectional=True)

        self.dropout = nn.Dropout(dropout_rate)
        self.attention = Attention(hidden_dim)

        self.out_word = nn.Linear(hidden_dim, output_dim)

        self.out_ner = nn.Linear(hidden_dim, output_dim)
        self.out_pos = nn.Linear(hidden_dim, output_dim)
    
    def _get_feature(self, words, ners=None, poss=None):
        embeds = self.word_embeds(words).view(
            len(words), 1, -1)

        if self.pos_embed_dim is not None and poss is not None:
            pos_embeds = self.word_embeds(poss).view(
                len(poss), 1, -1)
            embeds = torch.cat((embeds, pos_embeds), 2)

        if self.ner_embed_dim is not None and ners is not None:
            ner_embeds = self.word_embeds(ners).view(
                len(ners), 1, -1)
            embeds = torch.cat((embeds, ner_embeds), 2)

        return embeds

    def forward(self, word_ts, ner_ts, pos_ts, hidden):


        word_embeds = 


        embedded = self.embedding(input_tensor).view(
            1, 1, -1)  # 这里是一个个词的输入，因为要用到输出
        embedded = self.dropout(embedded)

        output, attn_weights = self.attention(embedded, hidden)

        output, hidden = self.gru(output, hidden)
        # output[0] 1 by hidden_size,
        output_word = F.log_softmax(self.out_word(output[0]), dim=1)
        output_final = output_word

        output_ner = None
        if self.has_ner:
            output_ner = F.log_softmax(self.out_ner(output[0]), dim=1)
            output_final = torch.cat((output_final, output_ner), 1)

        output_pos = None
        if self.has_pos:
            output_pos = F.log_softmax(self.out_pos(output[0]), dim=1)
            output_final = torch.cat((output_final, output_pos), 1)

        return (output_word, output_ner, output_pos), output_final, hidden, attn_weights

    def initWeight(self, encoder_outputs):
        self.attention.initWeight(encoder_outputs)

# 注意，encoder 的 output hidden是 *2的，因为用了双向gru
# def train_one(input_tensor, target_tensor, has_ner=False, has_pos=False):


def train_one(word_ts, word_tg,
              ner_ts=None, ner_tg=None,
              pos_ts=None, pos_tg=None, teacher_forcing_ratio=0.4):
    '''
        word_tensor, word_target
    '''
    length = word_ts.size(0)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    #####################################
    # encoder
    # encoder_hidden = encoder_m.initHidden()  # 其实模型自己应该会初始化一个hidden作为输入的
    output, encoder_hidden = encoder_m(word_ts, ner_ts, pos_ts)
    print('------')
    print(encoder_hidden.size())
    # 这里做cat 把encoder_output.size(0) 的长度加到max_length
    # encoder_output size max_length by encoder.hidden_size,
    if length < MAX_LENGTH:
        # print(output.size())
        # print(output.squeeze(1).size())
        # print(encoder_m.hidden_dim)
        # print(torch.zeros(
        #    (MAX_LENGTH - length, encoder_m.hidden_dim)).size())
        encoder_output = torch.cat((torch.zeros(
            (MAX_LENGTH - length, encoder_m.hidden_dim * 3 * 2), device=DEVICE), output.squeeze(1)), dim=0)
    else:  # 会在调用该方法之前先截断，保证小于等于MAX_LENGTH
        encoder_output = output

    #####################################
    # decoder
    # decode 的时候，每次需要把前一时刻的输出作为下一个时刻的输入，所以需要遍历
    # 1 by 3
    decoder_input = torch.tensor(
        [[SOS_token], [SOS_token], [SOS_token]], device=DEVICE)
    decoder_hidden = encoder_hidden
    decoder_m.initWeight(encoder_output)  # 这里其实是初始化了attention的参数

    loss = 0

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # 训练的时候，只输出target长度的结果，这个结果用于矫正模型的输出

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for ix in range(length):
            decoder_output_tuple, decoder_output, decoder_hidden, decoder_attention = decoder_m(
                decoder_input, decoder_hidden)
            word_output, ner_output, pos_output = decoder_output_tuple

            loss += criterion(word_output, word_tg[ix])
            loss += criterion(ner_output, ner_tg[ix])
            loss += criterion(pos_output, pos_tg[ix])

            decoder_input = word_ts
            decoder_input = torch.cat((decoder_input, ner_ts), 1)
            decoder_input = torch.cat((decoder_input, pos_ts), 1)

    else:
        for ix in range(length):
            # decoder_input tensor([[0]]) 这种size
            decoder_output_tuple, decoder_output, decoder_hidden, decoder_attention = decoder_m(
                decoder_input, decoder_hidden)

            word_output, ner_output, pos_output = decoder_output_tuple

            topv, topi = word_output.topk(1, dim=1)  # value, index
            word_input = topi.detach()
            decoder_input = word_input

            topv, topi = ner_output.topk(1, dim=1)  # value, index
            ner_input = topi.detach()
            decoder_input = torch.cat((decoder_input, ner_input), 1)

            topv, topi = pos_output.topk(1, dim=1)  # value, index
            pos_input = topi.detach()
            decoder_input = torch.cat((decoder_input, pos_input), 1)

            loss += criterion(word_output, word_tg[ix])
            loss += criterion(ner_output, ner_tg[ix])
            loss += criterion(pos_output, pos_tg[ix])

            if word_input.squeeze().item() == EOS_token or ner_input == EOS_token or pos_input == EOS_token:
                break

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / length


def train(train_ix: List[int], valid_ix: List[int]):
    '''
        这里是总的训练方法
    '''
    for epoch in range(TRAIN_EPOCHS):
        train_loss = 0
        start_time = time.time()
        for idx, ix in enumerate(train_ix):
            q, a = dataset[ix]
            word_ts, ner_ts, pos_ts = to_tensor(
                q['word']), to_tensor(q['ner']), to_tensor(q['pos'])
            word_tg, ner_tg, pos_tg = a['word'], a['ner'], a['pos']

            # 这里要保证截断
            if len(q['word']) > MAX_LENGTH:  # 暂时先采用 直接跳过的方式
                continue

            loss = train_one(word_ts, word_tg, ner_ts, ner_tg, pos_ts, pos_tg)
            train_loss += loss

            if (idx + 1) % 100 == 0:  # avoid 0
                print('------finish {:.2f} \t current loss {:.2f} \t used time {:.2f} mins \n'.format(
                    (idx + 1) / len(train_ix), train_loss / (idx + 1), (time.time() - start_time) / 60))

            break
        print('---epoch loss {:.2f} \t used time {:.2f} \n'.format(
            train_loss / len(train_ix), (time.time() - start_time) / 60))
        break
    print('The recent loss {:.2f}'.format(train_loss / len(train_ix)))


torch.manual_seed(SEED)
#####################################################################
# Load data
dataset = loadp('suQA_dataset', path=DATA_PATH)[:100]

word_to_ix = loadp('word_to_ix', path=DATA_PATH)

ner_to_ix = loadp('ner_to_ix', path=DATA_PATH)

pos_to_ix = loadp('pos_to_ix', path=DATA_PATH)

kfolds = KFold(n_splits=FOLD_NUM, shuffle=True,
               random_state=SEED).split(dataset)

# data description
start_time = time.time()
print('Dataset Size={} \t Word Size={} \t POS Size={} \t NER Size={} \n'.format(
    len(dataset), len(word_to_ix), len(pos_to_ix), len(ner_to_ix)))


word_embed_dim = EMBEDDING_DIM

#pre_word_embeds, word_embed_dim = load_wordEmbed(word_to_ix, GLOVE_PATH)
# print('Embedding Size={} \t Time={:.2f} mins \n'.format(
#    word_embed_dim, (time.time() - start_time) / 60))


#####################################################################
# Run training

encoder_m = Encoder(len(word_to_ix), word_embed_dim,
                    hidden_dim=HIDDEN_DIM, num_layers=NUM_UNITS,
                    pos_embed_dim=POS_DIM, pos_size=len(pos_to_ix),
                    ner_embed_dim=NER_DIM, ner_size=len(ner_to_ix))  # ,
# pre_word_embeds=pre_word_embeds)

# 如果加上了ner和pos，那么输出结果是 HIDDEN_DIM*3 ！！！！！会影响 decoder的结构

encoder_m.train()
encoder_optimizer = optim.SGD(encoder_m.parameters(), lr=LEARN_RATE)

decoder_m = Decoder(hidden_dim=HIDDEN_DIM * 3 * 2, output_dim=len(word_to_ix),
                    num_layers=NUM_UNITS, dropout_rate=DROPOUT,
                    has_ner=True, has_pos=True)
# * 2 是因为用了双向的
decoder_m.train()
decoder_optimizer = optim.SGD(decoder_m.parameters(), lr=LEARN_RATE)

criterion = nn.NLLLoss()


#####################################################################
# Run training
for fold_idx, (train_ix, valid_ix) in enumerate(kfolds):
    print('Fold {} ***********************************************'.format(fold_idx + 1))
    s_time = time.time()
    train(train_ix, valid_ix)
    print('*********************************************** Time {:.2f} mins \n'.format(
        (time.time() - s_time) / 60))
    checkpoint = {
        'encoder_m': encoder_m.state_dict(),
        'encoder_optimizer': encoder_optimizer.state_dict(),
        'decoder_m': encoder_m.state_dict(),
        'decoder_optimizer': encoder_optimizer.state_dict(),
    }
    break
    torch.save(checkpoint, SAVE_KF_PATH)
