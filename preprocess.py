# import torch
import pickle
from typing import List, Dict
import pandas as pd
import spacy
# python -m spacy download en_core_web_md
nlpWorker = spacy.load('en_core_web_md')
print('nlp worker ready!')
# for token in doc:
#    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
#          token.shape_, token.is_alpha, token.is_stop)


def savep(var, vname: str, path='temp/'):
    with open(path + vname + '.pickle', 'wb') as f:
        pickle.dump(var, f)


def loadp(vname: str, path='temp/'):
    with open(path + vname + '.pickle', 'rb') as f:
        var = pickle.load(f)
    return var

def item_to_idx(items: List, to_ix: Dict, to_item: Dict)->List:
    # 这里要手动加上结尾，开头会在训练的时候加，这里不加开头
    return [to_ix[it] for it in items]# + [1]  # add end mark which is always 1

'''
p = Processer()


def sent_to_tensor(sentence):
    indexes = [word_to_ix[w] for w in p.do(sentence)]
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=DEVICE).view(-1, 1)


def words_to_indexs(words):
    return [word_to_ix[w] for w in words]
'''

# 如何将support text 联系起来

#

# 新的索引
# START_TAG = '<START>'
# STOP_TAG = '<STOP>'


def extractQA_MoiveQA(rpath='', wpath=''):

    word_to_ix = {"<SOS>": 0, "<EOS>": 1}
    ner_to_ix = {"<START>": 0, "<STOP>": 1}
    pos_to_ix = {"<START>": 0, "<STOP>": 1}
    qadataset = []

    dataset = pd.read_json(rpath)
    for i in range(len(dataset)):
        answer = ' '.join(dataset['answers'][i]).lower()
        question = dataset['question'][i].lower()

        q = {'word': [], 'ner': [], 'pos': []}
        for tk in nlpWorker(question):
            if tk.pos_ == 'PUNCT':
                continue
            if tk.lemma_ == '-PRON-':
                continue
            if tk.is_stop is True:
                continue
            if len(tk.lemma_) == 1:
                continue
            q['word'].append(tk.lemma_)
            q['pos'].append(tk.pos_)
            q['ner'].append(tk.tag_)

            if tk.lemma_ not in word_to_ix:
                word_to_ix[tk.lemma_] = len(word_to_ix)

            if tk.pos_ not in pos_to_ix:
                pos_to_ix[tk.pos_] = len(pos_to_ix)

            if tk.tag_ not in ner_to_ix:
                ner_to_ix[tk.tag_] = len(ner_to_ix)

        a = {'word': [], 'ner': [], 'pos': []}
        for tk in nlpWorker(answer):
            if tk.pos_ == 'PUNCT':
                continue
            if tk.lemma_ == '-PRON-':
                continue
            if tk.is_stop is True:
                continue
            if len(tk.lemma_) == 1:
                continue
            a['word'].append(tk.lemma_)
            a['pos'].append(tk.pos_)
            a['ner'].append(tk.tag_)

            if tk.lemma_ not in word_to_ix:
                word_to_ix[tk.lemma_] = len(word_to_ix)

            if tk.pos_ not in pos_to_ix:
                pos_to_ix[tk.pos_] = len(pos_to_ix)

            if tk.tag_ not in ner_to_ix:
                ner_to_ix[tk.tag_] = len(ner_to_ix)

        if len(q['word']) == 0 or len(a['word']) == 0:
            continue
        qadataset.append((q, a))

    ix_to_word = {tp[1]: tp[0] for tp in word_to_ix.items()}
    ix_to_ner = {tp[1]: tp[0] for tp in ner_to_ix.items()}
    ix_to_pos = {tp[1]: tp[0] for tp in pos_to_ix.items()}

    savep(qadataset, 'qadataset', wpath)
    savep(ix_to_word, 'ix_to_word', wpath)
    savep(ix_to_pos, 'ix_to_pos', wpath)
    savep(ix_to_ner, 'ix_to_ner', wpath)
    savep(word_to_ix, 'word_to_ix', wpath)
    savep(pos_to_ix, 'pos_to_ix', wpath)
    savep(ner_to_ix, 'ner_to_ix', wpath)

    print('length is:')
    print(len(qadataset))


def data_to_indexes_mqa(rpath='', wpath='', name='suqa'):

    word_to_ix = loadp('word_to_ix', rpath)
    pos_to_ix = loadp('pos_to_ix', rpath)
    ner_to_ix = loadp('ner_to_ix', rpath)
    ix_to_word = loadp('ix_to_word', rpath)
    ix_to_pos = loadp('ix_to_pos', rpath)
    ix_to_ner = loadp('ix_to_ner', rpath)
    qadataset = loadp('qadataset', rpath)

    dataset = []
    for q, a in qadataset:
        q['word'] = item_to_idx(q['word'], word_to_ix, ix_to_word)
        a['word'] = item_to_idx(a['word'], word_to_ix, ix_to_word)
        q['pos'] = item_to_idx(q['pos'], pos_to_ix, ix_to_pos)
        a['pos'] = item_to_idx(a['pos'], pos_to_ix, ix_to_pos)
        q['ner'] = item_to_idx(q['ner'], ner_to_ix, ix_to_ner)
        a['ner'] = item_to_idx(a['ner'], ner_to_ix, ix_to_ner)

        dataset.append((q, a))

    savep(dataset, name + '_dataset', wpath)

    print(len(dataset))


def main_mqa():
    path1 = '/Users/alfonso/workplace/qasystem/MovieQA_benchmark/data/qa.json'
    path2 = '/Users/alfonso/workplace/qasystem/movieqa/data_mqa/'

    extractQA_MoiveQA(path1, path2)
    print('---------------finish step 1')
    data_to_indexes_mqa(path2, path2, 'mqa')

"""
def extractQA_SuQA(rpath='', wpath=''):
    # this function extract SuQA
    word_to_ix = {"<SOS>": 0, "<EOS>": 1}
    ner_to_ix = {"<START>": 0, "<STOP>": 1}
    pos_to_ix = {"<START>": 0, "<STOP>": 1}
    qadataset = {}
    # visit the whole data
    dataset = pd.read_json(rpath)
    for atc in dataset.data:
        qadata = []
        for para in atc['paragraphs']:
            for qas in para['qas']:
                if True:  # for qa in qas:
                    if qas['is_impossible'] == True:
                        continue
                    question = qas['question']
                    answer = qas['answers'][-1]['text']
                    q = {'word': [], 'ner': [], 'pos': []}
                    for tk in nlpWorker(question):
                        if tk.pos_ == 'PUNCT':
                            continue
                        if tk.lemma_ == '-PRON-':
                            continue
                        q['word'].append(tk.lemma_)
                        q['pos'].append(tk.pos_)
                        q['ner'].append(tk.tag_)

                        if tk.lemma_ not in word_to_ix:
                            word_to_ix[tk.lemma_] = len(word_to_ix)

                        if tk.pos_ not in pos_to_ix:
                            pos_to_ix[tk.pos_] = len(pos_to_ix)

                        if tk.tag_ not in ner_to_ix:
                            ner_to_ix[tk.tag_] = len(ner_to_ix)
                        # break

                    a = {'word': [], 'ner': [], 'pos': []}
                    for tk in nlpWorker(answer):
                        if tk.pos_ == 'PUNCT':
                            continue
                        if tk.lemma_ == '-PRON-':
                            continue
                        a['word'].append(tk.lemma_)
                        a['pos'].append(tk.pos_)
                        a['ner'].append(tk.tag_)

                        if tk.lemma_ not in word_to_ix:
                            word_to_ix[tk.lemma_] = len(word_to_ix)

                        if tk.pos_ not in pos_to_ix:
                            pos_to_ix[tk.pos_] = len(pos_to_ix)

                        if tk.tag_ not in ner_to_ix:
                            ner_to_ix[tk.tag_] = len(ner_to_ix)
                        # break
                    # print((q, a))
                    qadata.append((q, a))
                    # break
                # break
            qadataset[atc['title']] = qadata
            # break

    # ix_to_items
    ix_to_word = {tp[1]: tp[0] for tp in word_to_ix.items()}
    ix_to_ner = {tp[1]: tp[0] for tp in ner_to_ix.items()}
    ix_to_pos = {tp[1]: tp[0] for tp in pos_to_ix.items()}

    savep(qadataset, 'qadataset', wpath)
    savep(ix_to_word, 'ix_to_word', wpath)
    savep(ix_to_pos, 'ix_to_pos', wpath)
    savep(ix_to_ner, 'ix_to_ner', wpath)
    savep(word_to_ix, 'word_to_ix', wpath)
    savep(pos_to_ix, 'pos_to_ix', wpath)
    savep(ner_to_ix, 'ner_to_ix', wpath)

    print('length is:')
    print(len(qadataset))



def data_to_indexes(rpath='', wpath='', name='suqa'):

    word_to_ix = loadp('word_to_ix', rpath)
    pos_to_ix = loadp('pos_to_ix', rpath)
    ner_to_ix = loadp('ner_to_ix', rpath)
    ix_to_word = loadp('ix_to_word', rpath)
    ix_to_pos = loadp('ix_to_pos', rpath)
    ix_to_ner = loadp('ix_to_ner', rpath)
    qadataset = loadp('qadataset', rpath)

    dataset = []
    for title, qadata in qadataset.items():
        for q, a in qadata:
            q['word'] = item_to_idx(q['word'], word_to_ix, ix_to_word)
            a['word'] = item_to_idx(a['word'], word_to_ix, ix_to_word)
            q['pos'] = item_to_idx(q['pos'], pos_to_ix, ix_to_pos)
            a['pos'] = item_to_idx(a['pos'], pos_to_ix, ix_to_pos)
            q['ner'] = item_to_idx(q['ner'], ner_to_ix, ix_to_ner)
            a['ner'] = item_to_idx(a['ner'], ner_to_ix, ix_to_ner)

            dataset.append((q, a))

    savep(dataset, name + '_dataset', wpath)

    print(len(dataset))


def main():
    path1 = '/Users/alfonso/workplace/qasystem/generateQA/data/train-v2.0.json'
    path2 = '/Users/alfonso/workplace/qasystem/movieqa/data/'

    extractQA_SuQA(path1, path2)
    print('---------------finish step 1')
    data_to_indexes(path2, path2, 'suQA')

"""
if __name__ == '__main__':
    main_mqa()
