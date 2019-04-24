import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
import time
from typing import List
#####################################################################
# Model


class LinearRank(object):
    """docstring for DNN"""

    def __init__(self, score_size):
        super(LinearRank, self).__init__()
        self.weight = nn.Parameter(
            torch.FloatTensor((score_size, 1)))  # s by 1
        torch.nn.init.xavier_uniform_(self.weight)

    def rankloss(self, scores, target: int):
        '''
        '''
        sums = nn.Functional.Linear(scores, self.weight)  # N by 1
        inds = argsort(sums)  # rank
        pred = inds.index(target) + 1  # target rank
        loss = (1 - pred * sums[target])**2
        return loss

    def forward(self, scores):
        '''
            scores: N by s
        '''
        sums = nn.Functional.Linear(scores, self.weight)
        inds = argsort(sums)  # rank
        return inds


#####################################################################
# Setup training

model = LinearRank(score_size=sco_size)

# optimizer
optimizer = optim.RMSprop(
    model.parameters(), lr=LEARN_RATE, weight_decay=1e-4, momentum=0.9)


def train(train_ix: List[int], valid_ix: List[int]):
    for epoch in range(TRAIN_EPOCHS):
        train_loss = 0
        start_time = time.time()
        # loop data
        for idx, ix in enumerate(train_ix):
            data = dataset[ix]
            scores, target = data
            model.zero_grad()

            loss = model.rankloss(scores, target)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        elapsed_time = time.time() - start_time
        train_loss /= len(train_ix)

#####################################################################
# Run training

