# coding: utf8
# from __future__ import unicode_literals, print_function
import random
import cytoolz
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from pymongo import MongoClient
import numpy as np
from spacy.pipeline import TextCategorizer

class LSTMSentiment(nn.Module):

    def __init__(self, embedding_dim = 100, hidden_dim =100, vocab_size=259136, label_size=2, batch_size=5, dropout=0.5):
        super(LSTMSentiment, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.dropout = dropout
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # first is the hidden h
        # second is the cell c
        return (Variable(torch.zeros(1, self.batch_size, self.hidden_dim)),
                Variable(torch.zeros(1, self.batch_size, self.hidden_dim)))

    def forward(self, sentence):
        current_batchsize = sentence.shape[1]
        #print('sentence',sentence)
        print([torch.sum(sentence[:,i].float()==torch.zeros(sentence.shape[0])) for i in range(current_batchsize)])
        print(current_batchsize)
        x = self.embeddings(sentence).view(len(sentence), current_batchsize, -1)
        print(x.shape)
        lstm_out, self.hidden = self.lstm(x)
        y = self.hidden2label(lstm_out[-1])
        #print(y)
        log_probs = F.softmax(y)
        #print(log_probs)
        return log_probs

class FastText(nn.Module):
    """
    FastText model that implements https://arxiv.org/abs/1607.01759
    """
    def __init__(self, vocab_size, emb_dim):
        """
        @param vocab_size: size of the vocabulary. 
        @param emb_dim: size of the word embedding
        """
        super(FastText, self).__init__()
        # Note that the # of inputs dimension for embedding shall be vocab_size+1, why?
        # In the embedding, you need to set the padding_dx argument.
        # Please see http://pytorch.org/docs/master/nn.html
        self.embeds = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.linear = nn.Linear(emb_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, data,length):
        """
        @param data: matrix of size (batch_size, max_sentence_length). Each row in data represents a 
            review that is represented using n-gram index. Note that they are padded to have same length.
        @param length: an int tensor of size (batch_size), which represents the non-trivial (excludes padding)
            length of each sentences in the data.
        """
        #import ipdb;ipdb.set_trace()
        #print('batch_size',len(data))
        #print(data.shape)
        all_embs = self.embeds(data)
        sum_emb = torch.sum(all_embs, 1)
        # Note that instead of doing tensorwise mean, you need to calculate the sum and divide by the sentence length
        # in the length tensor.
        avg_emb = sum_emb / length.float().view(-1,1)
        out = self.linear(avg_emb)
        out = self.sigmoid(out.view(-1))
        return out
   

class Loss_TextCategorizer(TextCategorizer):
    name = 'textcat'
    def __call__(self, doc):
        #print("^^new call")
        scores, tensors = self.predict([doc])
        self.set_annotations([doc], scores, tensors=tensors)
        return doc

    def predict(self, docs):
        #import ipdb;ipdb.set_trace()
        print("^^new predict")
        scores = self.model(docs)
        scores = self.model.ops.asarray(scores)
        tensors = [doc.tensor for doc in docs]
        return scores, tensors
    def set_annotations(self, docs, scores, tensors=None):
        for i, doc in enumerate(docs):
            #for j, label in enumerate(self.labels):
            doc.cats['POSITIVE'] = float(scores[i])
            doc.cats['NEGATIVE'] = 1-float(scores[i])
            
    def update(self, docs, golds, state=None, drop=0., sgd=None, losses=None):
        #print("^^new update")
        scores, bp_scores = self.model.begin_update(docs, drop=drop)
        #loss, d_scores = self.get_loss(docs, golds, scores)
        truths = self.get_truth(docs, golds, scores)
        #print('sgd',sgd,sgd.b1,sgd.b2)
        loss = bp_scores(truths,sgd=sgd)
        print('loss',loss)
        if losses is not None:
            losses.setdefault(self.name, 0.0)
            losses[self.name] += loss

    def get_loss(self, docs, golds, scores):
        #print("^^new get_loss")
        truths = np.zeros((len(golds), len(self.labels)), dtype='f')
        not_missing = np.ones((len(golds), len(self.labels)), dtype='f')
        for i, gold in enumerate(golds):
            for j,label in enumerate(self.labels):
                #print("label",label,"gold",gold.cats)
                if label in gold.cats:
                    truths[i, j] = gold.cats[label]
                else:
                    not_missing[i, j] = 0.
        truths = self.model.ops.asarray(truths)
        not_missing = self.model.ops.asarray(not_missing)
        #print("scores shape",scores.shape,type(scores))
        #print("truth",truths.shape,type(truths))
        #d_scores = (scores-truths) / scores.shape[0]
        #print("BCEloss")
        if(0 in scores or 0 in 1-scores):
            d_scores = (-(truths+1e-36)/(scores+1e-36) + (1-truths+1e-36)/(1-scores+1e-36))/scores.shape[0]
        else:
            d_scores = (-truths/scores + (1-truths)/(1-scores))/scores.shape[0]
        d_scores *= not_missing
        #print(d_scores)
        mean_square_error = ((scores-truths)**2).sum(axis=1).mean()
        BCEloss = nn.BCELoss()
        bceloss = BCEloss(torch.Tensor(scores),torch.Tensor(truths))
        return bceloss, d_scores
    def get_truth(self, docs, golds, scores):
        #print("^^new get_loss")
        truths = np.zeros((len(golds), 1), dtype='f')
        not_missing = np.ones((len(golds), len(self.labels)), dtype='f')
        for i, gold in enumerate(golds):
            if ('POSITIVE' in gold.cats and gold.cats['POSITIVE'] == 1) or ('NEGATIVE' in gold.cats and gold.cats['NEGATIVE'] == 0):
                truths[i] = 1
            else:
                truths[i] = 0
        truths = self.model.ops.asarray(truths)
        return truths