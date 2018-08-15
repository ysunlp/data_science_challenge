# coding: utf8
# from __future__ import unicode_literals, print_function

import random
import cytoolz
import tqdm
import json
import prodigy
import numpy as np
import os.path
from pymongo import MongoClient
from datetime import datetime

import spacy
from spacy.util import minibatch, compounding

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer

from allennlp.data import Vocabulary
from allennlp.data.dataset import Batch
from typing import Dict
from data_reader import IMDBDatasetReader


def setup_mongo(collection_name, db_name = "prodigy"):
    mongo_uri = "mongodb://gbonev:e9aGdgaDdn67DoP@svpr-anl05:27017/admin?authSource=admin"
    client = MongoClient(mongo_uri)
    db = client[db_name]
    coll = db[collection_name]
    return coll    
class ProdigyWrapper():
    def __init__(self,model,label, vectorizer_path, example_file,track_dataset):
        self.labels = label.split(",") if type(label) ==str else label
        print(self.labels)
        self.model = model
        self.eval_set = None
        self.vectorizer = CountVectorizer(min_df=5, max_df = 0.8)
        self.init_vectorizer(vectorizer_path)
        self.init_model(example_file)
        self.init_track(track_dataset)

    def init_track(self,collection_name):
        if(collection_name):
            self.coll = setup_mongo(collection_name)
        else:
            self.coll = None

    def init_model(self,file_path):
        if(file_path):
            content, data = self.preprocess(file_path,1)
            label = self.get_label(data)
            self.model.partial_fit(content,label,[0,1])

    def init_vectorizer(self,file_path):
        content = []
        with open(file_path, "r") as data_file:
            for line in data_file:
                line = line.strip("\n")
                if not line:
                    continue
                review = json.loads(line)
                content.append(review['text'])
        self.vectorizer = self.vectorizer.fit(content)

    def init_eval(self,file_path):
        self.eval_set,data = self.preprocess(file_path,1)
        self.eval_label = self.get_label(data)

    def preprocess(self,data,result_flag=0):
        content = []
        data_dict = []
        if(type(data) == str and os.path.isfile(data)):
            with open(data, "r") as data_file:
                for line in data_file:
                    line = line.strip("\n")
                    if not line:
                        continue
                    review = json.loads(line)
                    data_dict.append(review)
                    content.append(review['text'])
        else:
            data_dict = data
            for review in data:
                content.append(review['text'])
        train_vector = self.vectorizer.transform(content)
        if(result_flag):
            return train_vector.toarray(),data_dict
        else:
            return train_vector.toarray()

    def model_score(self,data,params = None):
        # input: one batch of raw data
        xdata = self.preprocess(data)
        score = self.model.predict_proba(xdata)[:,1]
        # score = self.model.decision_function(xdata)
        # score = 1 / (1 + np.exp(-0.001*score))
        #score = 1 / (1 + np.exp(-score))
        return score

    def model_backprop(self,data,label):
        self.model.partial_fit(data,label)

    def predict(self,data,batch_size = 32):
        data = list(data)
        print('whole data size',len(data))
        for batch in cytoolz.partition_all(batch_size,data):
            batch = list(batch)
            score = self.model_score(batch)
            for i in range(len(batch)):
                #import ipdb;ipdb.set_trace()
                if('score' in list(self.coll.find({'text': batch[i]['text']}))[0].keys()):
                    current_score =  list(self.coll.find({'text': batch[i]['text']}))[0]['score']
                else:
                    current_score = []
                current_score.append(float(score[i]))
                self.coll.update_one({"text": batch[i]['text']},
                                {"$set": {"score": current_score}})
                #batch[i]['score'] = float(score[i])
                #print(batch[i]['answer'], float(score[i]))
                yield (float(score[i]),batch[i])

    def update(self,annotation):
        #import ipdb;ipdb.set_trace()
        truth = self.get_label(annotation)
        data = self.preprocess(annotation)
        self.model_backprop(data,truth)
        if(self.coll):
            for d in annotation:
                self.coll.update_one({"text": d['text']},
                                {"$set": {"iter": d['iter']}})


    def evaluate(self,data,batch_size = 32):
        if(self.eval_set == None):
            self.init_eval(data)
        score = self.model.score(self.eval_set,self.eval_label)
        return score

    def get_label(self,annotation):
        label = np.zeros((len(annotation),len(self.labels)))
        if(len(self.labels) == 1): # binary classification with one label
            for i,result in enumerate(annotation):
                label[i] = 1 if result['answer']=='accept' else 0
        elif(len(self.labels) == 2): # binary classification with two labels
            for i,result, in enumerate(annotation):
                label[i] == 1 if (result['label']==self.labels[0] and result['answer'] == 'accept') or (result['label']==self.labels[1] and result['answer'] == 'reject') else 0
        else: #multi-classification
            for i,result, in enumerate(annotation):
                for j in range(len(self.labels)):
                    if (result['label']==self.labels[j] and result['answer'] == 'accept'):
                        label[i][j] == 1 
                    else:
                        # didn't consider reject situation, should assign -1 if the label is reject to offer information to model.
                        label[i][j] == 0
        return label

class ProdigyWrapper_nn():
    def __init__(self,model,label,vocab_path,example_file,optimizer,loss,track_dataset=None):
        self.labels = label.split(",") if type(label) ==str else label
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.datareader = IMDBDatasetReader(self.labels)
        self.init_vocab(vocab_path)
        self.init_track(track_dataset)
        self.init_model(example_file)
        self.eval_set = None
    def init_model(self,example_file):
        if(example_file and os.path.isfile(example_file)):
            annotation = []
            with open(example_file, "r") as data_file:
                for line in data_file:
                    line = line.strip("\n")
                    if not line:
                        continue
                    review = json.loads(line)
                    annotation.append(review)
            loss = self.model_backprop(annotation)
        else:
            pass
    def init_track(self,collection_name):
        if(collection_name):
            self.coll = setup_mongo(collection_name)
        else:
            self.coll = None
    def init_vocab(self,file_path):
        # datareader = IMDBDatasetReader(1)
        if(file_path and os.path.isfile(file_path)):
            instance = self.datareader.read(file_path)
            self.vocab = Vocabulary.from_instances(instance)
        else:
            self.vocab = None
    def init_eval(self,file_path):
        if file_path:
            annotation = []
            with open(file_path, "r") as data_file:
                for line in data_file:
                    line = line.strip("\n")
                    if not line:
                        continue
                    review = json.loads(line)
                    annotation.append(review)
            self.eval_set = self.preprocess(annotation,1)

    def preprocess(self,data,result_flag=0):
        xdata =[]
        for review in data:
            if(result_flag):
                xlabel = self.get_label(review)
            else: # fake label when predicting
                xlabel = -1
            xdata.append(self.datareader.text_to_instance(review['text'],xlabel))
        data_batch = Batch(xdata)
        data_batch.index_instances(self.vocab)
        data_tensors = data_batch.as_tensor_dict(data_batch.get_padding_lengths())
        return data_tensors
    def model_score(self,data,params = None):
        # input: one batch of raw data
        data_tensor= self.preprocess(data)
        data_batch = torch.autograd.Variable(data_tensor['text']['tokens'])
        length_batch = torch.autograd.Variable(data_tensor['length'])
        score = self.model(data_batch,length_batch)
        return score
    def model_backprop(self,data):
        self.optimizer.zero_grad()
        data_tensor= self.preprocess(data,1)
        data_batch = torch.autograd.Variable(data_tensor['text']['tokens'])
        label_batch = data_tensor['label']
        length_batch = torch.autograd.Variable(data_tensor['length'])
        #print(type(xdata),type(length_list))
        output = self.model(data_batch,length_batch)
        #print(type(output))
        loss = self.loss(output,torch.autograd.Variable(label_batch).float())
        loss.backward()
        self.optimizer.step()
        #print(type(label))
        return loss

    def predict(self,data):
        data = list(data)
        #batches = minibatch(data,size=compounding(10,128,1.3))
        batches = minibatch(data,size=32)
        for batch in batches:
            print("batch size: ",len(batch))
            batch = list(batch)
            score = self.model_score(batch)
            for i in range(len(batch)):
                if('label' not in batch[i].keys()):
                    batch[i]['label'] = self.labels[0]
                if(self.coll):
                    if('score' in list(self.coll.find({'text': batch[i]['text']}))[0].keys()):
                        current_score =  list(self.coll.find({'text': batch[i]['text']}))[0]['score']
                    else:
                        current_score = []
                    current_score.append(float(score[i]))
                    self.coll.update_one({"text": batch[i]['text']}, {"$set": {"score": current_score}})
                yield (float(score[i]),batch[i])

    def update(self,annotation):
        loss = self.model_backprop(annotation)
        if(self.coll and 'iter' in annotation[0].keys()):
            for d in annotation:
                self.coll.update_one({"text": d['text']},
                                {"$set": {"iter": d['iter']}})
        return loss
    def evaluate(self,data,batch_size = 32):
        correct = 0
        total = 0
        self.model.eval() 
        if(self.eval_set == None):
            self.init_eval(data)
        output = self.model(self.eval_set['text']['tokens'],self.eval_set['length'])
        labels = self.eval_set['label']
        predicted = (output.data > 0.5).long().view(-1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        self.model.train()
        return (100 * correct / total)

    def get_label(self,result):
        if(len(self.labels) == 1): # binary classification with one label
            label = 1 if result['answer']=='accept' else 0
        elif(len(self.labels) == 2): # binary classification with two labels
            label == 1 if (result['label']==self.labels[0] and result['answer'] == 'accept') or (result['label']==self.labels[1] and result['answer'] == 'reject') else 0
        else: #multi-classification, need modification
            label = np.zeros(self.labels)
            for j in range(len(self.labels)):
                if (result['label']==self.labels[j] and result['answer'] == 'accept'):
                    label[j] == 1 
                else:
                    # didn't consider reject situation, should assign -1 if the label is reject to offer information to model.
                    label[j] == 0
        return label








