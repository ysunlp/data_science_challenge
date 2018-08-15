# coding: utf8

import spacy
import random
import cytoolz
import tqdm
import json
import numpy as np
import torch
import torch.nn as nn
from spacy.pipeline import TextCategorizer
from thinc.extra.wrappers import PyTorchWrapper

import prodigy
from prodigy.models.textcat import TextClassifier
from prodigy.components.loaders import JSONL
from prodigy.components import printers
from prodigy.components.loaders import get_stream
from prodigy.components.db import connect
from prodigy.components.sorters import prefer_uncertain
from prodigy.core import recipe, recipe_args
from prodigy.util import export_model_data, split_evals, get_print

from pymongo import MongoClient
from datetime import datetime
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB

from prodigy_wrapper import ProdigyWrapper,ProdigyWrapper_nn
from pytorch_model import FastText
def parse_model(name,label,vectorizer_path,init_path,track_dataset):
    if(name == "fasttext"):
        pt_model = FastText(vocab_size=50966, emb_dim = 300)
        optimizer = torch.optim.Adam(pt_model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        model = ProdigyWrapper_nn(pt_model,label,vectorizer_path,init_path,optimizer,criterion,track_dataset)
    else:
        if(name == "svm"):
            clf_model = linear_model.SGDClassifier(loss = "hinge")
        elif(name == "log"):
            clf_model = linear_model.SGDClassifier(loss = "log")
        elif(name == "gaussiannb"):
            clf_model = GaussianNB()
        else:
            pass
        model = ProdigyWrapper(clf_model,label,vectorizer_path,init_path,track_dataset)
    return model
def probability_stream(stream,predict):
    print("prefer_uncertain based on probability")
    stream = prefer_uncertain(predict(stream), algorithm="probability")
    return stream
def test_stream(stream,predict):
    stream = prefer_uncertain(predict(stream))
    print("prefer_uncertain based on ema")
    return stream


@recipe('textcat-teach',
        dataset=recipe_args['dataset'],
        model=recipe_args['spacy_model'],
        source=recipe_args['source'],
        label=recipe_args['label_set'],
        prefer_method=("type of prefer_uncertain algorithm", "option", "uncertain", str),
        exclude=recipe_args['exclude'],
        init_path=("filename of initial dataset source","option","init_file",str),
        vectorizer_path=("filename of dataset for build vocabulary","option","vec_file",str),
        track_dataset=("collection name of track log of score","option","track",str))
def textcat_teach(dataset, model,source=None,label='',prefer_method=False, exclude=None, init_path=None, vectorizer_path=None,track_dataset=None,exit_model=0):
    """
    Collect the best possible training data for a text classification model
    with the model in the loop. Based on your annotations, Prodigy will decide
    which questions to ask next.
    """
    if(type(model) == str): #choose model, if not str, it's of prodigywrapper class
        print("Build your customized model", model)
        model = parse_model(model,label,vectorizer_path,init_path,track_dataset)
    predict = model.predict
    update = model.update

    stream = get_stream(source,input_key = 'text')
    if(prefer_method == "probability"):
        #probability
        stream = probability_stream(stream,predict)
    else:
        #exponential moving average
        stream = test_stream(stream,predict)

    def updateDB(answers):
        model.update(answers)
        
    def on_exit():
        print("on_exit")
        return model
    if(exit_model):
        return {
            'view_id': 'classification',
            'dataset': dataset,
            'stream': stream,
            'exclude': exclude,
            'update': updateDB,
            'on_exit': on_exit,
            'config': {'labels': model.labels,'batch_size':32}
        }
    else:
        return {
            'view_id': 'classification',
            'dataset': dataset,
            'stream': stream,
            'exclude': exclude,
            'update': updateDB,
            'config': {'labels': model.labels,'batch_size':32}
        }

@recipe('textcat_log',
        dataset=recipe_args['dataset'],
        spacy_model=recipe_args['spacy_model'],
        source=recipe_args['source'],
        label=recipe_args['label_set'],
        api=recipe_args['api'],
        loader=recipe_args['loader'],
        patterns=recipe_args['patterns'],
        long_text=("Long text", "flag", "L", bool),
        exclude=recipe_args['exclude'])
def textcat_log(dataset, spacy_model,source=None, label='', api=None, patterns=None,
          loader=None, long_text=False, exclude=None):
    """
    Collect the best possible training data for a text classification model
    with the model in the loop. Based on your annotations, Prodigy will decide
    which questions to ask next.
    """
    # logDB = setup_mongo('activelearning')
    #nlp = spacy.load('/home/ysun/pytorchprodigy/')
    if(spacy_model is not None):
        if(type(spacy_model) == str):
            print("Load model ",spacy_model)
            nlp=spacy.load(spacy_model, disable=['ner', 'parser'])
            model = TextClassifier(nlp, label, long_text=long_text)
        else:
            model = spacy_model
    else:
        print("build your customized model,log")
        pt_model = linear_model.SGDClassifier(loss="log")
        # pt_model = linear_model.SGDClassifier()
        example = ["Could you check my order status"]
        example_label = [1]
        #vectorizer_path = "/liveperson/data/alloy/prodigy/data/db-out/tmo_order_status.jsonl"
        #example_path = "/liveperson/data/alloy/prodigy/data/newsgroup_initial.jsonl"
        example_path = "/liveperson/data/alloy/prodigy/data/newsgroup_example.jsonl"
        vectorizer_path = "/liveperson/data/alloy/prodigy/data/newsgroup_all.jsonl"
        model = Prodigy_log_cpu(pt_model,1,vectorizer_path,example_path)


    stream = get_stream(source,input_key = 'text')
    if patterns is None:
        predict = model.predict
        update = model.update
    
    stream = test_stream(stream,predict)
    # stream = probability_stream(stream,predict)

    def updateDB(answers):
        model.update(answers)
        
    def on_exit():
        print("on_exit")
        return model
    
    return {
        'view_id': 'classification',
        'dataset': dataset,
        'stream': stream,
        'exclude': exclude,
        'update': updateDB,
        'on_exit': on_exit,
        'config': {'labels': ['ORDER_STATUS'],'batch_size':32}
    }
@recipe('textcat_custom',
        dataset=recipe_args['dataset'],
        spacy_model=recipe_args['spacy_model'],
        source=recipe_args['source'],
        label=recipe_args['label_set'],
        api=recipe_args['api'],
        loader=recipe_args['loader'],
        patterns=recipe_args['patterns'],
        long_text=("Long text", "flag", "L", bool),
        exclude=recipe_args['exclude'])
def textcat_custom(dataset, spacy_model,source=None, label='', api=None, patterns=None,
          loader=None, long_text=False, exclude=None):
    """
    Collect the best possible training data for a text classification model
    with the model in the loop. Based on your annotations, Prodigy will decide
    which questions to ask next.
    """
    # logDB = setup_mongo('activelearning')
    #nlp = spacy.load('/home/ysun/pytorchprodigy/')
    if(spacy_model is not None):
        if(type(spacy_model) == str):
            print("Load model ",spacy_model)
            nlp=spacy.load(spacy_model, disable=['ner', 'parser'])
            model = TextClassifier(nlp, label, long_text=long_text)
        else:
            model = spacy_model
    else:
        print("build your customized model")
        #nlp = spacy.load('en_core_web_lg')
        pt_model = FastText(vocab_size=50966, emb_dim = 300)
        #pt_model.embeds.weight.data.copy_(torch.from_numpy(nlp.vocab.vectors.data))
        #model = PyTorchWrapper(pt_model)

        #textcat = Loss_TextCategorizer(nlp.vocab,model)
        #nlp.add_pipe(textcat)
        #model = TextClassifier(nlp, label, long_text=long_text)
        optimizer = torch.optim.Adam(pt_model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        #example_path = "/liveperson/data/alloy/prodigy/data/newsgroup_initial.jsonl"
        example_path = "/liveperson/data/alloy/prodigy/data/newsgroup_example.jsonl"
        vectorizer_path = "/liveperson/data/alloy/prodigy/data/newsgroup_all.jsonl"
        model = Prodigy_model_cpu(pt_model,vectorizer_path,None,label_size=1,optimizer=optimizer,loss=criterion)
        # model = Prodigy_svm_cpu(pt_model,label_size=1,optimizer=optimizer,loss=criterion)

    stream = get_stream(source,input_key = 'text')
    if patterns is None:
        predict = model.predict
        update = model.update
    
    stream = test_stream(stream,predict)

    def updateDB(answers):
        model.update(answers)
        
    def on_exit():
        print("on_exit")
        return model
    
    return {
        'view_id': 'classification',
        'dataset': dataset,
        'stream': stream,
        'exclude': exclude,
        'update': updateDB,
        'on_exit': on_exit,
        'config': {'labels': ['POSITIVE','NEGATIVE'],'batch_size':32}
    }



