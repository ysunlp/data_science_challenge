# coding: utf8

import spacy
import random
import cytoolz
import tqdm
import json
import numpy as np

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

from prodigy_wrapper import Prodigy_log_cpu,Prodigy_model_cpu
from pytorch_model import FastText
@recipe('textcat_al',
        dataset=recipe_args['dataset'],
        spacy_model=recipe_args['spacy_model'],
        source=recipe_args['source'],
        label=recipe_args['label_set'],
        api=recipe_args['api'],
        loader=recipe_args['loader'],
        patterns=recipe_args['patterns'],
        long_text=("Long text", "flag", "L", bool),
        exclude=recipe_args['exclude'])
def textcat_al(dataset, spacy_model,source=None, label='', api=None, patterns=None,
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
        nlp = spacy.load('en_core_web_lg')

    #pt_model = nn.Linear(100,1)
    #pt_model = LSTMSentiment(embedding_dim = 100, hidden_dim =100, vocab_size=259136, label_size=2, batch_size=3, dropout=0.5)
        pt_model = FastText_test(vocab_size=684831, emb_dim = 300)
        pt_model.embeds.weight.data.copy_(torch.from_numpy(nlp.vocab.vectors.data))
        model = PyTorchWrapper(pt_model)

        textcat = Loss_TextCategorizer(nlp.vocab,model)
        nlp.add_pipe(textcat)
        model = TextClassifier(nlp, label, long_text=long_text)
    stream = get_stream(source,input_key = 'text')
    if patterns is None:
        predict = model
        update = model.update
    else:
        matcher = PatternMatcher(model.nlp, prior_correct=5.,
                                 prior_incorrect=5., label_span=False,
                                 label_task=True)
        matcher = matcher.from_disk(patterns)
        #log("RECIPE: Created PatternMatcher and loaded in patterns", patterns)
        # Combine the textcat model with the PatternMatcher to annotate both
        # match results and predictions, and update both models.
        predict, update = combine_models(model, matcher)
    # Rank the stream. Note this is continuous, as model() is a generator.
    # As we call model.update(), the ranking of examples changes.
    stream = test_stream(stream,predict)

    def updateDB(answers):
        model.update(answers)
        #print("update model")
        #for eg in answers:
        #    print(eg)
        #for score,eg in model(answers):
        #    eg["update_score"] = score
        #    print("new",score)
        #print(answers)
        
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
        'config': {'labels': model.labels,'batch_size':1}
    }

@recipe('batch-train-custom-cumulate',
        dataset=recipe_args['dataset'],
        input_model=recipe_args['spacy_model'],
        output_model=recipe_args['output'],
        lang=recipe_args['lang'],
        factor=recipe_args['factor'],
        dropout=recipe_args['dropout'],
        n_iter=recipe_args['n_iter'],
        batch_size=recipe_args['batch_size'],
        eval_id=recipe_args['eval_id'],
        eval_split=recipe_args['eval_split'],
        long_text=("Long text", "flag", "L", bool),
        silent=recipe_args['silent'],
        gpu_id=("GPU device","option","g",int),
        shuffle=("shuffle flag", "flag", "shuffle", bool))
def batch_train_custom_cumulate(dataset, input_model=None, output_model=None, lang='en',
                factor=1, dropout=0.2, n_iter=1, batch_size=10,
                eval_id=None, eval_split=None, long_text=False, silent=False,shuffle=False,gpu_id = None):
    if(gpu_id == 0 and torch.cuda.is_available()):
        print("Using cuda")
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        cudnn.benchmark = True
    if(n_iter ==1):
        print("one pass mode")
    print("batch_size",batch_size)
    #print(factor,type(factor))
    DB = connect()
    print_ = get_print(silent)
    random.seed(0)
    if input_model is not None:
        nlp = spacy.load(input_model, disable=['ner'])
        print_('\nLoaded model {}'.format(input_model))
        model = TextClassifier(nlp, labels, long_text=long_text,
                               low_data=len(examples) < 1000)
    else:
        print("build your customized model")
        pt_model = FastText(vocab_size=684831, emb_dim = 300).cuda()
        optimizer = torch.optim.Adam(pt_model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        model = Prodigy_model(pt_model,label_size=1,optimizer=optimizer,loss=criterion)
    examples = DB.get_dataset(dataset)
    if eval_id:
        evals = DB.get_dataset(eval_id)
        print_("Loaded {} evaluation examples from '{}'"
               .format(len(evals), eval_id))
    examples = examples[:int(len(examples) * factor)]
    print_(printers.trainconf(dropout, n_iter, batch_size, factor,
                              len(examples)))
    if len(evals) > 0:
        print_(printers.tc_update_header())
    best_acc = {'accuracy': 0}
    best_model = None
    start_time = datetime.now()
    if len(evals) > 0:
        model.init_eval(evals)
    interval = 100
    for fac in np.arange(interval,len(examples)+interval,interval):
        examples_fac = examples[:fac]
        batch_number = examples_fac/batch_size
        for i in range(n_iter):
            if shuffle:
                print("it's shuffling")
                random.shuffle(examples)
            batch_idx = 0
            loss = 0
            for batch in cytoolz.partition_all(batch_size,
                                               tqdm.tqdm(examples, leave=False)):
                batch = list(batch)
                loss += model.update(batch)
                batch_idx += 1
            acc = model.evaluate(evals)     
            print_('Time:[{0} seconds], process: [{1}/{2}], Epoch: [{3}/{4}], step: [{5}/{6}], Loss: {7},Acc:{8}'.format(
               end_time.seconds,fac, len(examples)//interval, i+1, n_iter, batch_idx+1, len(examples_fac)//batch_size, loss/batch_number, acc))
    return acc

@recipe('batch-train-custom',
        dataset=recipe_args['dataset'],
        input_model=recipe_args['spacy_model'],
        output_model=recipe_args['output'],
        lang=recipe_args['lang'],
        factor=recipe_args['factor'],
        dropout=recipe_args['dropout'],
        n_iter=recipe_args['n_iter'],
        batch_size=recipe_args['batch_size'],
        eval_id=recipe_args['eval_id'],
        eval_split=recipe_args['eval_split'],
        long_text=("Long text", "flag", "L", bool),
        silent=recipe_args['silent'],
        shuffle=("shuffle flag", "flag", "shuffle", bool))
def batch_train_custom(dataset, input_model=None, output_model=None, lang='en',
                factor=1, dropout=0.2, n_iter=1, batch_size=10,
                eval_id=None, eval_split=None, long_text=False, silent=False,shuffle=False):
    if(n_iter ==1):
        print("one pass mode")
    print("batch_size",batch_size)
    #print(factor,type(factor))
    DB = connect()
    print_ = get_print(silent)
    random.seed(0)
    if input_model is not None:
        nlp = spacy.load(input_model, disable=['ner'])
        print_('\nLoaded model {}'.format(input_model))
        model = TextClassifier(nlp, labels, long_text=long_text,
                               low_data=len(examples) < 1000)
    else:
        print("build your customized model")
        pt_model = FastText(vocab_size=684831, emb_dim = 300)
        optimizer = torch.optim.Adam(pt_model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        model = Prodigy_model(pt_model,label_size=1,optimizer=optimizer,loss=criterion)
    examples = DB.get_dataset(dataset)
    if eval_id:
        evals = DB.get_dataset(eval_id)
        print_("Loaded {} evaluation examples from '{}'"
               .format(len(evals), eval_id))
    examples = examples[:int(len(examples) * factor)]
    print_(printers.trainconf(dropout, n_iter, batch_size, factor,
                              len(examples)))
    if len(evals) > 0:
        print_(printers.tc_update_header())
    best_acc = {'accuracy': 0}
    best_model = None
    for i in range(n_iter):
        if shuffle:
            random.shuffle(examples)
        batch_idx = 1
        for batch in cytoolz.partition_all(batch_size,
                                           tqdm.tqdm(examples, leave=False)):
            #print(j)
            batch = list(batch)
            loss = model.update(batch)
            if len(evals) > 0 and batch_idx % (4 * batch_size) == 0:
                acc = model.evaluate(evals)     
                #print_(printers.tc_update(batch_idx, loss, acc))
                print('Epoch: [{0}/{1}], Step: [{2}/{3}], Loss: {4}, Validation Acc:{5}'.format( 
                   i+1, n_iter, batch_idx, len(examples)//batch_size, loss, acc))
            batch_idx += 1
    return acc




@recipe('batch-train-one-pass',
        dataset=recipe_args['dataset'],
        input_model=recipe_args['spacy_model'],
        output_model=recipe_args['output'],
        lang=recipe_args['lang'],
        factor=recipe_args['factor'],
        dropout=recipe_args['dropout'],
        n_iter=recipe_args['n_iter'],
        batch_size=recipe_args['batch_size'],
        eval_id=recipe_args['eval_id'],
        eval_split=recipe_args['eval_split'],
        long_text=("Long text", "flag", "L", bool),
        silent=recipe_args['silent'],
        shuffle=("shuffle flag", "flag", "shuffle", bool))
def batch_train_one_pass(dataset, input_model=None, output_model=None, lang='en',
                factor=1, dropout=0.2, n_iter=1, batch_size=10,
                eval_id=None, eval_split=None, long_text=False, silent=False,shuffle=False):
    """
    Batch train a new text classification model from annotations. Prodigy will
    export the best result to the output directory, and include a JSONL file of
    the training and evaluation examples. You can either supply a dataset ID
    containing the evaluation data, or choose to split off a percentage of
    examples for evaluation.
    """
    #log("RECIPE: Starting recipe textcat.batch-train", locals())
    if(n_iter ==1):
        print("one pass mode")
    print("batch_size",batch_size)
    print(factor,type(factor))
    DB = connect()
    print_ = get_print(silent)
    random.seed(0)
    if input_model is not None:
        nlp = spacy.load(input_model, disable=['ner'])
        print_('\nLoaded model {}'.format(input_model))
    else:
        print("build your customized model")
        nlp = spacy.load('en_core_web_lg')
        pt_model = FastText(vocab_size=684831, emb_dim = 300)
        pt_model.embeds.weight.data.copy_(torch.from_numpy(nlp.vocab.vectors.data))
        model = PyTorchWrapper(pt_model)
        #textcat = TextCategorizer(nlp.vocab,model)
        textcat = Loss_TextCategorizer(nlp.vocab,model)
        nlp.add_pipe(textcat)

        #pt_model = LSTMSentiment(embedding_dim = 100, hidden_dim =100, vocab_size=259136, label_size=2, batch_size=3, dropout=0.5)
        #model = PyTorchWrapper(pt_model)
        #nlp = spacy.load('/home/ysun/pytorchprodigy/')
        #textcat = TextCategorizer(nlp.vocab,model)
        #nlp.add_pipe(textcat)
    examples = DB.get_dataset(dataset)
    labels = {eg['label'] for eg in examples}
    labels = list(sorted(labels))
    print(labels)
    model = TextClassifier(nlp, labels, long_text=long_text,
                           low_data=len(examples) < 1000)
    #log('RECIPE: Initialised TextClassifier with model {}'
    #    .format(input_model), model.nlp.meta)
    if shuffle:    
        print("it's shuffling")
        random.shuffle(examples)
    else:
        print("it's not shuffling")
    if eval_id:
        evals = DB.get_dataset(eval_id)
        print_("Loaded {} evaluation examples from '{}'"
               .format(len(evals), eval_id))
    else:
        examples, evals, eval_split = split_evals(examples, eval_split)
        print_("Using {}% of examples ({}) for evaluation"
               .format(round(eval_split * 100), len(evals)))
    if shuffle:
        random.shuffle(examples)
    examples = examples[:int(len(examples) * factor)]
    print_(printers.trainconf(dropout, n_iter, batch_size, factor,
                              len(examples)))
    if len(evals) > 0:
        print_(printers.tc_update_header())
    best_acc = {'accuracy': 0}
    best_model = None
    if long_text:
        examples = list(split_sentences(nlp, examples, min_length=False))
    for i in range(n_iter):
        if shuffle:
            random.shuffle(examples)
        for batch in cytoolz.partition_all(batch_size,
                                           tqdm.tqdm(examples, leave=False)):
            batch = list(batch)
            loss = model.update(batch, revise=False, drop=dropout)
            if len(evals) > 0:
                print("optimizer averages",model.optimizer.averages)
                #with nlp.use_params():
                with nlp.use_params(model.optimizer.averages):
                    acc = model.evaluate(tqdm.tqdm(evals, leave=False))
                    #if acc['accuracy'] > best_acc['accuracy']:
                        #best_acc = dict(acc)
                        #best_model = nlp.to_bytes()
                print_(printers.tc_update(i, loss, acc))
    # if len(evals) > 0:
    #     print_(printers.tc_result(best_acc))
    # if output_model is not None:
    #     if best_model is not None:
    #         nlp = nlp.from_bytes(best_model)
    #     msg = export_model_data(output_model, nlp, examples, evals)
    #     print_(msg)
    return best_acc['accuracy']

@recipe('textcat_al.batch-train',
        dataset=recipe_args['dataset'],
        input_model=recipe_args['spacy_model'],
        output_model=recipe_args['output'],
        lang=recipe_args['lang'],
        factor=recipe_args['factor'],
        dropout=recipe_args['dropout'],
        n_iter=recipe_args['n_iter'],
        batch_size=recipe_args['batch_size'],
        eval_id=recipe_args['eval_id'],
        eval_split=recipe_args['eval_split'],
        long_text=("Long text", "flag", "L", bool),
        silent=recipe_args['silent'],
        shuffle=("shuffle flag", "flag", "shuffle", bool))
def batch_train(dataset, input_model=None, output_model=None, lang='en',
                factor=1, dropout=0.2, n_iter=10, batch_size=10,
                eval_id=None, eval_split=None, long_text=False, silent=False,shuffle=False):
    """
    Batch train a new text classification model from annotations. Prodigy will
    export the best result to the output directory, and include a JSONL file of
    the training and evaluation examples. You can either supply a dataset ID
    containing the evaluation data, or choose to split off a percentage of
    examples for evaluation.
    """
    #log("RECIPE: Starting recipe textcat.batch-train", locals())
    print("batch_size",batch_size)
    print(factor,type(factor))
    DB = connect()
    print_ = get_print(silent)
    random.seed(0)
    if input_model is not None:
        nlp = spacy.load(input_model, disable=['ner'])
        print_('\nLoaded model {}'.format(input_model))
    else:
        print("build your customized model")
        nlp = spacy.load('en_core_web_lg')
        pt_model = FastText(vocab_size=684831, emb_dim = 300)
        pt_model.embeds.weight.data.copy_(torch.from_numpy(nlp.vocab.vectors.data))
        model = PyTorchWrapper(pt_model)
        textcat = TextCategorizer(nlp.vocab,model)
        nlp.add_pipe(textcat)

        #pt_model = LSTMSentiment(embedding_dim = 100, hidden_dim =100, vocab_size=259136, label_size=2, batch_size=3, dropout=0.5)
        #model = PyTorchWrapper(pt_model)
        #nlp = spacy.load('/home/ysun/pytorchprodigy/')
        #textcat = TextCategorizer(nlp.vocab,model)
        #nlp.add_pipe(textcat)
    examples = DB.get_dataset(dataset)
    labels = {eg['label'] for eg in examples}
    labels = list(sorted(labels))
    print(labels)
    model = TextClassifier(nlp, labels, long_text=long_text,
                           low_data=len(examples) < 1000)
    #log('RECIPE: Initialised TextClassifier with model {}'
    #    .format(input_model), model.nlp.meta)
    if shuffle:    
        print("it's shuffling")
        random.shuffle(examples)
    else:
        print("it's not shuffling")
    if eval_id:
        evals = DB.get_dataset(eval_id)
        print_("Loaded {} evaluation examples from '{}'"
               .format(len(evals), eval_id))
    else:
        examples, evals, eval_split = split_evals(examples, eval_split)
        print_("Using {}% of examples ({}) for evaluation"
               .format(round(eval_split * 100), len(evals)))
    if shuffle:
        random.shuffle(examples)
    examples = examples[:int(len(examples) * factor)]
    print_(printers.trainconf(dropout, n_iter, batch_size, factor,
                              len(examples)))
    if len(evals) > 0:
        print_(printers.tc_update_header())
    best_acc = {'accuracy': 0}
    best_model = None
    if long_text:
        examples = list(split_sentences(nlp, examples, min_length=False))
    for i in range(n_iter):
        loss = 0.
        random.shuffle(examples)
        for batch in cytoolz.partition_all(batch_size,
                                           tqdm.tqdm(examples, leave=False)):
            batch = list(batch)
            loss += model.update(batch, revise=False, drop=dropout)
        if len(evals) > 0:
            with nlp.use_params(model.optimizer.averages):
                acc = model.evaluate(tqdm.tqdm(evals, leave=False))
                if acc['accuracy'] > best_acc['accuracy']:
                    best_acc = dict(acc)
                    best_model = nlp.to_bytes()
            print_(printers.tc_update(i, loss, acc))
    if len(evals) > 0:
        print_(printers.tc_result(best_acc))
    if output_model is not None:
        if best_model is not None:
            nlp = nlp.from_bytes(best_model)
        msg = export_model_data(output_model, nlp, examples, evals)
        print_(msg)
    return best_acc['accuracy']


@recipe('textcat_al.train-curve',
        dataset=recipe_args['dataset'],
        input_model=recipe_args['spacy_model'],
        dropout=recipe_args['dropout'],
        n_iter=recipe_args['n_iter'],
        batch_size=recipe_args['batch_size'],
        eval_id=recipe_args['eval_id'],
        eval_split=recipe_args['eval_split'],
        n_samples=recipe_args['n_samples'],
        random_flag = ("flag to show whether shuffling data","flag","shuffle",bool))
def train_curve(dataset, random_flag, input_model=None, dropout=0.2, n_iter=5,
                batch_size=10, eval_id=None, eval_split=None, n_samples=4):
    """
    Batch-train models with different portions of the training examples and
    print the accuracy figures and accuracy improvements.
    """
    print("*****************random flag",random_flag)
    #log("RECIPE: Starting recipe textcat.train-curve", locals())
    factors = [(i + 1) / n_samples for i in range(n_samples)]
    prev_acc = 0
    if input_model is not None:
        print("\nStarting with model {}".format(input_model))
    else:
        print("\nStarting with blank model")
    print(printers.trainconf(dropout, n_iter, batch_size, samples=n_samples))
    print(printers.tc_curve_header())
    for factor in factors:
        best_acc = batch_train(dataset, input_model=input_model,
                               factor=factor, dropout=dropout, n_iter=n_iter,
                               batch_size=batch_size, eval_id=eval_id,
                               eval_split=eval_split, silent=True,shuffle=random_flag)
        print(printers.tc_curve(factor, best_acc, prev_acc))
        prev_acc = best_acc


@recipe('batch-train-increment',
        dataset=recipe_args['dataset'],
        input_model=recipe_args['spacy_model'],
        output_model=recipe_args['output'],
        lang=recipe_args['lang'],
        factor=recipe_args['factor'],
        dropout=recipe_args['dropout'],
        n_iter=recipe_args['n_iter'],
        batch_size=recipe_args['batch_size'],
        eval_id=recipe_args['eval_id'],
        eval_split=recipe_args['eval_split'],
        long_text=("Long text", "flag", "L", bool),
        silent=recipe_args['silent'],
        gpu_id=("GPU device","option","g",int),
        shuffle=("shuffle flag", "flag", "shuffle", bool))
def batch_train_cumulative(dataset, input_model=None, output_model=None, lang='en',
                factor=1, dropout=0.2, n_iter=1, batch_size=10,
                eval_id=None, eval_split=None, long_text=False, silent=False,shuffle=False,gpu_id = None):
    """
    Batch train a new text classification model from annotations. Prodigy will
    export the best result to the output directory, and include a JSONL file of
    the training and evaluation examples. You can either supply a dataset ID
    containing the evaluation data, or choose to split off a percentage of
    examples for evaluation.
    """
    #log("RECIPE: Starting recipe textcat.batch-train", locals())
    if(gpu_id):
        spacy.util.use_gpu(gpu_id)
    if(n_iter ==1):
        print("one pass mode")
    print("batch_size",batch_size)
    print(factor,type(factor))
    DB = connect()
    print_ = get_print(silent)
    random.seed(0)
    if input_model is not None:
        nlp = spacy.load(input_model, disable=['ner'])
        print_('\nLoaded model {}'.format(input_model))
    else:
        print("build your customized model")
        nlp = spacy.load('en_core_web_lg')
        pt_model = FastText(vocab_size=684831, emb_dim = 300)
        pt_model.embeds.weight.data.copy_(torch.from_numpy(nlp.vocab.vectors.data))
        model = PyTorchWrapper(pt_model)
        #textcat = TextCategorizer(nlp.vocab,model)
        textcat = Loss_TextCategorizer(nlp.vocab,model)
        nlp.add_pipe(textcat)
    examples = DB.get_dataset(dataset)
    labels = {eg['label'] for eg in examples}
    labels = list(sorted(labels))
    print(labels)
    model = TextClassifier(nlp, labels, long_text=long_text,
                           low_data=len(examples) < 1000)
    if shuffle:    
        print("it's shuffling")
        random.shuffle(examples)
    else:
        print("it's not shuffling")
    if eval_id:
        evals = DB.get_dataset(eval_id)
        print_("Loaded {} evaluation examples from '{}'"
               .format(len(evals), eval_id))
    else:
        examples, evals, eval_split = split_evals(examples, eval_split)
        print_("Using {}% of examples ({}) for evaluation"
               .format(round(eval_split * 100), len(evals)))
    if shuffle:
        random.shuffle(examples)
    examples = examples[:int(len(examples) * factor)]
    print_(printers.trainconf(dropout, n_iter, batch_size, factor,
                              len(examples)))
    if len(evals) > 0:
        print_(printers.tc_update_header())
    # best_acc = {'accuracy': 0}
    # best_model = None
    if long_text:
        examples = list(split_sentences(nlp, examples, min_length=False))
    batch_idx = 0
    start_time = datetime.now()
    for batch in cytoolz.partition_all(batch_size,
                                       tqdm.tqdm(examples, leave=False)):
        batch = list(batch)
        for i in range(n_iter):
            loss = model.update(batch, revise=False, drop=dropout)
            if len(evals) > 0:
                #print("optimizer averages",model.optimizer.averages)
                with nlp.use_params(model.optimizer.averages):
                    acc = model.evaluate(tqdm.tqdm(evals, leave=False))
                #print_(printers.tc_update(i, loss, acc))
                end_time = datetime.now() -start_time
                print('Time:[{0} seconds ]Epoch: [{1}/{2}], batch: [{3}/{4}], Loss: {5},Acc:{6}'.format( 
                   end_time.seconds,i+1, n_iter, batch_idx+1, len(examples)//batch_size, loss, acc))
            batch_idx += 1
    return acc

@recipe('batch-train-increment',
        dataset=recipe_args['dataset'],
        input_model=recipe_args['spacy_model'],
        output_model=recipe_args['output'],
        lang=recipe_args['lang'],
        factor=recipe_args['factor'],
        dropout=recipe_args['dropout'],
        n_iter=recipe_args['n_iter'],
        batch_size=recipe_args['batch_size'],
        eval_id=recipe_args['eval_id'],
        eval_split=recipe_args['eval_split'],
        long_text=("Long text", "flag", "L", bool),
        silent=recipe_args['silent'],
        gpu_id=("GPU device","option","g",int),
        shuffle=("shuffle flag", "flag", "shuffle", bool))
def batch_train_increment(dataset, input_model=None, output_model=None, lang='en',
                factor=1, dropout=0.2, n_iter=1, batch_size=10,
                eval_id=None, eval_split=None, long_text=False, silent=False,shuffle=False,gpu_id = None):
    """
    Batch train a new text classification model from annotations. Prodigy will
    export the best result to the output directory, and include a JSONL file of
    the training and evaluation examples. You can either supply a dataset ID
    containing the evaluation data, or choose to split off a percentage of
    examples for evaluation.
    """
    #log("RECIPE: Starting recipe textcat.batch-train", locals())
    if(gpu_id):
        spacy.util.use_gpu(gpu_id)
    if(n_iter ==1):
        print("one pass mode")
    print("batch_size",batch_size)
    print(factor,type(factor))
    DB = connect()
    print_ = get_print(silent)
    random.seed(0)
    if input_model is not None:
        nlp = spacy.load(input_model, disable=['ner'])
        print_('\nLoaded model {}'.format(input_model))
    else:
        print("build your customized model")
        nlp = spacy.load('en_core_web_lg')
        pt_model = FastText(vocab_size=684831, emb_dim = 300)
        pt_model.embeds.weight.data.copy_(torch.from_numpy(nlp.vocab.vectors.data))
        model = PyTorchWrapper(pt_model)
        #textcat = TextCategorizer(nlp.vocab,model)
        textcat = Loss_TextCategorizer(nlp.vocab,model)
        nlp.add_pipe(textcat)
    examples = DB.get_dataset(dataset)
    labels = {eg['label'] for eg in examples}
    labels = list(sorted(labels))
    print(labels)
    model = TextClassifier(nlp, labels, long_text=long_text,
                           low_data=len(examples) < 1000)
    if shuffle:    
        print("it's shuffling")
        random.shuffle(examples)
    else:
        print("it's not shuffling")
    if eval_id:
        evals = DB.get_dataset(eval_id)
        print_("Loaded {} evaluation examples from '{}'"
               .format(len(evals), eval_id))
    else:
        examples, evals, eval_split = split_evals(examples, eval_split)
        print_("Using {}% of examples ({}) for evaluation"
               .format(round(eval_split * 100), len(evals)))
    if shuffle:
        random.shuffle(examples)
    examples = examples[:int(len(examples) * factor)]
    print_(printers.trainconf(dropout, n_iter, batch_size, factor,
                              len(examples)))
    if len(evals) > 0:
        print_(printers.tc_update_header())
    # best_acc = {'accuracy': 0}
    # best_model = None
    if long_text:
        examples = list(split_sentences(nlp, examples, min_length=False))
    batch_idx = 0
    start_time = datetime.now()
    for batch in cytoolz.partition_all(batch_size,
                                       tqdm.tqdm(examples, leave=False)):
        batch = list(batch)
        for i in range(n_iter):
            loss = model.update(batch, revise=False, drop=dropout)
            if len(evals) > 0:
                #print("optimizer averages",model.optimizer.averages)
                with nlp.use_params(model.optimizer.averages):
                    acc = model.evaluate(tqdm.tqdm(evals, leave=False))
                #print_(printers.tc_update(i, loss, acc))
                end_time = datetime.now() -start_time
                print('Time:[{0} seconds], Epoch: [{1}/{2}], batch: [{3}/{4}], Loss:{5}, Accuracy:{6}'.format( 
                   end_time.seconds,i+1, n_iter, batch_idx+1, len(examples)//batch_size, loss, acc['accuracy']))
            batch_idx += 1
    return acc