import prodigy
from textcat_al import textcat_al
#from prodigy.recipes.textcat import teach
from prodigy.components.db import connect

import spacy
from spacy.matcher import Matcher
import json
from prodigy.core import Controller
from prodigy.core import get_recipe, set_recipe  # noqa
import argparse

# set arguments
parser = argparse.ArgumentParser(description='auto annotation')
parser.add_argument('--type',type=str,default='newgroup_log',help='determine which data')
parser.add_argument('--file-path',type=str,default='/liveperson/data/alloy/prodigy/data/newsgroup_train.jsonl',help='source data')
parser.add_argument('--dataset',type=str,default='newgroup_log',help='output dataset')
args = parser.parse_args()
arg = args.type
if(arg == "default_vector"):
    recipe_name = 'textcat.teach'
    recipe_path = "/liveperson/data/alloy/prodigy/code/textcat_stream.py"
    dataset = 'default_vector_experiment'
    file_path = '/liveperson/data/alloy/prodigy/data/IMDB_train_2000.jsonl'
    output_model = "/home/ysun/Pytorch/Experiment2/default_vector_model"
    input_model = "en_core_web_lg"

if(arg == "fasttext"):
    print("custom vector model, fasttext")
    dataset = args.dataset
    file_path = args.file_path
    recipe_name = 'textcat_custom'
    recipe_path = "/liveperson/data/alloy/prodigy/code/textcat_al.py"
    output_model = None
    input_model = None # build customized model

if(arg == "experiment_svm"):
    print("custom vector model, SVM")
    recipe_name = 'textcat_svm'
    recipe_path = "/liveperson/data/alloy/prodigy/code/textcat_al.py"
    #dataset = 'custom_svm_pro_32'
    #dataset = 'custom_log_pro_32'
    #dataset = 'custom_log_ema_32_uni'
    # dataset = 'custom_svm_ema_32_uni_normal'
    dataset = 'custom_nb_ema_32_uni'
    file_path = '/liveperson/data/alloy/prodigy/data/IMDB_train_2000.jsonl'
    output_model = None
    input_model = None # build customized model

if(arg == "experiment_log"):
    print("custom vector model, LOG")
    recipe_name = 'textcat_log'
    recipe_path = "/liveperson/data/alloy/prodigy/code/textcat_al.py"
    #dataset = 'custom_svm_pro_32'
    #dataset = 'custom_log_pro_32'
    #dataset = 'custom_log_ema_32_uni'
    dataset = 'custom_log_ema_32_uni_sig'
    file_path = '/liveperson/data/alloy/prodigy/data/IMDB_train_2000.jsonl'
    output_model = None
    input_model = None # build customized model
label = ['POSITIVE','NEGATIVE']
if(arg == "order_log"):
    print("custom vector model, LOG")
    recipe_name = 'textcat_log'
    recipe_path = "/liveperson/data/alloy/prodigy/code/textcat_al.py"
    dataset = "order_log"
    file_path = "/liveperson/data/alloy/prodigy/data/order_status_train.jsonl"
    output_model = None
    input_model = None 
    label = ["ORDER_STATUS"]
if(arg == "newgroup_log"):
    print("custom vector model, LOG")
    recipe_name = 'textcat_log'
    recipe_path = "/liveperson/data/alloy/prodigy/code/textcat_al.py"
    dataset = args.dataset
    file_path = args.file_path
    output_model = None
    input_model = None 
    label = ["windows"]

#get recipes
recipe = get_recipe(recipe_name,recipe_path)
#controller = recipe(dataset,file_path,label)
controller = recipe(dataset,input_model,file_path,label,None,None,None,None,[dataset])
auto_annotation = []
def auto_annotation(tasks,index):
    auto_annotation = []
    print('get_questions',len(tasks))
    #nonlocal auto_annotation  # want to update this outside of the function
    for eg in tasks:
        if('answer' in eg.keys()):
            pass
        else:
            if eg['meta']['label_answer'] == eg['label']:
            # has to have appropriate label and not be a match in order to reject
                eg['answer'] = 'accept'  # auto-reject
                auto_annotation.append(eg)
            else:
                eg['answer'] = 'reject'  # auto-reject
        eg['iter'] = index
        auto_annotation.append(eg)
        #print("answer",eg["answer"])
    controller.receive_answers(auto_annotation)
    print('receive_answers',len(auto_annotation))
    #auto_annotation = []
#text_set = dict()
process = 0
last_process = 0
index = 0
while True:
    #print('begin get questions')
    #import ipdb;ipdb.set_trace()
    tasks = controller.get_questions()
    if(process%512 == 0):
        print(process,"data have been annotated")
    if(len(tasks) == 0):
        print('No task is available')
        print("Totally,",process,"data have been annotated")
        print("final iteration index is ", index)
        break
    elif(None in tasks):
        tasks =[t for t in tasks if t != None]
        if(len(tasks)):
            process += len(tasks)
            auto_annotation(tasks,index)
            print('No task of this pass is available',process-last_process,"has been annotated in this pass")
            last_process = process
            model = controller.on_exit()
            if(output_model):
                model.nlp.to_disk(output_model)
                print("export model to ", output_model)
                controller = recipe(dataset,output_model,file_path,label,None,None,None,None,[dataset])
            else:
                controller = recipe(dataset,model,file_path,label,None,None,None,None,[dataset])
            print("begin a new round")
    else:
        process += len(tasks)
        auto_annotation(tasks,index)
        index += 1
