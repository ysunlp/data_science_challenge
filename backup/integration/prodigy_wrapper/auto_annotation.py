import prodigy
from textcat_recipe import textcat_teach
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
parser.add_argument('--recipe-name',type=str,default='textcat_teach',help='recipe name')
parser.add_argument('--recipe-path',type=str,default='./textcat_recipe.py',help='recipe name')
parser.add_argument('--file-path',type=str,default='/liveperson/data/alloy/prodigy/data/newsgroup_train.jsonl',help='source data')
parser.add_argument('--dataset',type=str,default='newgroup_log',help='output dataset')
parser.add_argument('--input',type=str,default= 'log',help='input model')
parser.add_argument('--label',type=str,default= 'POSITIVE',help='label')
parser.add_argument('--init-path',type=str,default='/liveperson/data/alloy/prodigy/data/newsgroup_example.jsonl',help='file to init model')
parser.add_argument('--vectorizer-path',type=str,default='/liveperson/data/alloy/prodigy/data/newsgroup_all.jsonl',help='file to build vocabulary')
parser.add_argument('--track_dataset',type=str,default=None,help='collection name to track the score')
args = parser.parse_args()

recipe_name = args.recipe_name
recipe_path = args.recipe_path
dataset = args.dataset
file_path = args.file_path
input_model = args.input
label = args.input
init_path = args.init_path
vectorizer_path = args.vectorizer_path
track_dataset = args.track_dataset

print("Using "+input_model+" model to do annotation")
#get recipes
recipe = get_recipe(recipe_name,recipe_path)
#controller = recipe(dataset,file_path,label)
controller = recipe(dataset,input_model,file_path,label,False,[dataset],init_path,vectorizer_path,track_dataset,1)
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
            # two rounds of annotation should use the same vectorizer. confirm it!
            # after confirmation, CountVectorizer generate same vectorizer when the inputs are the same.
            controller = recipe(dataset,model,file_path,label,False,[dataset],None,vectorizer_path,track_dataset,1)
            print("begin a new round")
    else:
        process += len(tasks)
        auto_annotation(tasks,index)
        index += 1
