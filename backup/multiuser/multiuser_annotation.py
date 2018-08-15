import prodigy
from multiprocessing import Process
from time import sleep
import atexit
from pathlib import Path

from prodigy.components import printers
from prodigy.components.loaders import get_stream
from prodigy.core import recipe, recipe_args
from prodigy.util import TASK_HASH_ATTR, log
from datetime import datetime
from pymongo import MongoClient
from bson.json_util import dumps
from bson.objectid import ObjectId
from random import shuffle
from prodigy.components.loaders import JSONL
from prodigy.components.db import connect
# Config:
# - add list of coders
# - ?? add port per coder?
# - base file name for files
# - recipe, db, model, output

class DBStream:
    def __init__(self, active_coder, collection_name = "multiuser"):
        self.coll = setup_mongo(collection_name)
        self.active_coder = active_coder
        print("Total tasks in collection: ", self.coll.count())

    def get_examples_v1(self,num_coder = 10):
        print("preassign ",num_coder," questions")
        examples = self.coll.find({"$and" : [
            {"reserved" : 0},
            {"annotated" : 0}]}).limit(num_coder)
        examples = list(examples)
        id_list = [i["_id"] for i in examples]
        self.coll.update_many({"_id": {"$in":id_list}}, 
                                {"$set": {"reserved": 1}})
        #print("After preassign: ",list(self.coll.find({"_id": {"$in":id_list}})))
        # for i in examples:
        #     i['_id'] = str(i['_id'])
        # shuffle(examples)
        self.examples = iter(examples)

    def get_examples(self, max_ann = 3):
        #print("preassign ",num_coder," questions")
        # examples = self.coll.find({"$and" : [
        #     {"reserved" : 0},
        #     {"annotated" : 0}]}).limit(num_coder)
        ### should here need limit number?
        
        examples = self.coll.find({"$and" : [
            {"seen" : {"$lt": max_ann}},
            {"coders" :  {"$nin" : [self.active_coder]}}]})
        examples = list(examples)
        shuffle(examples)
        print("inside get_exampless",self.active_coder,len(examples),examples[0]["_id"])
        # for i in examples:
        #    i['_id'] = str(i['_id'])
        
        self.examples = iter(examples)

    def check_status(self, eg, max_ann = 3):
        examples = self.coll.find_one({"_id":eg["_id"]})
        print("seen",int(examples["seen"]))
        return int(examples["seen"])

def setup_mongo(collection_name, db_name = "prodigy"):
    mongo_uri = "mongodb://gbonev:e9aGdgaDdn67DoP@svpr-anl05:27017/admin?authSource=admin"
    client = MongoClient(mongo_uri)
    db = client[db_name]
    coll = db[collection_name]
    return coll

@prodigy.recipe('mark_custom',
        dataset=recipe_args['dataset'],
        file_path = ("Path to texts","positional",None, str),
        source = recipe_args['source'],
        api=recipe_args['api'],
        loader=recipe_args['loader'],
        label=recipe_args['label'],
        view_id=recipe_args['view'],
        memorize=recipe_args['memorize'],
        exclude=recipe_args['exclude'])
def mark_custom(dataset, file_path,source=None, view_id=None, label='', api=None,
         loader=None, memorize=False, exclude=None):
    """
    Click through pre-prepared examples, with no model in the loop.
    """

    log('RECIPE: Starting recipe mark', locals())
    coder = source # repurposing input slot
    stream_empty = iter([])
    print("coder name: ",coder)
    stream = DBStream(coder, "multiuser")
    #stream = JSONL(file_path)
    stream.get_examples()
    # print("Initial number of examples in queue:", len(list(stream.examples)))
    # print("Initial examples in queue:", list(stream.examples))

    #def ask_questions(stream_empty):
    def ask_questions():
        max_ann = 2
        for eg in stream.examples:
            #stream.get_examples()
            if(stream.check_status(eg) < max_ann):
                eg['time_loaded'] = datetime.now().isoformat()
                # not serializeable
                eg['_id'] = str(eg['_id'])
                print(eg)
                yield eg


    #### Problem with the post-answer update.
    ## Not refreshing

    def recv_answers(answers):
        for eg in answers:
            print("Answer back: ")#, eg)
            # Get the example from the DB again in case it's changed
            updated_ex = list(stream.coll.find({'_id': ObjectId(eg['_id'])}))
            curr_cod = updated_ex[0]['coders']
            # add current coder to the list
            curr_cod.append(coder)
            stream.coll.update_one({"_id": ObjectId(eg['_id'])}, # convert back
                                {"$set": {"coders": curr_cod,
                                         "seen" : len(curr_cod)}})
            eg['time_returned'] = datetime.now().isoformat()
            eg['active_coder'] = coder
            eg['coders'] = curr_cod
            eg['seen'] = len(curr_cod)
            print("update example",eg)
        #print("Refreshing stream...")
        #stream.get_examples()

    def print_results(ctrl):
        print(printers.answers(counts))

    def get_progress(session=0, total=0, loss=0):
        progress = 0#len(counts) / len(stream)
        return progress

    return {
        'view_id': view_id,
        'dataset': dataset,
        'stream': ask_questions(),
        'exclude': exclude,
        'update': recv_answers,
        'on_exit': print_results,
        'config': {'batch_size': 10}
    }

class MultiProdigy:
    def __init__(self,
        coder_list = [#{"name" : "Daniel", "port" : 9010},
                       {"name" : "Youseff", "port" : 7100},
                       #{"name" : "Emad", "port" : 9012},
                       #{"name" : "Rafeef", "port" : 9013},
                       #{"name" : "Mahmoud", "port" : 9014},
                       #{"name" : "Zach", "port" : 9015},
                       {"name" : "Collin", "port" : 9016},
                      ]):
        self.coder_list = coder_list
        self.processes = []

    def serve(self, coder, port):
        print(coder)
        #base = "data/protest_for_classification_"
        #filename = "{0}{1}.jsonl".format(base, coder)
        prodigy.serve('mark_custom',       # recipe
                      "new_test",  # db
                      "/liveperson/data/alloy/prodigy/data/grillbot_textcat_data3.jsonl", #path
                      coder, # input file, repurposed for coder
                      "classification", # view ID
                      "PROTEST",
                      None, # api
                      None, # loaders
                      True, # memorize
                      "new_test", # exclude
                      port=port)  # port

    def make_prodigies(self):
        for coder_info in enumerate(self.coder_list):
            coder_info = coder_info[1] # wut
            thread = Process(target=self.serve, args = (coder_info['name'], coder_info['port']))
            self.processes.append(thread)

    def start_prodigies(self):
        print("Starting Prodigy processes...")
        for p in self.processes:
            p.start()
            sleep(1)

    def kill_prodigies(self):
        print("Killing Prodigy threads")
        for i in self.processes:
            try:
                i.terminate()
            except AttributeError:
                print("Process {0} doesn't exist?".format(i))
        self.processes = []


if __name__ == "__main__":
    mp = MultiProdigy()
    print(1)
    atexit.register(mp.kill_prodigies)
    print(2)
    mp.make_prodigies()
    print(3)
    mp.start_prodigies()
    print(4)
    while True:
        sleep(60 * 60)
        print("Restarting Prodigy...")
        mp.kill_prodigies()
        mp.make_prodigies()
        mp.start_prodigies()

    #    if datetime.datetime.now() > mp.retrain_time:
    #        print("Retraining model and scheduling next retraining for tomorrow")
    #        mp.make_retrain_time() # bump to tomorrow
    #        mp.train_and_restart()

