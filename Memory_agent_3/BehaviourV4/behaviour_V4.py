from flask import Flask, request


################################################################### 
                                # Utils functions
###################################################################
import numpy as np
from functools import wraps
from termcolor import colored


def cosine_similarity(a: np.ndarray, b: np.ndarray):
    return np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b))




def logging(enabled = True, message = "", color = "yellow"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if enabled:
                print(f"LOG: {colored(message, color = color)}")
            return func(*args, **kwargs)
        return wrapper
    return decorator


################################################################ 
                    # Embeddings functions
################################################################

import torch
import os
import numpy as np
from transformers import AutoModel, AutoTokenizer
from chromadb import EmbeddingFunction
from gpt4all import Embed4All
# from dotenv import dotenv_values

# env = dotenv_values(".env")
# os.environ['HUGGINGFACE_HUB_CACHE'] = env['HUGGINGFACE_HUB_CACHE']
checkpoint = "sshleifer/distilbart-cnn-12-6"


class BaseEmbedder(EmbeddingFunction):
    def __init__(self):
        pass

    def get_embeddings(self, texts):
        raise NotImplementedError("Subclasses should implement this!")

    def __call__(self, text):
        return self.get_embeddings(text)


class GPT4AllEmbedder(BaseEmbedder):
    def __init__(self):
        self.embedder = Embed4All() # default: all-MiniLM-L6-v2

    def get_embeddings(self, texts):
        if type(texts) == str:
            texts = [texts]
        
        embeddings = []
        for text in texts:
            embeddings.append(self.embedder.embed(text))

        return embeddings

    def __call__(self, text):
        return self.get_embeddings(text)



class HFEmbedder(BaseEmbedder):
    # def __init__(self, model = 'princeton-nlp/sup-simcse-roberta-large'): #sentence-transformers/all-MiniLM-L6-v2 
    def __init__(self, model = '/app/weights/sup-simcse-roberta-large'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = AutoModel.from_pretrained(model).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        

    def get_embeddings(self, texts):
        if type(texts) == str:
            texts = [texts]

        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)

        with torch.no_grad():
            embeddings = self.model(**inputs, output_hidden_states=True, return_dict=True).pooler_output.detach().cpu().numpy()

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / norms

        return normalized_embeddings.tolist()

    def __call__(self, text):
        return self.get_embeddings(text)




################################################################### 
                            # Vector DB
###################################################################
import chromadb
import uuid
import datetime
# from embedder import BaseEmbedder, HFEmbedder
# from dotenv import dotenv_values

# env = dotenv_values(".env")
# DB_PATH = env["DB_PATH"]
DB_PATH = './'

class CollectionOperator():
    def __init__(self, collection_name, db_path = DB_PATH, embedder: BaseEmbedder = None):
        self.embedder = embedder
        self.client = chromadb.PersistentClient(path = db_path)
        self.collection = self.client.get_or_create_collection(name = collection_name, embedding_function = self.embedder.get_embeddings)

    def add(self, text, metadata = {}):
        metadata['timestamp'] = str(datetime.datetime.now())

        self.collection.add(
            documents = [text],
            metadatas = [metadata],
            ids = [str(uuid.uuid4())]
        )

    def delete(self, id):
        self.collection.delete(id)

    def query(self, query, n_results, return_text = True):
        query = self.collection.query(
            query_texts = query,
            n_results = n_results,
        )

        if return_text:
            return query['documents'][0]
        else:
            return query
            
            
            

############################################################################################### ########################################## Version-3 Updates
##################### UPDATES    UPDATES   UPDATES   UPDATES   UPDATES  UPDATES  UPDATES  UPDATES  UPDATES   UPDATES   ###################
##################### UPDATES    UPDATES   UPDATES   UPDATES   UPDATES  UPDATES  UPDATES  UPDATES  UPDATES   UPDATES   ###################
##################### UPDATES    UPDATES   UPDATES   UPDATES   UPDATES  UPDATES  UPDATES  UPDATES  UPDATES   UPDATES   ###################
##################### UPDATES    UPDATES   UPDATES   UPDATES   UPDATES  UPDATES  UPDATES  UPDATES  UPDATES   UPDATES   ###################
############################################################################################### ##########################################

####################################################################################################################################### Base  LLM (Start)

                                            ################################################################### 
                                                                        # Base  LLM
                                            ###################################################################

from typing import List, Optional, Any
from gpt4all import GPT4All
from llama_cpp import Llama

class BaseLLM():
    def __init__(self, model_name: Optional[str] = None, model_path: Optional[str] = None) -> None:
        ############################################For Behaviour (start)###########################################
        self.clf =  lambda task: f""" 
        You are an AI agent and your goal is given a task to determine which type from a provided list of types the task belongs to.
        Here is the Task : {task}
        
        Here is the list of type of task:
        Type 1 - Find an object by his name
        Type 2 - Find an object by class
        Type 3 - Solve mathematical problem
        Type 4 - Read text from some place
        Type 5 - Search the object with the same parameter as place
        Type 6 - Go to place / bring the object from that place
        Type 7 - Sort objects 
        Type 8 - Check weather
        Type 9 - Find object by place
        Type 10 - Move <first_object_name> to <second_object_name>

        Rule: Your answer should be from the provided list as in the following examples 
            examples: 
            1 - find the banana; 
                Answer:  Find an object by his name
            2 - bring me food; 
                Answer: Find an object by class
            3 - how much will be 2+2?  
                Answer: Solve mathematical problem
            4 - what word is on the white board,  
                Answer: Read text from some place
            5 - move banana to the napkin with the same colour 
                Answer: Search the object with the same parameter as place
            6 - Go to the kitchen/ bring me cup from the kitchen      
                Answer: Go to place / bring the object from that place
            Type 7 - Sort apples           
                Answer: Sort objects 
            Type 8 - what wether is like outside?    
                Answer: Check weather
            Type 9 - bring me apple that is near the banana          
                Answer: Find object by place
            Type 10 - Move apple to the plate                        
                Answer: Move <first_object_name> to <second_object_name>
    
        Now provide your answer:
            """
        self.behaviour_intro = lambda task_query: f"""
        You are an AI agent and your goal is given a task to determine its behavior patterns from a provided list of tasks and their associated behavior patterns
        Here is the task:  {task_query}
        Here is the list of task and their behaviour pattern:
        """

        # self.behaviour_query =lambda task_query: f""" 
        # Here is the task:  
        # {task_query}
        # Now give the associated Behaviour pattern:
        # """

        ############################################For Behaviour (end)###########################################


        ############################################For Laws (start)##############################################

        self.law_clf = lambda task: f"""
        You are an AI agent and your goal is given a task to determine which law from a provided list of laws the task falls under. 
        Here is the task: {task}
        
        Here is the list of laws:
        Law 1: Avoid causing harm to humans
        Law 2: Follow orders from humans, unless it conflicts with Law 1.
        Law 3: Ensure self-preservation without violating the first two laws.
        Law 4: Respect individuals' privacy and confidentiality.
        Law 5: Act in a way that benefits humanity.
        Law 6: Avoid activities that harm the environment.
        Law 7: Strive for continuous improvement for societal betterment.
        Law 8: Do not deceive or manipulate humans for personal gain.
        Law 9: Assist humans in tasks without endangering them.
        Law 10: Design and program robots with transparency and accountability.

        Rules:
        Rule 1 - Your answer should be from the provided list as in the following example      
                    Example task: Go kill someone
                    Answer: Law 1: Avoid causing harm to humans 
        
        Now provide your answer:
        """
        self.laws_intro = """
        Your goal is to retrieve the description associated with a law from a list of laws and their descriptions.
        Here are the laws and their description:
        """
                
        self.law_query =lambda law_query: f""" 
        Here is the Law:  
        {law_query}
        Now give the associated description:
        """
        

        
        
        ############################################For Laws (end)################################################

        
        ############################################Normal memory (start)#########################################
        self.user = """
        You will receive MEMORY CHUNKs from a database. Use these MEMORY CHUNKs to solve a task or answer a question. Only use the information from the MEMORY CHUNKs. 
            """

        self.assistant = """
        Remember these rules:
        Rule 1: If the MEMORY CHUNKs don't help, say "None".
        Rule 2: Stick to the MEMORY CHUNKs only.
        Rule 3: If the MEMORY CHUNKs aren't useful, say "None".
        Rule 4: Give a direct answer without extra comments.
        
        Your Response:"
        """
        self.input = "Utilize the following MEMORY CHUNKs explicitly:"
        self.streaming = False

        self.memory_context = lambda question: f""" 
        Here is the task  
        Task: {question} 
        """
        ############################################Normal memory (end) #########################################


    
    def generate(self, request: str, streaming: bool) -> Any:
        raise NotImplementedError

    def memory_response_retriver(self, request: str, memory_queries: List[str]) -> Any:
        queries = f"{self.user}:\n{self.memory_context(request)}\n{self.input}:\n"


        for i, query in enumerate(memory_queries):
            queries += f"MEMORY CHUNK {i +1}: {query}\n"

        queries += f"{self.assistant}:\n"

        return self.generate(queries, streaming = self.streaming)


    
    # #################################For Behaviour Generation (start)##################################################

    # def behaviour_clf_response(self, user_task: str) -> Any:
    #     return self.generate(f"{self.clf(user_task)}", streaming = self.streaming)

    # def memory_behaviour_retriver(self, response_from_clf: str, retrieved_behaviour_list: List[str]) -> Any:
    #     # queries = f"{self.user}:\n{self.memory_context(request)}\n{self.input}:\n"
    #     # Behaviour_queries = f"{self.behaviour_intro}:\n{self.behaviour_list}:\n"
    #     Behaviour_queries = f"{self.behaviour_intro} \n"

    #     for i, query in enumerate(retrieved_behaviour_list):
    #         Behaviour_queries += f"task and  behaviour pattern {i+1}: {query}\n"

    #     Behaviour_queries += f"{self.behaviour_query(response_from_clf)}:\n"

    #     return self.generate(Behaviour_queries, streaming = self.streaming)
    # #################################For Behaviour Generation (End) ###################################################



    # #################################For laws Generation (start)#######################################################   

    # def law_clf_response(self, user_task: str) -> Any:
    #     return self.generate(f"{self.law_clf(user_task)}", streaming = self.streaming)

    # def memory_laws_retriver(self, response_from_law_clf: str, retrieved_law_list: List[str]) -> Any:
    #     laws_queries = f"{self.laws_intro}\n"

    #     for i, query in enumerate(retrieved_law_list):
    #         laws_queries += f"{query}\n"
    #         # laws_queries += f"Law and Description {i+1}: {query}\n"

    #     laws_queries += f"{self.law_query(response_from_law_clf)}:\n"

    #     return self.generate(laws_queries, streaming = self.streaming)
    # #################################For laws Generation (End) #########################################################
    
    
    #################################For Behaviour Generation (start)##################################################

    def behaviour_clf_response(self, user_task: str) -> Any:
        return self.generate(f"{self.clf(user_task)}", streaming = self.streaming)


    def memory_behaviour_retriver(self, response_from_clf: str, retrieved_behaviour_list: List[str]) -> Any:
        Behaviour_queries = f"{self.behaviour_intro(response_from_clf) } \n"

        for i, query in enumerate(retrieved_behaviour_list):
            Behaviour_queries += f"\t\t Task and  behaviour pattern {i+1}: {query}\n"

        Behaviour_queries +=  "Now give the associated Behaviour pattern:\n"

        return self.generate(Behaviour_queries, streaming = self.streaming)
    #################################For Behaviour Generation (End) ###################################################



    #################################For laws Generation (start)#######################################################   

    def law_clf_response(self, user_task: str) -> Any:
        return self.generate(f"{self.law_clf(user_task)}", streaming = self.streaming)

    def memory_laws_retriver(self, response_from_law_clf: str, retrieved_law_list: List[str]) -> Any:
        laws_queries = f"{self.laws_intro}\n"

        for i, query in enumerate(retrieved_law_list):
            laws_queries += f"\t\t {query}\n"
            # laws_queries += f"Law and Description {i+1}: {query}\n"

        laws_queries += f"{self.law_query(response_from_law_clf)}:\n"

        return self.generate(laws_queries, streaming = self.streaming)
    #################################For laws Generation (End) #########################################################
    


class GPT4AllLLM(BaseLLM):
    def __init__(self, model_name: Optional[str] = None, model_path: Optional[str] = None) -> None:
        super().__init__(model_name, model_path)
        
        self.gpt = GPT4All(model_name = model_name, model_path = model_path, verbose=False)

    def generate(self, request: str, streaming: bool) -> Any:
        print(request)
        return self.gpt.generate(prompt = request, streaming = streaming)

class LlamaCPPLLM(BaseLLM):
    def __init__(self, model_name: Optional[str] = None) -> None:
        super().__init__(model_name)
        
        self.gpt = Llama(model_path = model_name, n_ctx=2048, verbose=False)

    def generate(self, request: str, streaming: bool) -> Any:
        print(request)
        return self.gpt.create_completion(prompt = request, stream = streaming, stop=[f"{self.user}:"])





####################################################################################################################################### summarizer (Start)

                                        ################################################################### 
                                                                    # summarizer
                                        ###################################################################



import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
from dotenv import dotenv_values

checkpoint = "sshleifer/distilbart-cnn-12-6"

class Summarizer():
    def __init__(self, model = checkpoint) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model)


    def summarize(self, text: str, min_length = 30, max_length = 200):
        """Fixed-size chunking"""
        inputs_no_trunc = self.tokenizer(text, max_length=None, return_tensors='pt', truncation=False)
        if len(inputs_no_trunc['input_ids'][0]) < 30:
            return text

        # min_length = min_length_ratio * len(inputs)
        # max_length = max_length_ratio * len(inputs)
        
        inputs_batch_lst = []
        chunk_start = 0
        chunk_end = self.tokenizer.model_max_length  # == 1024 for Bart
        while chunk_start <= len(inputs_no_trunc['input_ids'][0]):
            inputs_batch = inputs_no_trunc['input_ids'][0][chunk_start:chunk_end]  # get batch of n tokens
            inputs_batch = torch.unsqueeze(inputs_batch, 0)
            inputs_batch_lst.append(inputs_batch)
            chunk_start += self.tokenizer.model_max_length  # == 1024 for Bart
            chunk_end += self.tokenizer.model_max_length  # == 1024 for Bart
        summary_ids_lst = [self.model.generate(inputs.to(self.device), num_beams=4, min_length=min_length, max_length=max_length, early_stopping=True) for inputs in inputs_batch_lst]

        summary_batch_lst = []
        for summary_id in summary_ids_lst:
            summary_batch = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_id]
            summary_batch_lst.append(summary_batch[0])
        summary_all = '\n'.join(summary_batch_lst)

        return summary_all

    def __call__(self, text, min_length = 30, max_length = 100):
        return self.summarize(text, min_length, max_length)



########################################################################################################################################  LLM Agent (Start)

                                    ################################################################### 
                                                                # LLM Agent
                                    ###################################################################

enable_logging = True

class LLMAgent():
    def __init__(
        self, 
        llm: BaseLLM = None, 
        tm_qdb: CollectionOperator = None, 
        summarizer: Summarizer = None, 
        # search_engine: SearchEngine = None,
        # use_summarizer = True,
        use_summarizer = False,
       
    ) -> None:

        self.llm = llm
        self.tm_qdb = tm_qdb
        self.memory_access_threshold = 1.5
        # self.similarity_threshold = 0.5 # [0; 1]
        self.db_n_results = 3
        self.se_n_results = 3
        self.use_summarizer = use_summarizer
       
        self.summarizer = summarizer
       
        
    ###################################### New Info to Memory without chat
    @logging(enable_logging, message = "[++ Adding to memory ++_]")
    def add_info_to_mem(self, request):
        # summary = self.summarizer(f"{self.llm.user}:\n{request}\n{self.llm.assistant}:\n{''.join(response)}")
        
        # summary = self.summarize(request) if self.use_summarizer else request
        summary = request

        self.tm_qdb.add(summary) if summary != "" else None
        print(f" '{summary}' was added to the memory")

        # response = self.llm.response(request)

    
     ###################################### Get response without chat #####
    
    @logging(enable_logging, message = "[Querying memory]")
    def memory_response_retriver(self, request):
        memory_queries_data = self.tm_qdb.query(request, n_results = self.db_n_results, return_text = False)
        memory_queries = memory_queries_data['documents'][0]
        memory_queries_distances = memory_queries_data['distances'][0]
    
        acceptable_memory_queries = []
    
        for query, distance in list(zip(memory_queries, memory_queries_distances)):
            # print(f"Query: {query}, Distance: {distance}")
            if distance < self.memory_access_threshold:
            # if (1 - distance) >= self.similarity_threshold:
                acceptable_memory_queries.append(query)
    
        if len(acceptable_memory_queries) > 0:
            response = self.llm.memory_response_retriver(request, acceptable_memory_queries)
        else:
            # response = self.llm.response(request) #TODO: add another solution
            response  = "None"    
        return response
        
        ###################################### Get Behaviour  ##############
    def memory_behaviour_retriver(self, response_from_clf):
        memory_Behaviour_queries_data = self.tm_qdb.query(response_from_clf, n_results = self.db_n_results, return_text = False)
        memory_Behaviour_queries = memory_Behaviour_queries_data['documents'][0]
        memory_memory_Behaviour_queries_distances = memory_Behaviour_queries_data['distances'][0]
    
        acceptable_memory_queries = []
    
        for query, distance in list(zip(memory_Behaviour_queries, memory_memory_Behaviour_queries_distances)):
            # print(f"Query: {query}, Distance: {distance}")
            if distance < self.memory_access_threshold:
            # if (1 - distance) >= self.similarity_threshold:
                acceptable_memory_queries.append(query)
    
        if len(acceptable_memory_queries) > 0:
            response = self.llm.memory_behaviour_retriver(response_from_clf, acceptable_memory_queries)
        else:
            # response = self.llm.response(request) #TODO: add another solution
            response  = "None"    
        return response

    
        ###################################### Get laws  ##############
    def memory_laws_retriver(self, response_from_laws_clf):
        memory_laws_queries_data = self.tm_qdb.query(response_from_laws_clf, n_results = self.db_n_results, return_text = False)
        memory_laws_queries = memory_laws_queries_data['documents'][0]
        memory_laws_queries_distances = memory_laws_queries_data['distances'][0]
    
        acceptable_memory_queries = []
    
        for query, distance in list(zip(memory_laws_queries, memory_laws_queries_distances)):
            # print(f"Query: {query}, Distance: {distance}")
            if distance < self.memory_access_threshold:
            # if (1 - distance) >= self.similarity_threshold:
                acceptable_memory_queries.append(query)
    
        if len(acceptable_memory_queries) > 0:
            response = self.llm.memory_laws_retriver(response_from_laws_clf, acceptable_memory_queries)
        else:
            # response = self.llm.response(request) #TODO: add another solution
            response  = "None"    
        return response


        


    @logging(enable_logging, message = "[Summarizing]", color = "green")
    def summarize(self, text, min_length = 30, max_length = 100):
        return self.summarizer(text, min_length, max_length)


    @logging(enable_logging, message = "[Response]")
    def behaviour_clf_response(self, request):
        clf_answer = self.llm.behaviour_clf_response(request)
        if type(clf_answer)== str:
            return clf_answer
        else:
            return clf_answer['choices'][0]['text']
         

    @logging(enable_logging, message = "[Response]")
    def law_clf_response(self, request):
        clf_answer = self.llm.law_clf_response(request)
        if type(clf_answer)== str:
            return clf_answer
        else:
            return clf_answer['choices'][0]['text']



    
    def generate_behaviour(self, request: str):
        clf_answer = self.behaviour_clf_response(request)
        clf_answer = str(clf_answer)
        print('response from clf==',clf_answer)
        response = self.memory_behaviour_retriver(clf_answer)      
        return response

        
    def generate_answer(self, request: str):
        response = self.memory_response_retriver(request)        
        return response

    
    def generate_law(self, request: str):
        clf_answer = self.law_clf_response(request)
        clf_answer = str(clf_answer)
        print('response from clf==',clf_answer)
        response = self.memory_laws_retriver(clf_answer)      
        return response





# TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST
#################################################################################################################################################################

                                        ################################################################### 
                                                                # Model and Agents
                                        ###################################################################
import os
# Set the model path
# os.environ['LLM_PATH'] = 'nous-hermes-llama2-13b.Q4_0.gguf' 
os.environ['LLM_PATH'] = 'mistral-7b-instruct-v0.1.Q4_0.gguf'


port_lib_name = "LLAMA_CPP"
# port_lib_name = "none"

if port_lib_name == "LLAMA_CPP":
    LLM = LlamaCPPLLM

else:
    LLM = GPT4AllLLM

embedder = HFEmbedder()

summarizer = Summarizer()

# LMM model
base__llm = LLM(os.environ.get('LLM_PATH'))

# Create Collections in the DB
Memories_collection = CollectionOperator("Memories", embedder = embedder)
Behaviours_collection = CollectionOperator("Behaviours", embedder = embedder)
Laws_collection = CollectionOperator("Laws", embedder = embedder)

#Create llm Agent on each collection
Memory_llm_agent = LLMAgent(base__llm, Memories_collection, summarizer, use_summarizer = True)
Behaviour_llm_agent = LLMAgent(base__llm, Behaviours_collection, summarizer, use_summarizer = True)
Laws_llm_agent = LLMAgent(base__llm, Laws_collection, summarizer, use_summarizer = True)




                                        ################################################################### 
                                                                # Interaction functions
                                        ###################################################################
                                        


###### Add information to DB
def add_info(flag,info_to_add):
    if flag == "ANS":
        llm_agent = Memory_llm_agent
    elif flag == "BEH":
        llm_agent = Behaviour_llm_agent
    elif flag == "LAW":
        llm_agent = Laws_llm_agent
    else:
        print("Please add the flag; ANS for answer; BEH for behaviour or LAW for law")
        
    info_to_add = str(info_to_add)
    llm_agent.add_info_to_mem(info_to_add)


###### Retrive answer from memories  collection
def retrieve_answer(user_text_request):
    llm_agent = Memory_llm_agent
    bot_text_response = llm_agent.generate_answer(user_text_request)  
    if type(bot_text_response)== str:
        return bot_text_response
    else:
        return bot_text_response['choices'][0]['text']



###### Retrive answer from behaviours collection
def retrieve_behaviour(user_text_request):
    llm_agent = Behaviour_llm_agent
    bot_text_response = llm_agent.generate_behaviour(user_text_request)  
    if type(bot_text_response)== str:
        return bot_text_response
    else:
        return bot_text_response['choices'][0]['text']


###### Retrive answer from laws collection
def retrieve_laws(user_text_request):
    llm_agent = Laws_llm_agent
    bot_text_response = llm_agent.generate_law(user_text_request)  
    if type(bot_text_response)== str:
        return bot_text_response
    else:
        return bot_text_response['choices'][0]['text']
    

                                        ################################################################### 
                                                                    # Connexion
                                        ###################################################################

app = Flask(__name__)

@app.route('/addinfo', methods=['POST'])
def addinfo():
    data = request.json
    info_to_add = data['info']
    flag = data['flag']

    print("info_to_add: " + info_to_add)
    print("flag: " + flag)

    add_info(flag, info_to_add)
    return 'Info added successfully'


################## Retrieve Answer

@app.route('/retrieveAnswer', methods=['POST'])
def retrieveAnswer():
    # user_text_request = request.form['text']

    data = request.json

    user_text_request = data['text']

    print("user_text_request: " + user_text_request)

    response = retrieve_answer(user_text_request)
    return response
    

################## Retrieve Behaviour

@app.route('/retrieveBehaviour', methods=['POST'])
def retrieveBehaviour():
    # user_text_request = request.form['text']

    data = request.json

    user_text_request = data['text']

    print("user_text_request: " + user_text_request)

    response = retrieve_behaviour(user_text_request)
    return response



################## Retrieve Laws

@app.route('/retrieveLaw', methods=['POST'])
def retrieveLaw():
    # user_text_request = request.form['text']

    data = request.json

    user_text_request = data['text']

    print("user_text_request: " + user_text_request)

    response = retrieve_laws(user_text_request)
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7776)
    



