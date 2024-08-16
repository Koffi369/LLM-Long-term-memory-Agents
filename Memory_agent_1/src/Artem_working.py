from flask import Flask, request


#########################################################################################

################################################################### Utils
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



################################################################ Embeddings

import torch
import os
import numpy as np
from transformers import AutoModel, AutoTokenizer
from chromadb import EmbeddingFunction
from gpt4all import Embed4All
# from dotenv import dotenv_values

# env = dotenv_values(".env")
# os.environ['HUGGINGFACE_HUB_CACHE'] = env['HUGGINGFACE_HUB_CACHE']



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




################################################################### VectDB
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




######################################################################### new llm

from typing import List, Optional, Any
from gpt4all import GPT4All
from llama_cpp import Llama

class BaseLLM():
    def __init__(self, model_name: Optional[str] = None, model_path: Optional[str] = None) -> None:

        self.user = '''We have a database that serves as the memory of a robot. This memory is used by the robot to provide relevant and highly accurate 
        information to solve tasks or answer questions. Your goal is to analyze the information in the memory and assist the robot in providing the most 
        relevant information to solve the task.'''

        self.assistant = '''
        Here are the rules you should follow explicitly: 
        Rules: 
        1 - If there is no relevant information in the memory to answer the question, you must answer with "No relevant information."
        2 - You are not allowed to use any information outside of the provided memory to answer the question.
        
        Remember: There might be cases where the information from the memory is not relevant to solve the task. In such cases, you should explicitly answer 
        with "No relevant information."
        
        Your Response:'''
        self.input = "Here are the pieces of information from the memory:"
        self.streaming = False

        ##############
        self.memory_context = lambda question: f""" 
        Here is the task  \n 
        Task: {question} \n
        """
        ##############

    def generate(self, request: str, streaming: bool) -> Any:
        raise NotImplementedError

    def response(self, request: str) -> Any:
        return self.generate(f"{self.user}:\n{request}\n{self.assistant}:\n", streaming = self.streaming)

    ############################################################################################################################
    def memory_response(self, request: str, memory_queries: List[str]) -> Any:
        queries = f"{self.user}:\n{self.memory_context(request)}\n{self.input}:\n"


        for i, query in enumerate(memory_queries):
            queries += f"MEMORY CHUNK {i}: {query}\n"

        queries += f"{self.assistant}:\n"

        return self.generate(queries, streaming = self.streaming)
    ############################################################################################################################

class GPT4AllLLM(BaseLLM):
    def __init__(self, model_name: Optional[str] = None, model_path: Optional[str] = None) -> None:
        super().__init__(model_name, model_path)
        
        self.gpt = GPT4All(model_name = model_name, model_path = model_path, verbose=False)

    def generate(self, request: str, streaming: bool) -> Any:
        return self.gpt.generate(prompt = request, streaming = streaming)

class LlamaCPPLLM(BaseLLM):
    def __init__(self, model_name: Optional[str] = None) -> None:
        super().__init__(model_name)
        
        self.gpt = Llama(model_path = model_name, n_ctx=2048, verbose=False)

    def generate(self, request: str, streaming: bool) -> Any:
        return self.gpt.create_completion(prompt = request, stream = streaming, stop=[f"{self.user}:"])



######################################################################### summarizer


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


    def summarize(self, text: str, min_length = 30, max_length = 100):
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

######################################################################### llm Agent

enable_logging = True

class LLMAgent():
    def __init__(
        self, 
        llm: BaseLLM = None, 
        tm_qdb: CollectionOperator = None, 
        summarizer: Summarizer = None, 
        # search_engine: SearchEngine = None,
        use_summarizer = True,
       
    ) -> None:

        self.llm = llm
        self.tm_qdb = tm_qdb
        self.memory_access_threshold = 1.5
        # self.similarity_threshold = 0.5 # [0; 1]
        self.db_n_results = 3
        self.se_n_results = 3
        self.use_summarizer = use_summarizer
       
        self.summarizer = summarizer
        # self.search_engine = search_engine
       

    @logging(enable_logging, message = "[Adding to memory]")
    def add(self, request):
        
        summary = self.summarize(request) if self.use_summarizer else request

        self.tm_qdb.add(summary) if summary != "" else None

        response = self.llm.response(request)

        return response
        
    ###################################### New Info to Memory without chat
    
    def add_info_to_mem(self, request):
        # summary = self.summarizer(f"{self.llm.user}:\n{request}\n{self.llm.assistant}:\n{''.join(response)}")
        
        summary = self.summarize(request) if self.use_summarizer else request

        self.tm_qdb.add(summary) if summary != "" else None
        print(f" '{summary}' was added to the memory")

        # response = self.llm.response(request)

    
     ###################################### Get response without chat #####

    
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
            response = self.llm.memory_response(request, acceptable_memory_queries)
        else:
            # response = self.llm.response(request) #TODO: add another solution
            response  = "No relevant information."    
        return response
        

    ############################################################################################################################
    @logging(enable_logging, message = "[Querying memory]")
    def memory_response(self, request):
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
            response = self.llm.memory_response(request, acceptable_memory_queries)
        else:
            response = self.llm.response(request) #TODO: add another solution

        return response
    ############################################################################################################################

    @logging(enable_logging, message = "[Summarizing]", color = "green")
    def summarize(self, text, min_length = 30, max_length = 100):
        return self.summarizer(text, min_length, max_length)


    @logging(enable_logging, message = "[Response]")
    def response(self, request):
        return self.llm.response(request)

    
    def generate(self, request: str):
        if request.upper().startswith("MEM"):
            response = self.memory_response(request[len("MEM"):])
        elif request.upper().startswith("REMEM"): #and len(acceptable_memory_queries) == 0
            response = self.add(request[len("REMEM"):])
        # elif request.upper().startswith("WEB"):
        #     response = self.search(request[len("WEB"):])
        else:
            response = self.response(request)
            
        return response







#########################################################################################

import os

###############################################################################
# Set the model path
os.environ['LLM_PATH'] = 'nous-hermes-llama2-13b.Q4_0.gguf'

port_lib_name = "LLAMA_CPP"
# port_lib_name = "none"

if port_lib_name == "LLAMA_CPP":
    LLM = LlamaCPPLLM

else:
    LLM = GPT4AllLLM


llm = LLM(os.environ.get('LLM_PATH'))

embedder = HFEmbedder()

summarizer = Summarizer()

total_memory_co = CollectionOperator("total-memory", embedder = embedder)

llm_agent = LLMAgent(llm, total_memory_co, summarizer, use_summarizer = True)




def add_info(info_to_add):
    info_to_add = str(info_to_add)
    llm_agent.add_info_to_mem(info_to_add)

def retrieve_info(user_text_request):
    bot_text_response = llm_agent.memory_response_retriver(user_text_request)  
    type(bot_text_response)
    return bot_text_response['choices'][0]['text']


#########################################################################################

app = Flask(__name__)

# # Define your functions here
# def add_info(info_to_add):
#     info_to_add = str(info_to_add)
#     llm_agent.add_info_to_mem(info_to_add)

# def retrieve_info(user_text_request):
#     bot_text_response = llm_agent.memory_response_retriever(user_text_request)
#     print(bot_text_response)
#     bot_text_response = dict(bot_text_response)
#     return bot_text_response['choices'][0]['text']

# Define your API endpoints
@app.route('/addinfo', methods=['POST'])
def addinfo():

    data = request.json

    info_to_add = data['info']

    print("info_to_add: " + info_to_add)

    # info_to_add = request.form['info']
    add_info(info_to_add)
    return 'Info added successfully'

@app.route('/retrieveinfo', methods=['POST'])
def retrieveinfo():
    # user_text_request = request.form['text']

    data = request.json

    user_text_request = data['text']

    print("user_text_request: " + user_text_request)

    response = retrieve_info(user_text_request)
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7778)