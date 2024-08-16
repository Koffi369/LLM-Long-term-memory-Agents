


#uses base model and cpu
import chromadb.utils.embedding_functions as embedding_functions
ef = embedding_functions.InstructorEmbeddingFunction() 

###################################################################


import chromadb.utils.embedding_functions as embedding_functions
ef = embedding_functions.InstructorEmbeddingFunction(
model_name="hkunlp/instructor-xl", device="cuda")

###################################################################

import time
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
# This will download the model to your machine and set it up for GPU support
ef = SentenceTransformerEmbeddingFunction(model_name="thenlper/gte-small", device="cuda")

# Test with 10k documents
docs = []
for i in range(10000):
    docs.append(f"this is a document with id {i}")

start_time = time.perf_counter()
embeddings = ef(docs)
end_time = time.perf_counter()
print(f"Elapsed time: {end_time - start_time} seconds")


################################################################## GPT




import torch
import os
import numpy as np
from transformers import AutoModel, AutoTokenizer
from chromadb import EmbeddingFunction
from gpt4all import Embed4All
import chromadb.utils.embedding_functions as embedding_functions

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
    def __init__(self, model='/app/all-MiniLM-L6-v2'):
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


class ChromaDBEmbedder(BaseEmbedder):
    def __init__(self, model_name="hkunlp/instructor-xl", device="cuda"):
        self.ef = embedding_functions.InstructorEmbeddingFunction(model_name=model_name, device=device)

    def get_embeddings(self, texts):
        if type(texts) == str:
            texts = [texts]

        embeddings = []
        for text in texts:
            embeddings.append(self.ef(text))

        return embeddings

    def __call__(self, text):
        return self.get_embeddings(text)

# Example usage
if __name__ == "__main__":
    # Instantiate the desired embedder
    embedder = ChromaDBEmbedder()

    # Example texts
    texts = ["Example text 1", "Example text 2"]

    # Obtain embeddings
    embeddings = embedder(texts)

    # Example print
    for text, embedding in zip(texts, embeddings):
        print(f"Text: {text}, Embedding: {embedding}")



################################################################### Monica


import torch
from transformers import AutoModel, AutoTokenizer
from chromadb import EmbeddingFunction
import chromadb.utils.embedding_functions as embedding_functions

class BaseEmbedder(EmbeddingFunction):
    def __init__(self):
        pass

    def get_embeddings(self, texts):
        raise NotImplementedError("Subclasses should implement this!")

    def __call__(self, text):
        return self.get_embeddings(text)

# Existing Embedder Classes (GPT4AllEmbedder, HFEmbedder) remain unchanged

# New Instructor Embedder Class
class InstructorEmbedder(BaseEmbedder):
    def __init__(self):
        self.model_name = "hkunlp/instructor-xl"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embedder = embedding_functions.InstructorEmbeddingFunction(model_name=self.model_name, device=self.device)

    def get_embeddings(self, texts):
        if type(texts) == str:
            texts = [texts]
        
        embeddings = self.embedder(texts)
        return embeddings

    def __call__(self, text):
        return self.get_embeddings(text)





##################################################################


import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU is not available, using CPU instead")



###################################################################



import torch
from typing import Optional, Any
from gpt4all import GPT4All
from llama import Llama

class BaseLLM:
    def __init__(self, model_name: Optional[str] = None, model_path: Optional[str] = None) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def generate(self, request: str, streaming: bool) -> Any:
        raise NotImplementedError("Subclasses should implement this!")

class GPT4AllLLM(BaseLLM):
    def __init__(self, model_name: Optional[str] = None, model_path: Optional[str] = None) -> None:
        super().__init__(model_name, model_path)
        self.gpt = GPT4All(model_name=model_name, model_path=model_path, verbose=False)
        self.gpt.to(self.device)

    def generate(self, request: str, streaming: bool) -> Any:
        print(request)
        return self.gpt.generate(prompt=request, streaming=streaming)

class LlamaCPPLLM(BaseLLM):
    def __init__(self, model_name: Optional[str] = None) -> None:
        super().__init__(model_name)
        self.gpt = Llama(model_path=model_name, n_ctx=2048, verbose=False)

    def generate(self, request: str, streaming: bool) -> Any:
        print(request)
        return self.gpt.create_completion(prompt=request, stream=streaming, stop=[f"{self.user}:"])

#########

import os
from hfembedder import HFEmbedder  # Assuming HFEmbedder is defined in 'hfembedder.py'
from summarizer import Summarizer  # Assuming Summarizer is defined in 'summarizer.py'
from collection_operator import CollectionOperator  # Assuming CollectionOperator is defined in 'collection_operator.py'
from llm_agent import LLMAgent  # Assuming LLMAgent is defined in 'llm_agent.py'

# Set the model path
os.environ['LLM_PATH'] = 'mistral-7b-instruct-v0.1.Q4_0.gguf'

port_lib_name = "LLAMA_CPP"
# port_lib_name = "none"

if port_lib_name == "LLAMA_CPP":
    LLM = LlamaCPPLLM
else:
    LLM = GPT4AllLLM

embedder = HFEmbedder()

summarizer = Summarizer()

# LLM model
base_llm = LLM(os.environ.get('LLM_PATH'))

# Create Collections in the DB
Memories_collection = CollectionOperator("Memories", embedder=embedder)
Behaviours_collection = CollectionOperator("Behaviours", embedder=embedder)
Laws_collection = CollectionOperator("Laws", embedder=embedder)

# Create llm Agent on each collection
Memory_llm_agent = LLMAgent(base_llm, Memories_collection, summarizer, use_summarizer=True)
Behaviour_llm_agent = LLMAgent(base_llm, Behaviours_collection, summarizer, use_summarizer=True)
Laws_llm_agent = LLMAgent(base_llm, Laws_collection, summarizer, use_summarizer=True)


