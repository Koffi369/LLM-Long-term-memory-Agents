import os
import argparse
from Memory import LLMAgent
from Memory import LlamaCPPLLM, GPT4AllLLM
from Memory import GPT4AllEmbedder
from Memory import Summarizer
from Memory import CollectionOperator



# Set the model path
os.environ['LLM_PATH'] = 'nous-hermes-llama2-13b.Q4_0.gguf'

port_lib_name = "LLAMA_CPP"
# port_lib_name = "none"

if port_lib_name == "LLAMA_CPP":
    LLM = LlamaCPPLLM

else:
    LLM = GPT4AllLLM


llm = LLM(os.environ.get('LLM_PATH'))

embedder = GPT4AllEmbedder()

summarizer = Summarizer()

total_memory_co = CollectionOperator("total-memory", embedder = embedder)

llm_agent = LLMAgent(llm, total_memory_co, summarizer, use_summarizer = True)




def add_info(info_to_add):
    info_to_add = str(info_to_add)
    llm_agent.add_info_to_mem(info_to_add)

def retrieve_info(user_text_request):
    bot_text_response = llm_agent.memory_response_retriver(user_text_request)  
    # print(bot_text_response)
    bot_text_response = dict(bot_text_response)
    
    return bot_text_response['choices'][0]['text']
















# ### Adding info
# info_to_add = " The child is going to school "
# llm_agent.add_info_to_mem(info_to_add)


# ### Retriving Info

# user_text_request = " where is the banana"
# bot_text_response = llm_agent.memory_response_retriver(user_text_request)

# bot_text_response['choices'][0]['text']


