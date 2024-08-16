######################################################################### new llm

from typing import List, Optional, Any
from gpt4all import GPT4All
from llama_cpp import Llama

class BaseLLM():
    def __init__(self, model_name: Optional[str] = None, model_path: Optional[str] = None) -> None:

        self.user = """
            I will provide you with MEMORY CHUNKs retrieved from a database. Your goal is to use these MEMORY CHUNKs to solve a task or answer a question. 
            Rely solely on the information from the MEMORY CHUNKs for solving the task or responding to the query.
            """

       self.assistant = """
        Please adhere to these essential rules when formulating your responses:
        
        Rule 1: If the MEMORY CHUNKs do not contain relevant information to solve the task or answer the question, respond explicitly with "None".
        Rule 2: Do not use any information beyond what is provided in the MEMORY CHUNKs.
        Rule 3: In cases where the MEMORY CHUNKs' information is irrelevant to the task, respond explicitly with "None".
        Rule 4: Provide a direct answer to the question without any additional commentary.
        """
        self.input = "Utilize the following MEMORY CHUNKs explicitly:"
        self.streaming = False

        ##############
        self.memory_context = lambda question: f""" 
        Here is the task  
        Task: {question} 
        """
        ##############

    def generate(self, request: str, streaming: bool) -> Any:
        raise NotImplementedError

    def response(self, request: str) -> Any:
        user = "Instruction" 
        self.assistant = " Response" 
        return self.generate(f"{user}:\n{request}\n{assistant}:\n", streaming = self.streaming)

    ############################################################################################################################
    # def memory_response(self, request: str, memory_queries: List[str]) -> Any:
    #     queries = f"{self.user}:\n{self.memory_context(request)}\n{self.input}:\n"


    #     for i, query in enumerate(memory_queries):
    #         queries += f"MEMORY CHUNK {i}: {query}\n"

    #     queries += f"{self.assistant}:\n"

    #     return self.generate(queries, streaming = self.streaming)

    def memory_response(self, request: str, memory_queries: List[str]) -> Any:
        queries = f"{self.user}\n{self.assistant}\n{self.memory_context(request)}\n{self.input}:\n"


        for i, query in enumerate(memory_queries):
            queries += f"MEMORY CHUNK {i}: {query}\n"

        queries = queries.replace('\n', '')

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



##############################################################################################################################################################################################################



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
            response  = "None"    
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






