import json
import os
import warnings
from typing import List, Dict, Any

# Suppress LangChain deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# LangChain imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate

# Check for tiktoken
try:
    import tiktoken
except ImportError:
    print("Warning: tiktoken not found. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tiktoken"])

# Environment setup
import dotenv
dotenv.load_dotenv()

class RAGChatbot:
    def __init__(self, model_name="gpt-4", temperature=0):
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set.")
            
        try:
            self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
            self.embeddings = OpenAIEmbeddings()
            self.memory = ConversationBufferMemory(
                memory_key="chat_history", 
                return_messages=True,
                output_key='answer'  # Explicit output key for memory
            )
            self.vectorstore = None
            self.conversation_chain = None
        except Exception as e:
            raise Exception(f"Error initializing language models: {e}")
        
    def load_json_data(self, json_file_path: str) -> List[Document]:
        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
            
            documents = []
            for item in data.get("questions", []):
                question = item.get("question", "")
                answer = item.get("answer", "")
                
                question_doc = f"Question: {question}\nAnswer: {answer}"
                documents.append(Document(
                    page_content=question_doc, 
                    metadata={"source": json_file_path, "type": "qa_pair"}
                ))
                
                answer_doc = f"Information: {answer}\nThis information answers the question: {question}"
                documents.append(Document(
                    page_content=answer_doc, 
                    metadata={"source": json_file_path, "type": "answer"}
                ))
            
            return documents
        except Exception as e:
            print(f"Error loading JSON data: {e}")
            return []
    
    def create_vectorstore(self, documents: List[Document], chunk_size=1000, chunk_overlap=200):
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            split_docs = text_splitter.split_documents(documents)
            self.vectorstore = FAISS.from_documents(split_docs, self.embeddings)
            print(f"Created vectorstore with {len(split_docs)} document chunks")
        except Exception as e:
            print(f"Error creating vectorstore: {e}")
    
    def build_retrieval_chain(self):
        if self.vectorstore is None:
            raise ValueError("Vectorstore must be created first")
        
        custom_prompt = PromptTemplate(
            template="""You are a helpful AI assistant that answers questions based on the provided knowledge base.
            
            Context: {context}
            
            Chat History: {chat_history}
            
            Question: {question}
            
            Answer:""",
            input_variables=["context", "chat_history", "question"]
        )
        
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": custom_prompt},
            return_source_documents=True,
            output_key="answer"  # This matches the memory's output_key
        )
    
    def initialize_from_json(self, json_file_path: str = "data.json"):
        try:
            documents = self.load_json_data(json_file_path)
            if not documents:
                print(f"No valid documents in {json_file_path}")
                return False
                
            self.create_vectorstore(documents)
            if not self.vectorstore:
                print("Failed to create vectorstore")
                return False
                
            self.build_retrieval_chain()
            print("Chatbot initialized successfully!")
            return True
        except Exception as e:
            print(f"Error initializing from JSON: {e}")
            return False
    
    def ask(self, question: str) -> str:
        if not self.conversation_chain:
            return "I'm not initialized yet. Please load data first."
        
        lower_question = question.lower().strip()
        
        if lower_question in ['hello', 'hi', 'hey']:
            return "Hello! How can I help you today?"
        if lower_question in ['who are you', 'what are you']:
            return "I'm an AI assistant that answers questions based on the provided knowledge base."
        if lower_question in ['thanks', 'thank you']:
            return "You're welcome!"
        if lower_question in ['ok', 'okay']:
            return "Let me know if you have any questions."
            
        try:
            response = self.conversation_chain.invoke({"question": question})
            return response["answer"]
        except Exception as e:
            return f"I encountered an error: {str(e)}"

def main():
    import time
    
    print("\n" + "="*50)
    print("AI Knowledge Base Chatbot")
    print("="*50)
    
    if not os.getenv("OPENAI_API_KEY"):
        api_key = input("\nEnter your OpenAI API key: ")
        if api_key.strip():
            os.environ["OPENAI_API_KEY"] = api_key
        else:
            print("API key required")
            return
    
    print("\nInitializing assistant...")
    chatbot = RAGChatbot(model_name="gpt-4", temperature=0.7)
    
    # Loading animation
    print("\nLoading knowledge base...")
    for i in range(5):
        print(".", end="", flush=True)
        time.sleep(0.5)
    print()
    
    if not chatbot.initialize_from_json():
        return
    
    print("\nReady to assist! Type 'quit' to exit.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            break
            
        print("Thinking...", end="\r")
        response = chatbot.ask(user_input)
        print(f"Assistant: {response}")
    
    print("\nGoodbye!")

if __name__ == "__main__":
    main()