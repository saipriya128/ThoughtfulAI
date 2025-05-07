import streamlit as st
import json
import os
from typing import List
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
import dotenv

# Set page config
st.set_page_config(
    page_title="Healthcare Agent Chatbot",
    page_icon="⚕️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_key_entered" not in st.session_state:
    st.session_state.api_key_entered = False
if "chatbot_initialized" not in st.session_state:
    st.session_state.chatbot_initialized = False
if "chatbot" not in st.session_state:
    st.session_state.chatbot = None

class RAGChatbot:
    def __init__(self, model_name="gpt-4", temperature=0):
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.embeddings = OpenAIEmbeddings()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True,
            output_key='answer'
        )
        self.vectorstore = None
        self.conversation_chain = None
    
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
            st.error(f"Error loading JSON data: {e}")
            return []
    
    def create_vectorstore(self, documents: List[Document], chunk_size=1000, chunk_overlap=200):
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            split_docs = text_splitter.split_documents(documents)
            self.vectorstore = FAISS.from_documents(split_docs, self.embeddings)
            return True
        except Exception as e:
            st.error(f"Error creating vectorstore: {e}")
            return False
    
    def build_retrieval_chain(self):
        if self.vectorstore is None:
            raise ValueError("Vectorstore must be created first")
        
        custom_prompt = PromptTemplate(
            template="""You are a helpful AI assistant that specializes in healthcare agents.
            Provide accurate, concise answers based on the context below.
            
            Context: {context}
            
            Chat History: {chat_history}
            
            Question: {question}
            
            Answer in a professional tone:""",
            input_variables=["context", "chat_history", "question"]
        )
        
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": custom_prompt},
            return_source_documents=True,
            output_key="answer"
        )
    
    def initialize(self, json_file_path: str = "data.json"):
        try:
            documents = self.load_json_data(json_file_path)
            if not documents:
                st.error(f"No valid documents in {json_file_path}")
                return False
                
            if not self.create_vectorstore(documents):
                return False
                
            self.build_retrieval_chain()
            return True
        except Exception as e:
            st.error(f"Error initializing chatbot: {e}")
            return False
    
    def ask(self, question: str) -> str:
        if not self.conversation_chain:
            return "I'm not initialized yet."
        
        lower_question = question.lower().strip()
        
        greetings = {
            'hello': "Hello! I'm your healthcare agent assistant. How can I help you today?",
            'hi': "Hi there! Ask me anything about healthcare agents.",
            'hey': "Hey! Ready to answer your healthcare agent questions."
        }
        
        if lower_question in greetings:
            return greetings[lower_question]
        if 'who are you' in lower_question or 'what are you' in lower_question:
            return "I'm an AI assistant specialized in healthcare agents like EVA and CAM."
        if 'thank' in lower_question:
            return "You're welcome! Let me know if you have more questions."
        if lower_question in ['ok', 'okay', 'got it']:
            return "Understood. What else would you like to know?"
            
        try:
            response = self.conversation_chain.invoke({"question": question})
            return response["answer"]
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"

# Main App
def main():
    st.title("⚕️ Healthcare Agent Chatbot")
    st.caption("Ask me about eligibility verification (EVA) and claims processing (CAM) agents")

    # API Key Entry Section
    if not st.session_state.api_key_entered:
        with st.form("api_key_form"):
            st.subheader("Enter Your OpenAI API Key")
            api_key = st.text_input(
                "API Key", 
                type="password",
                help="Get your API key from platform.openai.com"
            )
            submitted = st.form_submit_button("Submit")
            
            if submitted:
                if api_key:
                    os.environ["OPENAI_API_KEY"] = api_key
                    st.session_state.api_key_entered = True
                    st.rerun()
                else:
                    st.warning("Please enter a valid API key")

    # Chat Interface
    if st.session_state.api_key_entered:
        # Initialize chatbot if not already initialized
        if not st.session_state.chatbot_initialized:
            with st.spinner("Initializing healthcare knowledge base..."):
                st.session_state.chatbot = RAGChatbot(model_name="gpt-4", temperature=0.7)
                if st.session_state.chatbot.initialize():
                    st.session_state.chatbot_initialized = True
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": "Hello! I'm your healthcare agent assistant. Ask me about EVA, CAM, or other healthcare agents!"
                    })
                else:
                    st.error("Failed to initialize chatbot. Please check your API key and try again.")
                    st.session_state.api_key_entered = False
                    st.rerun()

        # Display chat messages
        for message in st.session_state.messages:
            avatar = "⚕️" if message["role"] == "assistant" else None
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask about healthcare agents..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get assistant response
            with st.chat_message("assistant", avatar="⚕️"):
                with st.spinner("Analyzing healthcare knowledge..."):
                    response = st.session_state.chatbot.ask(prompt)
                st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

        # Add a button to reset the chat
        if st.button("Reset Chat"):
            st.session_state.messages = []
            st.session_state.chatbot.memory.clear()
            st.rerun()

if __name__ == "__main__":
    main()