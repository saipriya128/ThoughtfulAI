# AI Chatbot with RAG System

This project implements a Retrieval Augmented Generation (RAG) chatbot that can answer questions based on a custom knowledge base in JSON format.

## Features

- Uses LangChain framework for the conversational retrieval chain
- FAISS vector database for efficient similarity search
- Supports JSON data in question-answer format
- Includes both CLI and Streamlit web interfaces
- Conversation memory to maintain context
- Save and load vector database for faster startup

## Requirements

- Python 3.8+
- OpenAI API key

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd rag-chatbot
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key:
   
   Create a `.env` file in the project root with:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

## Usage

### Command Line Interface

Run the chatbot from the command line:

```
python rag_chatbot.py
```

You'll be prompted to provide the path to your JSON file. The default is `data.json`.

### Web Interface

Launch the Streamlit web application:

```
streamlit run streamlit_app.py
```

This will open a browser window with the chatbot interface. You'll need to:
1. Enter your OpenAI API key
2. Start asking questions!

## Data Format

The chatbot expects a JSON file with the following structure:

```json
{
  "questions": [
    {
      "question": "What is example question 1?",
      "answer": "This is the answer to example question 1."
    },
    {
      "question": "What is example question 2?",
      "answer": "This is the answer to example question 2."
    }
  ]
}
```

## Saving and Loading

The chatbot can save the FAISS index to disk for faster initialization in the future. Use:

```python
chatbot.save_vectorstore("your_path")
chatbot.load_vectorstore("your_path")
```

## Performance Considerations

- For large documents, consider increasing chunk size and overlap
- Choose the appropriate LLM model based on your needs (gpt-3.5-turbo is faster, gpt-4 is more accurate)
- When deploying in production, implement proper rate limiting and error handling
