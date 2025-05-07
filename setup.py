#!/usr/bin/env python3
"""
Setup script to install required dependencies for the RAG chatbot
"""
import subprocess
import sys
import os

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)

def install_dependencies():
    """Install required packages from requirements.txt"""
    print("Installing required packages...")
    requirements_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])
        print("✅ Packages installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install packages")
        sys.exit(1)

def check_openai_key():
    """Check if OpenAI API key is set"""
    if not os.environ.get("OPENAI_API_KEY"):
        # Check for .env file
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        if os.path.exists(env_path):
            print("✅ .env file found, it should contain your OpenAI API key")
        else:
            # Create .env file from example
            example_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env.example")
            if os.path.exists(example_path):
                print("⚠️  OpenAI API key not found in environment variables")
                print("Creating .env file from example...")
                with open(example_path, "r") as example, open(env_path, "w") as env_file:
                    env_file.write(example.read())
                print("✅ .env file created. Please edit it to add your OpenAI API key")
            else:
                print("❌ OpenAI API key not found and .env.example not found")
                print("Please set the OPENAI_API_KEY environment variable")

def main():
    """Main setup function"""
    print("Setting up RAG Chatbot environment...")
    check_python_version()
    install_dependencies()
    check_openai_key()
    
    print("\nSetup complete! You can now run the chatbot with:")
    print("- Command line: python rag_chatbot.py")
    print("- Web interface: streamlit run streamlit_app.py")

if __name__ == "__main__":
    main()