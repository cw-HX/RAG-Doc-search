import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

class Config:
    # This reads your 'gsk_' key from the .env file
    API_KEY = os.getenv("XAI_API_KEY") 
    
    # Model Configuration - UPDATED MODEL NAME
    # Groq recommended llama-3.3-70b-versatile as the replacement
    LLM_MODEL = "llama-3.3-70b-versatile"
    
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    DEFAULT_URLS = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/"
    ]
    
    @classmethod
    def get_llm(cls):
        # Redirecting to Groq infrastructure
        return ChatOpenAI(
            model=cls.LLM_MODEL,
            api_key=cls.API_KEY,
            base_url="https://api.groq.com/openai/v1"
        )