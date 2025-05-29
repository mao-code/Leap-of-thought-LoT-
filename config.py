from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv
load_dotenv()

class Settings(BaseSettings):
    # Put your Together-AI key in .env or export TOGETHER_API_KEY=...
    together_api_key: str = Field(..., env="TOGETHER_API_KEY")
    model_name: str = "deepseek-ai/DeepSeek-R1"
    temperature: float = 0.6
    max_tokens: int = 2048
    log_dir: str = "logs"

settings = Settings()  # singleton