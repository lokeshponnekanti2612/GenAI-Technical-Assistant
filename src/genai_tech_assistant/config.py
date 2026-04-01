from pydantic import BaseModel
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

Project_Root = Path(__file__).resolve().parents[2]
Data_Path = Project_Root / "data"


class Settings(BaseModel):
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model_name: str = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    pdf_input_dir: Path = Path(
        os.getenv("PDF_INPUT_DIR", str(Data_Path / "raw_findings")))
    vector_store_dir: Path = Path(
        os.getenv("VECTOR_STORE_DIR", str(Data_Path / "Vector_Store")))
    ollama_base_url: str = os.getenv(
        "OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model_name: str = os.getenv("OLLAMA_MODEL_NAME", "llama3.2")


settings = Settings()
