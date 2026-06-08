from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    # OpenRouter
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "google/gemini-2.0-flash-exp:free"

    # Groq ← YE ADD KARO
    groq_api_key: str = ""

    # App
    app_name: str = "DocuMind AI"
    app_version: str = "2.0.0"
    debug: bool = False

    # File Upload
    upload_dir: str = "uploads"
    max_file_size_mb: int = 50

    # Vector Store
    vectorstore_dir: str = "vectorstore"
    embedding_model: str = "BAAI/bge-base-en-v1.5"

    # RAG — improved values vs v1
    chunk_size: int = 900        # was 1000
    chunk_overlap: int = 200      # was 200
    retrieval_k: int = 8        # was 6  — fetch more, reranker trims
    reranker_top_n: int = 4       # reranker keeps best N

    # LLM — default 1024 avoids OpenRouter 402 credit errors on low-balance accounts
    max_tokens: int = 1024
    temperature: float = 0.1
    streaming: bool = True

    # CORS
    allowed_origins: list = ["http://localhost:3000", "http://localhost:8501", "*"]

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
