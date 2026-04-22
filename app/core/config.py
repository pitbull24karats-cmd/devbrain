from pydantic_settings import BaseSettings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    app_name: str = "DevBrain"
    port: int = 8003

    ollama_base_url: str = "http://192.168.243.196:11434"
    embed_model: str = "nomic-embed-text"
    llm_model: str = "qwen2.5:7b"

    data_dir: Path = BASE_DIR / "data"
    projects_dir: Path = BASE_DIR / "data" / "projects"
    global_dir: Path = BASE_DIR / "data" / "global"
    embeddings_dir: Path = BASE_DIR / "data" / "embeddings"
    input_drop_dir: Path = BASE_DIR / "data" / "input_drop"
    prompts_dir: Path = BASE_DIR / "config" / "prompts"

    db_path: Path = BASE_DIR / "data" / "devbrain.db"

    chunk_min_tokens: int = 300
    chunk_max_tokens: int = 800

    hybrid_vector_weight: float = 0.7
    hybrid_keyword_weight: float = 0.3

    class Config:
        env_file = ".env"


settings = Settings()

for _d in [
    settings.projects_dir,
    settings.global_dir / "patterns",
    settings.global_dir / "reusable",
    settings.global_dir / "concepts",
    settings.embeddings_dir,
    settings.input_drop_dir,
]:
    _d.mkdir(parents=True, exist_ok=True)
