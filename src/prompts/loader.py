import yaml
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate

class PromptLoader:
    def __init__(self, prompt_dir: str = "prompts"):
        self.prompt_dir = Path(prompt_dir)

    def load(self, prompt_id: str) -> tuple[ChatPromptTemplate, dict]:
        """
        Returns (ChatPromptTemplate, metadata_dict).
        metadata carries version, model, temperature, tags for logging.
        """
        path = self.prompt_dir / f"{prompt_id}"
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        prompt = ChatPromptTemplate.from_messages([
            ("system", cfg["system"]),
            ("human",  cfg["user_template"]),
        ])

        meta = {k: v for k, v in cfg.items() if k not in ("system", "user_template")}
        return prompt, meta