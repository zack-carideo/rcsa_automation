import yaml
from pathlib import Path

from .loader import PromptLoader

_PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"

class PromptRegistry:
    def __init__(self, registry_path: Path = _PROMPTS_DIR / "registry.yaml"):
        registry_path = Path(registry_path)
        print(f"Loading prompt registry from {registry_path}...")
        with open(registry_path, encoding="utf-8") as f:
            self._registry = yaml.safe_load(f)["prompts"]
        print(f"Loaded {len(self._registry)} prompts: {list(self._registry.keys())}")
        self._loader = PromptLoader(prompt_dir=registry_path.parent)
        print(f"PromptRegistry initialized: {self._loader.prompt_dir} ")

    def get(self, prompt_id: str):
        entry = self._registry[prompt_id]
        print(f'Retrieving prompt "{prompt_id}": {entry}')
        return self._loader.load(entry['file'])
