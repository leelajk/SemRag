import subprocess
import json
from typing import Optional


class LLMClient:
    """
    Thin wrapper around ollama.

    Assumes you have `ollama` installed and a model pulled, e.g.:
    ollama pull mistral
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> str:
        cmd = [
            "ollama",
            "run",
            self.model_name,
            "--json",
        ]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        request = json.dumps({"prompt": prompt, "num_predict": max_tokens})
        stdout, stderr = proc.communicate(request)

        if stderr:
            # For interview-demo: you can log this instead of raising
            print("LLM error:", stderr)

        # Ollama streams JSON lines; we keep the last non-empty one
        text_out = ""
        for line in stdout.splitlines():
            try:
                obj = json.loads(line)
                text_out += obj.get("response", "")
            except json.JSONDecodeError:
                continue
        return text_out.strip()
