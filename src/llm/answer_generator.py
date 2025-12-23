from typing import List, Dict, Any

from .llm_client import LLMClient
from .prompt_templates import build_answer_prompt


class AnswerGenerator:
    def __init__(self, llm_client: LLMClient, max_tokens: int) -> None:
        self.llm = llm_client
        self.max_tokens = max_tokens

    def generate_answer(
        self,
        question: str,
        retrieved_local: List[Dict[str, Any]],
        retrieved_global: List[Dict[str, Any]],
    ) -> str:
        prompt = build_answer_prompt(question, retrieved_local, retrieved_global)
        answer = self.llm.generate(prompt, max_tokens=self.max_tokens)
        return answer
