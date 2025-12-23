from typing import List, Dict, Any


def build_answer_prompt(
    question: str,
    local_context: List[Dict[str, Any]],
    global_context: List[Dict[str, Any]],
) -> str:
    local_texts = [c["text"] for c in local_context]
    global_texts = [c["summary"] if "summary" in c else c["text"] for c in global_context]

    prompt = (
        "You are an assistant answering questions about Dr. B.R. Ambedkar's works.\n"
        "You must strictly answer based only on the provided context.\n"
        "If the answer is not contained in the context, say you do not know.\n\n"
        "=== LOCAL EVIDENCE ===\n"
    )

    for i, txt in enumerate(local_texts):
        prompt += f"[LOCAL_{i}]\n{txt}\n\n"

    prompt += "=== GLOBAL CONTEXT ===\n"
    for i, txt in enumerate(global_texts):
        prompt += f"[GLOBAL_{i}]\n{txt}\n\n"

    prompt += (
        f"=== QUESTION ===\n{question}\n\n"
        "=== INSTRUCTIONS ===\n"
        "- Answer concisely in a few paragraphs.\n"
        "- Do NOT introduce facts not present in the context.\n"
        "- Mention relevant entities explicitly where helpful.\n"
    )

    return prompt
