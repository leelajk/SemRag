from typing import List


def merge_with_buffer(sentences: List[str], buffer_size: int) -> List[str]:
    """
    Simple buffer merge: for each sentence, include a small window
    of neighbors to preserve local context.
    """
    if buffer_size <= 0:
        return sentences

    merged = []
    n = len(sentences)

    for i in range(n):
        left = max(0, i - buffer_size)
        right = min(n, i + buffer_size + 1)
        merged_sent = " ".join(sentences[left:right])
        merged.append(merged_sent)

    return merged
