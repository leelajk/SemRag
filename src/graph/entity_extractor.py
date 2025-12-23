from typing import List, Dict, Any

import spacy


class EntityExtractor:
    """
    Uses spaCy to extract entities from chunks.
    """

    def __init__(self) -> None:
        self.nlp = spacy.load("en_core_web_sm")

    def extract_entities(
        self, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        enriched_chunks = []
        for ch in chunks:
            doc = self.nlp(ch["text"])
            entities = [ent.text for ent in doc.ents]
            enriched = {**ch, "entities": entities}
            enriched_chunks.append(enriched)
        return enriched_chunks
