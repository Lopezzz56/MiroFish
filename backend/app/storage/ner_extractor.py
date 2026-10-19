"""
NER/RE Extractor — entity and relation extraction via Groq

Uses a dedicated Groq model for graph extraction. Embeddings remain in the
local embedding service and are not routed through this client.
"""

import json
import logging
import math
import re
import time
from typing import Dict, Any, Optional

from openai import RateLimitError

from ..config import Config
from ..utils.llm_client import LLMClient, LLMOutputTruncatedError

logger = logging.getLogger('mirofish.ner_extractor')

# Keep each extraction response below Groq's 1,000 output-token free-tier limit.
_MAX_EXTRACTION_OUTPUT_TOKENS = 800
_MAX_TRUNCATION_SPLIT_DEPTH = 2

# System prompt template for NER/RE extraction
_SYSTEM_PROMPT = """Extract all explicitly stated or strongly implied entities and relations using this ontology:
{ontology_description}
Use only listed types; if no types are supplied, infer suitable types. Canonicalize names.
Entities need name, type, and optional attributes.
Relations need source, target, type, and a concise fact sentence. If none, use empty arrays.
Return only JSON: {{"entities":[{{"name":"","type":"","attributes":{{}}}}],"relations":[{{"source":"","target":"","type":"","fact":""}}]}}"""

_USER_PROMPT = "Extract entities and relations from this text:\n{text}"


class NERExtractor:
    """Extract entities and relations from text using the configured Groq model."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        max_retries: int = 1,
    ):
        self.llm = llm_client
        self.max_retries = max(0, min(max_retries, 1))

    def _get_client(self) -> LLMClient:
        if self.llm is not None:
            return self.llm
        if not Config.GROQ_API_KEY:
            raise ValueError(
                "Graph extraction requires GROQ_API_KEY; local Ollama is reserved "
                "for embeddings."
            )
        if not Config.GROQ_GRAPH_MODEL:
            raise ValueError(
                "Set GROQ_GRAPH_MODEL to an account-available Groq model before "
                "building a graph."
            )
        if not Config.GROQ_GRAPH_FREE_TIER_CONFIRMED:
            raise ValueError(
                "Graph extraction is disabled until you verify the model is covered "
                "by your Groq free-tier account, then set "
                "GROQ_GRAPH_FREE_TIER_CONFIRMED=true."
            )
        timeout = Config.GROQ_GRAPH_TIMEOUT_SECONDS
        if not math.isfinite(timeout):
            raise ValueError("GROQ_GRAPH_TIMEOUT_SECONDS must be a finite number")
        timeout = max(1.0, min(timeout, 300.0))
        client = LLMClient(
            api_key=Config.GROQ_API_KEY,
            base_url=Config.GROQ_BASE_URL,
            model=Config.GROQ_GRAPH_MODEL,
            timeout=timeout,
        )
        self.llm = client
        return client

    @staticmethod
    def _rate_limit_retry_delay(error: RateLimitError) -> Optional[float]:
        retry_after = error.response.headers.get("retry-after")
        if retry_after is not None:
            try:
                delay = float(retry_after)
                if 0 < delay <= 60:
                    return delay
            except ValueError:
                pass

        message = str(error.body or error)
        match = re.search(r"try again in\s+(\d+(?:\.\d+)?)\s*s", message, re.IGNORECASE)
        if not match:
            return None
        delay = float(match.group(1))
        return delay if 0 < delay <= 60 else None

    def extract(self, text: str, ontology: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract entities and relations from text, guided by ontology.

        Args:
            text: Input text chunk
            ontology: Dict with 'entity_types' and 'relation_types' from graph

        Returns:
            Dict with 'entities' and 'relations' lists:
            {
                "entities": [{"name": str, "type": str, "attributes": dict}],
                "relations": [{"source": str, "target": str, "type": str, "fact": str}]
            }
        """
        if not text or not text.strip():
            return {"entities": [], "relations": []}

        ontology_desc = self._format_ontology(ontology)
        system_msg = _SYSTEM_PROMPT.format(ontology_description=ontology_desc)
        client = self._get_client()
        return self._extract_with_bounded_split(
            text.strip(), ontology, system_msg, client, split_depth=0
        )

    def _extract_with_bounded_split(
        self,
        text: str,
        ontology: Dict[str, Any],
        system_msg: str,
        client: LLMClient,
        split_depth: int,
    ) -> Dict[str, Any]:
        """Subdivide only when the provider confirms its JSON output was truncated."""
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": _USER_PROMPT.format(text=text)},
        ]

        for attempt in range(self.max_retries + 1):
            started = time.monotonic()
            try:
                result = client.chat_json(
                    messages=messages,
                    temperature=0.1,  # Low temp for extraction precision
                    max_tokens=_MAX_EXTRACTION_OUTPUT_TOKENS,
                )
                cleaned = self._validate_and_clean(result, ontology)
                logger.info(
                    "Graph extraction completed with %s in %.1fs: %s entities, "
                    "%s relations",
                    Config.GROQ_GRAPH_MODEL,
                    time.monotonic() - started,
                    len(cleaned["entities"]),
                    len(cleaned["relations"]),
                )
                return cleaned
            except LLMOutputTruncatedError as error:
                if split_depth >= _MAX_TRUNCATION_SPLIT_DEPTH:
                    raise RuntimeError(
                        "Graph extraction output remained truncated after bounded "
                        "subdivision; reduce graph chunk size and retry."
                    ) from error
                left_text, right_text = self._split_with_overlap(text)
                logger.warning(
                    "Graph extraction output truncated; splitting %d characters "
                    "into overlapping subchunks (depth %d)",
                    len(text),
                    split_depth + 1,
                )
                left = self._extract_with_bounded_split(
                    left_text, ontology, system_msg, client, split_depth + 1
                )
                right = self._extract_with_bounded_split(
                    right_text, ontology, system_msg, client, split_depth + 1
                )
                return self._merge_extractions(left, right)
            except RateLimitError as error:
                delay = self._rate_limit_retry_delay(error)
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        "Groq rate limit reached; the bounded retry limit was exhausted."
                    ) from error
                if delay is None:
                    raise RuntimeError(
                        "Groq rate limit reached; retry stopped because the server's "
                        "retry delay was missing or exceeded the 60-second safety cap."
                    ) from error
                logger.warning("Groq rate limit reached; retrying once after %.1fs", delay)
                time.sleep(delay)

    @staticmethod
    def _split_with_overlap(text: str) -> tuple[str, str]:
        midpoint = len(text) // 2
        window = max(1, len(text) // 10)
        boundary = text.rfind(" ", max(1, midpoint - window), midpoint + 1)
        if boundary < 0:
            boundary = text.find(" ", midpoint, min(len(text), midpoint + window))
        if boundary < 0:
            boundary = midpoint
        else:
            boundary += 1

        overlap = min(120, max(1, len(text) // 10))
        return text[:boundary + overlap], text[max(0, boundary - overlap):]

    @staticmethod
    def _merge_extractions(
        first: Dict[str, Any], second: Dict[str, Any]
    ) -> Dict[str, Any]:
        entities_by_name = {
            entity["name"].casefold(): {
                **entity,
                "attributes": dict(entity["attributes"]),
            }
            for entity in first["entities"]
        }
        for entity in second["entities"]:
            key = entity["name"].casefold()
            existing = entities_by_name.get(key)
            if existing is None:
                entities_by_name[key] = entity
            else:
                existing["attributes"].update(entity["attributes"])
                if existing["type"] == "Entity" and entity["type"] != "Entity":
                    existing["type"] = entity["type"]

        relations_by_key = {}
        for relation in first["relations"] + second["relations"]:
            key = tuple(
                relation[field].casefold()
                for field in ("source", "target", "type", "fact")
            )
            relations_by_key.setdefault(key, relation)

        return {
            "entities": list(entities_by_name.values()),
            "relations": list(relations_by_key.values()),
        }

    def _format_ontology(self, ontology: Dict[str, Any]) -> str:
        """Serialize ontology semantics compactly without dropping descriptions."""
        entities = []
        for entity_type in ontology.get("entity_types", []):
            if isinstance(entity_type, dict):
                entities.append({
                    key: entity_type[key]
                    for key in ("name", "description", "attributes")
                    if key in entity_type
                })
            else:
                entities.append(entity_type)

        relations = []
        for relation_type in ontology.get("relation_types", ontology.get("edge_types", [])):
            if isinstance(relation_type, dict):
                relations.append({
                    key: relation_type[key]
                    for key in ("name", "description", "source_targets")
                    if key in relation_type
                })
            else:
                relations.append(relation_type)

        if not entities and not relations:
            return "No specific ontology defined; infer suitable entity and relation types."

        return json.dumps(
            {"entity_types": entities, "relation_types": relations},
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def _validate_and_clean(
        self, result: Dict[str, Any], ontology: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate and normalize LLM output."""
        if not isinstance(result, dict):
            raise ValueError("Graph extraction response must be a JSON object")
        entities = result.get("entities", [])
        relations = result.get("relations", [])
        if not isinstance(entities, list) or not isinstance(relations, list):
            raise ValueError("Graph extraction entities and relations must be lists")

        # Get valid type names from ontology
        valid_entity_types = set()
        for et in ontology.get("entity_types", []):
            if isinstance(et, dict):
                valid_entity_types.add(et.get("name", "").strip())
            else:
                valid_entity_types.add(str(et).strip())

        valid_relation_types = set()
        for rt in ontology.get("relation_types", ontology.get("edge_types", [])):
            if isinstance(rt, dict):
                valid_relation_types.add(rt.get("name", "").strip())
            else:
                valid_relation_types.add(str(rt).strip())

        # Clean entities
        cleaned_entities = []
        seen_names = set()
        for entity in entities:
            if not isinstance(entity, dict):
                raise ValueError("Every extracted entity must be a JSON object")
            if not isinstance(entity.get("name"), str) or not isinstance(entity.get("type"), str):
                raise ValueError("Every entity must include string name and type fields")
            name = entity["name"].strip()
            etype = entity["type"].strip()
            if not name:
                raise ValueError("Extracted entity names must not be empty")
            attributes = entity.get("attributes", {})
            if not isinstance(attributes, dict):
                raise ValueError(f"Attributes for entity '{name}' must be a JSON object")

            # Deduplicate by normalized name
            name_lower = name.lower()
            if name_lower in seen_names:
                continue
            seen_names.add(name_lower)

            # If ontology has types, warn but keep entities with unknown types
            if valid_entity_types and etype not in valid_entity_types:
                logger.debug(f"Entity '{name}' has type '{etype}' not in ontology, keeping anyway")

            cleaned_entities.append({
                "name": name,
                "type": etype,
                "attributes": attributes,
            })

        # Clean relations
        cleaned_relations = []
        entity_names_lower = {e["name"].lower() for e in cleaned_entities}
        seen_relations = set()
        for relation in relations:
            if not isinstance(relation, dict):
                raise ValueError("Every extracted relation must be a JSON object")
            fields = ("source", "target", "type", "fact")
            if any(not isinstance(relation.get(field), str) for field in fields):
                raise ValueError(
                    "Every relation must include string source, target, type, and fact fields"
                )
            source, target, rtype, fact = (
                relation[field].strip() for field in fields
            )
            if not source or not target or not rtype:
                raise ValueError("Relation source, target, and type must not be empty")

            # Ensure source and target entities exist
            # (they might not if LLM hallucinated a relation without the entity)
            if source.lower() not in entity_names_lower:
                cleaned_entities.append({
                    "name": source,
                    "type": "Entity",
                    "attributes": {},
                })
                entity_names_lower.add(source.lower())

            if target.lower() not in entity_names_lower:
                cleaned_entities.append({
                    "name": target,
                    "type": "Entity",
                    "attributes": {},
                })
                entity_names_lower.add(target.lower())

            normalized_fact = fact or f"{source} {rtype} {target}"
            relation_key = (
                source.casefold(),
                target.casefold(),
                rtype.casefold(),
                normalized_fact.casefold(),
            )
            if relation_key in seen_relations:
                continue
            seen_relations.add(relation_key)
            cleaned_relations.append({
                "source": source,
                "target": target,
                "type": rtype,
                "fact": normalized_fact,
            })

        return {
            "entities": cleaned_entities,
            "relations": cleaned_relations,
        }
