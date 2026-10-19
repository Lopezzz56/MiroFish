import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
from openai import BadRequestError, RateLimitError

from app.config import Config
from app.api.graph import _has_active_graph_task
from app.services.graph_builder import GraphBuilderService
from app.models.task import TaskManager
from app.storage.ner_extractor import NERExtractor
from app.utils.file_parser import split_text_into_chunks
from app.utils.llm_client import LLMClient, LLMOutputTruncatedError


class StubLLM:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.messages = []
        self.kwargs = []

    def chat_json(self, messages, **kwargs):
        self.messages.append(messages)
        self.kwargs.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class GraphExtractionTests(unittest.TestCase):
    def test_chunking_covers_all_source_characters_with_bounded_overlap(self):
        text = "abcdefghijklmnopqrstuvwxyz" * 18
        chunks = split_text_into_chunks(text, chunk_size=100, overlap=20)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(0 < len(chunk) <= 100 for chunk in chunks))
        covered = [False] * len(text)
        start = 0
        for chunk in chunks:
            self.assertEqual(text[start:start + len(chunk)], chunk)
            for index in range(start, start + len(chunk)):
                covered[index] = True
            start += len(chunk) - 20
        self.assertTrue(all(covered))

    def test_chunking_rejects_invalid_overlap(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            split_text_into_chunks("text", chunk_size=10, overlap=10)

    def test_extraction_sends_complete_chunk_and_deduplicates_relations(self):
        text = "Earlier section. " + ("Later section " * 39) + "Later section"
        response = {
            "entities": [
                {"name": "Ada", "type": "Person"},
                {"name": "ada", "type": "Person"},
                {"name": "Analytical Engine", "type": "Machine"},
            ],
            "relations": [
                {
                    "source": "Ada",
                    "target": "Analytical Engine",
                    "type": "DESIGNED",
                    "fact": "Ada designed the Analytical Engine.",
                },
                {
                    "source": "Ada",
                    "target": "Analytical Engine",
                    "type": "DESIGNED",
                    "fact": "Ada designed the Analytical Engine.",
                },
            ],
        }
        llm = StubLLM(response)
        result = NERExtractor(llm_client=llm).extract(
            text, {"entity_types": ["Person", "Machine"], "relation_types": ["DESIGNED"]}
        )

        self.assertTrue(llm.messages[0][1]["content"].endswith(text))
        self.assertEqual(llm.kwargs[0]["max_tokens"], 800)
        self.assertEqual([e["name"] for e in result["entities"]], ["Ada", "Analytical Engine"])
        self.assertEqual(len(result["relations"]), 1)

    def test_truncated_output_splits_chunk_and_merges_all_extractions(self):
        text = "first half mentions Ada. " + ("context " * 70) + "last half mentions Engine."
        llm = StubLLM(
            LLMOutputTruncatedError("truncated"),
            {
                "entities": [{"name": "Ada", "type": "Person", "attributes": {}}],
                "relations": [],
            },
            {
                "entities": [{"name": "Engine", "type": "Machine", "attributes": {}}],
                "relations": [{
                    "source": "Ada",
                    "target": "Engine",
                    "type": "BUILT",
                    "fact": "Ada built the Engine.",
                }],
            },
        )

        result = NERExtractor(llm_client=llm).extract(text, {})

        self.assertEqual(len(llm.messages), 3)
        self.assertEqual(
            [entity["name"] for entity in result["entities"]],
            ["Ada", "Engine"],
        )
        self.assertEqual(result["relations"][0]["target"], "Engine")
        prefix = "Extract entities and relations from this text:\n"
        source_chunks = [messages[1]["content"][len(prefix):] for messages in llm.messages[1:]]
        coverage = [False] * len(text)
        for chunk in source_chunks:
            start = text.find(chunk)
            self.assertGreaterEqual(start, 0)
            for index in range(start, start + len(chunk)):
                coverage[index] = True
        self.assertTrue(all(coverage))

    def test_compact_ontology_keeps_descriptions_attributes_and_endpoints(self):
        ontology = {
            "entity_types": [{
                "name": "Person",
                "description": "A named individual.",
                "attributes": [{"name": "role", "description": "Their position."}],
            }],
            "relation_types": [{
                "name": "WORKS_AT",
                "description": "Employment relationship.",
                "source_targets": [{"source": "Person", "target": "Company"}],
            }],
        }

        formatted = NERExtractor(llm_client=StubLLM({}))._format_ontology(ontology)
        self.assertEqual(
            json.loads(formatted),
            {
                "entity_types": ontology["entity_types"],
                "relation_types": ontology["relation_types"],
            },
        )
        self.assertLess(len(formatted), len(json.dumps(ontology, indent=2)))

    def test_invalid_schema_fails_instead_of_returning_empty_extraction(self):
        extractor = NERExtractor(llm_client=StubLLM({"entities": "not a list", "relations": []}))
        with self.assertRaisesRegex(ValueError, "must be lists"):
            extractor.extract("Text with entities", {})

    def test_unconfirmed_free_tier_blocks_groq_client_creation(self):
        extractor = NERExtractor()
        with (
            patch.object(Config, "GROQ_API_KEY", "configured"),
            patch.object(Config, "GROQ_GRAPH_MODEL", "qwen/qwen3.8-27b"),
            patch.object(Config, "GROQ_GRAPH_FREE_TIER_CONFIRMED", False),
        ):
            with self.assertRaisesRegex(ValueError, "until you verify"):
                extractor._get_client()

    def test_confirmed_graph_route_uses_dedicated_groq_configuration(self):
        extractor = NERExtractor()
        with (
            patch.object(Config, "GROQ_API_KEY", "test-key"),
            patch.object(Config, "GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            patch.object(Config, "GROQ_GRAPH_MODEL", "test-model"),
            patch.object(Config, "GROQ_GRAPH_FREE_TIER_CONFIRMED", True),
            patch.object(Config, "GROQ_GRAPH_TIMEOUT_SECONDS", 45),
            patch("app.storage.ner_extractor.LLMClient") as client_class,
        ):
            client = extractor._get_client()

        self.assertIs(client, client_class.return_value)
        client_class.assert_called_once_with(
            api_key="test-key",
            base_url="https://api.groq.com/openai/v1",
            model="test-model",
            timeout=45,
        )

    def test_rate_limit_retries_once_after_bounded_retry_after(self):
        response = httpx.Response(
            429,
            request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
            headers={"retry-after": "1"},
        )
        rate_limit = RateLimitError("rate limited", response=response, body={})
        llm = StubLLM(rate_limit, {"entities": [], "relations": []})

        with patch("app.storage.ner_extractor.time.sleep") as sleep:
            result = NERExtractor(llm_client=llm).extract("Text", {})

        self.assertEqual(result, {"entities": [], "relations": []})
        self.assertEqual(len(llm.messages), 2)
        sleep.assert_called_once_with(1.0)

    def test_rate_limit_uses_bounded_delay_from_groq_error_message(self):
        response = httpx.Response(
            429,
            request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
        )
        rate_limit = RateLimitError(
            "rate limited",
            response=response,
            body={"error": {"message": "Please try again in 27s."}},
        )

        self.assertEqual(NERExtractor._rate_limit_retry_delay(rate_limit), 27)

    def test_graph_builder_reports_progress_after_each_chunk(self):
        storage = MagicMock()
        storage.add_text.side_effect = ["episode-1", "episode-2", "episode-3"]
        service = GraphBuilderService(storage=storage)
        progress = []

        episodes = service.add_text_batches(
            "graph-1",
            ["first", "middle", "last"],
            batch_size=2,
            progress_callback=lambda message, ratio: progress.append((message, ratio)),
        )

        self.assertEqual(episodes, ["episode-1", "episode-2", "episode-3"])
        self.assertEqual([ratio for _, ratio in progress], [1 / 3, 2 / 3, 1])
        self.assertIn("3/3", progress[-1][0])

    def test_graph_build_retry_detects_missing_or_finished_tasks_as_stale(self):
        task_manager = TaskManager()
        task_id = task_manager.create_task("graph_build")

        self.assertTrue(_has_active_graph_task(task_id))
        self.assertFalse(_has_active_graph_task("missing-task"))

        task_manager.complete_task(task_id, {})
        self.assertFalse(_has_active_graph_task(task_id))

    def test_chat_json_retries_without_only_unsupported_response_format(self):
        response = httpx.Response(
            400,
            request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
        )
        unsupported = BadRequestError(
            "response_format is not supported",
            response=response,
            body={"message": "response_format is not supported"},
        )
        completion = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content='{"entities":[]}'),
                finish_reason="stop",
            )]
        )
        create = MagicMock(side_effect=[unsupported, completion])

        client = LLMClient(api_key="test-key", base_url="https://api.groq.com/openai/v1", model="test")
        client.client.chat.completions.create = create
        result = client.chat_json([{"role": "user", "content": "return json"}])

        self.assertEqual(result, {"entities": []})
        self.assertEqual(create.call_count, 2)
        self.assertIn("response_format", create.call_args_list[0].kwargs)
        self.assertNotIn("response_format", create.call_args_list[1].kwargs)

    def test_chat_raises_specific_error_when_provider_truncates_output(self):
        completion = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content='{"entities":['),
                finish_reason="length",
            )]
        )
        client = LLMClient(api_key="test-key", base_url="https://api.groq.com/openai/v1", model="test")
        client.client.chat.completions.create = MagicMock(return_value=completion)

        with self.assertRaisesRegex(LLMOutputTruncatedError, "output-token limit"):
            client.chat_json([{"role": "user", "content": "return json"}])


if __name__ == "__main__":
    unittest.main()
