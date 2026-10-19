import tempfile
import unittest
from unittest.mock import MagicMock, patch

from app.config import Config
from app.services.graph_tools import InsightForgeResult
from app.services.report_agent import (
    Report,
    ReportAgent,
    ReportManager,
    ReportOutline,
    ReportSection,
    ReportStatus,
    _REPORT_MAX_OUTPUT_TOKENS,
)
from app.utils.llm_client import LLMOutputTruncatedError


class InsightForgeResultTests(unittest.TestCase):
    def test_report_llm_output_stays_below_groq_free_tier_limit(self):
        self.assertEqual(_REPORT_MAX_OUTPUT_TOKENS, 800)

    def test_report_agent_uses_configured_tool_call_limit(self):
        self.assertEqual(
            ReportAgent.MAX_TOOL_CALLS_PER_SECTION,
            Config.REPORT_AGENT_MAX_TOOL_CALLS,
        )

    def test_report_context_can_limit_facts_and_omit_repeated_scenario(self):
        result = InsightForgeResult(
            query="What changed?",
            simulation_requirement="Long scenario details",
            sub_queries=[],
            semantic_facts=[f"Fact {index}" for index in range(12)],
            total_facts=12,
        )

        report_text = result.to_text(max_facts=10, include_simulation_requirement=False)

        self.assertNotIn("Long scenario details", report_text)
        self.assertIn('"Fact 0"', report_text)
        self.assertIn('"Fact 9"', report_text)
        self.assertNotIn('"Fact 10"', report_text)
        self.assertIn("... and 2 additional related facts", report_text)

    def test_default_text_keeps_full_scenario_and_facts(self):
        result = InsightForgeResult(
            query="What changed?",
            simulation_requirement="Scenario details",
            sub_queries=[],
            semantic_facts=["Fact 1", "Fact 2"],
            total_facts=2,
        )

        full_text = result.to_text()

        self.assertIn("Scenario details", full_text)
        self.assertIn('"Fact 1"', full_text)
        self.assertIn('"Fact 2"', full_text)

    def test_truncated_llm_response_is_continued_before_returning(self):
        agent = ReportAgent(
            graph_id="graph",
            simulation_id="simulation",
            simulation_requirement="requirement",
            llm_client=MagicMock(),
            graph_tools=MagicMock(),
        )
        agent.llm.chat.side_effect = [
            LLMOutputTruncatedError("truncated", content="Final Ans"),
            "wer: complete",
        ]

        response = agent._chat_with_continuation(
            [{"role": "user", "content": "Write a section"}],
            temperature=0.5,
        )

        self.assertEqual(response, "Final Answer: complete")
        continuation_messages = agent.llm.chat.call_args_list[1].kwargs["messages"]
        self.assertEqual(continuation_messages[-2]["content"], "Final Ans")
        self.assertIn("Continue exactly", continuation_messages[-1]["content"])

    def test_react_context_bounds_tool_results_and_discards_redundant_assistant_turns(self):
        tool_result = "START" + ("x" * 3000) + "END"
        bounded_result = ReportAgent._bound_tool_result(tool_result)
        self.assertLessEqual(len(bounded_result), 700)
        self.assertTrue(bounded_result.startswith("START"))
        self.assertTrue(bounded_result.endswith("END"))

        compacted = ReportAgent._compact_react_messages([
            {"role": "system", "content": "system"},
            {"role": "user", "content": "original request"},
            {"role": "assistant", "content": "verbose tool call"},
            {"role": "user", "content": "retrieval evidence"},
            {"role": "assistant", "content": "verbose analysis"},
            {"role": "user", "content": "continue instruction"},
        ])

        self.assertEqual([message["role"] for message in compacted], ["system", "user"])
        self.assertIn("original request", compacted[1]["content"])
        self.assertIn("retrieval evidence", compacted[1]["content"])
        self.assertIn("continue instruction", compacted[1]["content"])
        self.assertNotIn("verbose analysis", compacted[1]["content"])

    def test_long_simulation_requirement_keeps_brief_ends_and_competitor_evidence(self):
        requirement = (
            "Independent market review. " + ("background " * 150)
            + "\nCompare Amadeus HotSOS, GoAudits, Optii and SafetyCulture using verified evidence."
            + "\n" + ("irrelevant detail " * 150)
            + "\nLabel unknowns and simulation assumptions. No predetermined winner."
        )

        compacted = ReportAgent._compact_simulation_requirement(requirement)

        self.assertLess(len(compacted), len(requirement))
        self.assertIn("Independent market review.", compacted)
        self.assertIn("Amadeus HotSOS", compacted)
        self.assertIn("SafetyCulture", compacted)
        self.assertIn("No predetermined winner.", compacted)

    def test_resume_reuses_outline_and_preserves_completed_sections(self):
        with tempfile.TemporaryDirectory() as reports_dir:
            outline = ReportOutline(
                title="Report",
                summary="Summary",
                sections=[
                    ReportSection("Saved section"),
                    ReportSection("Missing section"),
                ],
            )
            report = Report(
                report_id="report_resume_test",
                simulation_id="simulation",
                graph_id="graph",
                simulation_requirement="requirement",
                status=ReportStatus.FAILED,
                outline=outline,
                error="previous failure",
            )

            with (
                patch.object(ReportManager, "REPORTS_DIR", reports_dir),
                patch("app.services.report_agent.ReportLogger"),
                patch("app.services.report_agent.ReportConsoleLogger"),
                patch.object(ReportAgent, "plan_outline") as plan_outline,
                patch.object(
                    ReportAgent,
                    "_generate_section_react",
                    return_value="Newly generated content",
                ) as generate_section,
            ):
                ReportManager.save_report(report)
                ReportManager.save_section(
                    report.report_id,
                    1,
                    ReportSection("Saved section", "Preserve this content"),
                )
                agent = ReportAgent(
                    graph_id="graph",
                    simulation_id="simulation",
                    simulation_requirement="requirement",
                    llm_client=MagicMock(),
                    graph_tools=MagicMock(),
                )

                resumed = agent.generate_report(report_id=report.report_id, resume=True)

                plan_outline.assert_not_called()
                generate_section.assert_called_once()
                self.assertEqual(generate_section.call_args.kwargs["section_index"], 2)
                self.assertEqual(resumed.status, ReportStatus.COMPLETED)
                saved_sections = ReportManager.get_generated_sections(report.report_id)
                self.assertEqual(len(saved_sections), 2)
                self.assertIn("Preserve this content", saved_sections[0]["content"])
                self.assertIn("Newly generated content", saved_sections[1]["content"])


if __name__ == "__main__":
    unittest.main()
