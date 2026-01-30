"""
Tests for the custom writing instructions feature.

Verifies that custom_instructions flows correctly through the entire pipeline:
  Frontend → server_utils → websocket_manager → report types → agent → writer → LLM prompt

Tests are organized by layer, from extraction to prompt injection.

Note: backend/ modules use bare imports (e.g. `from utils import ...`) rather than
package-relative imports, so they can't be imported directly as `backend.server.X`.
We add backend/ to sys.path where needed to make those imports resolve.
"""

import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

# Backend modules need their parent on sys.path for bare imports to resolve
_BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..", "backend")
_BACKEND_SERVER_DIR = os.path.join(_BACKEND_DIR, "server")


def _with_backend_on_path(fn):
    """Decorator that temporarily adds backend/ to sys.path for import resolution."""
    def wrapper(*args, **kwargs):
        added = []
        for d in (_BACKEND_DIR, _BACKEND_SERVER_DIR):
            d = os.path.abspath(d)
            if d not in sys.path:
                sys.path.insert(0, d)
                added.append(d)
        try:
            return fn(*args, **kwargs)
        finally:
            for d in added:
                sys.path.remove(d)
    return wrapper


class TestExtractCommandData(unittest.TestCase):
    """Tests for server_utils.extract_command_data — entry point for custom_instructions."""

    @_with_backend_on_path
    def _extract(self, data):
        from server_utils import extract_command_data
        return extract_command_data(data)

    def test_extracts_custom_instructions(self):
        data = {"task": "test", "custom_instructions": "Use bullet points"}
        result = self._extract(data)
        # custom_instructions is the last element of the returned tuple
        self.assertEqual(result[-1], "Use bullet points")

    def test_defaults_to_empty_string(self):
        data = {"task": "test"}
        result = self._extract(data)
        self.assertEqual(result[-1], "")

    def test_empty_string_preserved(self):
        data = {"task": "test", "custom_instructions": ""}
        result = self._extract(data)
        self.assertEqual(result[-1], "")


class TestRunAgentGuard(unittest.TestCase):
    """Tests the guard logic: only research_report and detailed_report receive instructions.

    websocket_manager.py uses mixed import styles (bare + relative) making it
    un-importable from the test root. We test the guard logic directly — it's a
    one-liner from run_agent() line 137-138:
        effective_instructions = custom_instructions if report_type in (
            ReportType.DetailedReport.value, ReportType.ResearchReport.value) else ""
    """

    def _apply_guard(self, report_type, custom_instructions):
        from gpt_researcher.utils.enum import ReportType
        return custom_instructions if report_type in (
            ReportType.DetailedReport.value, ReportType.ResearchReport.value) else ""

    def test_research_report_receives_instructions(self):
        result = self._apply_guard("research_report", "formal tone")
        self.assertEqual(result, "formal tone")

    def test_detailed_report_receives_instructions(self):
        result = self._apply_guard("detailed_report", "APA format")
        self.assertEqual(result, "APA format")

    def test_unsupported_report_type_gets_empty_instructions(self):
        result = self._apply_guard("outline_report", "should be ignored")
        self.assertEqual(result, "")

    def test_multi_agents_gets_empty_instructions(self):
        result = self._apply_guard("multi_agents", "should be ignored")
        self.assertEqual(result, "")

    def test_empty_instructions_pass_through_for_supported_type(self):
        result = self._apply_guard("research_report", "")
        self.assertEqual(result, "")


class TestBasicReportPassthrough(unittest.TestCase):
    """Tests that BasicReport stores instructions and passes them only at write time."""

    @patch('langchain_openai.OpenAIEmbeddings')
    def test_stores_custom_instructions(self, _mock_embed):
        from backend.report_type.basic_report.basic_report import BasicReport

        report = BasicReport(
            query="test",
            query_domains=[],
            report_type="research_report",
            report_source="web",
            source_urls=[],
            document_urls=[],
            tone="Objective",
            config_path="default",
            websocket=MagicMock(),
            custom_instructions="bullet points only",
        )
        self.assertEqual(report.custom_instructions, "bullet points only")

    @patch('langchain_openai.OpenAIEmbeddings')
    def test_does_not_pass_instructions_to_researcher_init(self, _mock_embed):
        """custom_instructions must NOT be passed to GPTResearcher constructor."""
        from backend.report_type.basic_report.basic_report import BasicReport

        with patch('backend.report_type.basic_report.basic_report.GPTResearcher') as mock_gptr:
            mock_gptr.return_value = MagicMock()
            BasicReport(
                query="test",
                query_domains=[],
                report_type="research_report",
                report_source="web",
                source_urls=[],
                document_urls=[],
                tone="Objective",
                config_path="default",
                websocket=MagicMock(),
                custom_instructions="should not appear in GPTResearcher",
            )
            _, kwargs = mock_gptr.call_args
            self.assertNotIn("custom_instructions", kwargs)

    @patch('langchain_openai.OpenAIEmbeddings')
    def test_run_passes_instructions_to_write_report(self, _mock_embed):
        from backend.report_type.basic_report.basic_report import BasicReport

        with patch('backend.report_type.basic_report.basic_report.GPTResearcher') as mock_gptr:
            instance = MagicMock()
            instance.conduct_research = AsyncMock()
            instance.write_report = AsyncMock(return_value="final report")
            mock_gptr.return_value = instance

            report = BasicReport(
                query="test",
                query_domains=[],
                report_type="research_report",
                report_source="web",
                source_urls=[],
                document_urls=[],
                tone="Objective",
                config_path="default",
                websocket=MagicMock(),
                custom_instructions="use formal tone",
            )
            result = asyncio.run(report.run())

            instance.write_report.assert_called_once_with(
                custom_instructions="use formal tone")
            self.assertEqual(result, "final report")


class TestDetailedReportPassthrough(unittest.TestCase):
    """Tests that DetailedReport passes instructions to introduction, subtopics, and conclusion."""

    @patch('langchain_openai.OpenAIEmbeddings')
    def test_stores_custom_instructions(self, _mock_embed):
        from backend.report_type.detailed_report.detailed_report import DetailedReport

        report = DetailedReport(
            query="test",
            report_type="detailed_report",
            report_source="web",
            custom_instructions="academic style",
        )
        self.assertEqual(report.custom_instructions, "academic style")

    @patch('langchain_openai.OpenAIEmbeddings')
    def test_does_not_pass_instructions_to_researcher_init(self, _mock_embed):
        from backend.report_type.detailed_report.detailed_report import DetailedReport

        with patch('backend.report_type.detailed_report.detailed_report.GPTResearcher') as mock_gptr:
            mock_gptr.return_value = MagicMock()
            mock_gptr.return_value.visited_urls = set()
            DetailedReport(
                query="test",
                report_type="detailed_report",
                report_source="web",
                custom_instructions="should not appear in GPTResearcher",
            )
            _, kwargs = mock_gptr.call_args
            self.assertNotIn("custom_instructions", kwargs)


class TestGenerateReportPromptInjection(unittest.TestCase):
    """Tests that generate_report appends custom_instructions to the LLM prompt content."""

    @patch('gpt_researcher.actions.report_generation.create_chat_completion', new_callable=AsyncMock)
    def test_instructions_appended_to_prompt(self, mock_llm):
        from gpt_researcher.actions.report_generation import generate_report
        from gpt_researcher.utils.enum import Tone

        mock_llm.return_value = "Generated report"
        cfg = MagicMock()
        cfg.report_format = "apa"
        cfg.total_words = 1000
        cfg.language = "en"
        cfg.smart_llm_model = "test-model"
        cfg.smart_llm_provider = "openai"
        cfg.smart_token_limit = 4000
        cfg.llm_kwargs = {}

        asyncio.run(generate_report(
            query="test query",
            context="some context",
            agent_role_prompt="You are a researcher",
            report_type="research_report",
            tone=Tone.Objective,
            report_source="web",
            websocket=None,
            cfg=cfg,
            custom_instructions="Write in bullet points, formal tone",
        ))

        # The user message should contain the custom instructions block
        call_kwargs = mock_llm.call_args[1]
        user_content = call_kwargs["messages"][1]["content"]
        self.assertIn("USER WRITING INSTRUCTIONS", user_content)
        self.assertIn("Write in bullet points, formal tone", user_content)

    @patch('gpt_researcher.actions.report_generation.create_chat_completion', new_callable=AsyncMock)
    def test_empty_instructions_not_appended(self, mock_llm):
        from gpt_researcher.actions.report_generation import generate_report
        from gpt_researcher.utils.enum import Tone

        mock_llm.return_value = "Generated report"
        cfg = MagicMock()
        cfg.report_format = "apa"
        cfg.total_words = 1000
        cfg.language = "en"
        cfg.smart_llm_model = "test-model"
        cfg.smart_llm_provider = "openai"
        cfg.smart_token_limit = 4000
        cfg.llm_kwargs = {}

        asyncio.run(generate_report(
            query="test query",
            context="some context",
            agent_role_prompt="You are a researcher",
            report_type="research_report",
            tone=Tone.Objective,
            report_source="web",
            websocket=None,
            cfg=cfg,
            custom_instructions="",
        ))

        call_kwargs = mock_llm.call_args[1]
        user_content = call_kwargs["messages"][1]["content"]
        self.assertNotIn("USER WRITING INSTRUCTIONS", user_content)


class TestWriteIntroductionPromptInjection(unittest.TestCase):
    """Tests that write_report_introduction appends custom_instructions."""

    @patch('gpt_researcher.actions.report_generation.create_chat_completion', new_callable=AsyncMock)
    def test_instructions_appended_to_intro_prompt(self, mock_llm):
        from gpt_researcher.actions.report_generation import write_report_introduction
        from gpt_researcher.config.config import Config

        mock_llm.return_value = "Introduction text"
        config = Config()

        asyncio.run(write_report_introduction(
            query="test query",
            context="research context",
            agent_role_prompt="You are a researcher",
            config=config,
            custom_instructions="Use academic language",
        ))

        call_kwargs = mock_llm.call_args[1]
        user_content = call_kwargs["messages"][1]["content"]
        self.assertIn("Additional writing instructions:", user_content)
        self.assertIn("Use academic language", user_content)

    @patch('gpt_researcher.actions.report_generation.create_chat_completion', new_callable=AsyncMock)
    def test_empty_instructions_not_appended_to_intro(self, mock_llm):
        from gpt_researcher.actions.report_generation import write_report_introduction
        from gpt_researcher.config.config import Config

        mock_llm.return_value = "Introduction text"
        config = Config()

        asyncio.run(write_report_introduction(
            query="test query",
            context="research context",
            agent_role_prompt="You are a researcher",
            config=config,
            custom_instructions="",
        ))

        call_kwargs = mock_llm.call_args[1]
        user_content = call_kwargs["messages"][1]["content"]
        self.assertNotIn("Additional writing instructions:", user_content)


class TestWriteConclusionPromptInjection(unittest.TestCase):
    """Tests that write_conclusion appends custom_instructions."""

    @patch('gpt_researcher.actions.report_generation.create_chat_completion', new_callable=AsyncMock)
    def test_instructions_appended_to_conclusion_prompt(self, mock_llm):
        from gpt_researcher.actions.report_generation import write_conclusion
        from gpt_researcher.config.config import Config

        mock_llm.return_value = "Conclusion text"
        config = Config()

        asyncio.run(write_conclusion(
            query="test query",
            context="report body content",
            agent_role_prompt="You are a researcher",
            config=config,
            custom_instructions="End with recommendations",
        ))

        call_kwargs = mock_llm.call_args[1]
        user_content = call_kwargs["messages"][1]["content"]
        self.assertIn("Additional writing instructions:", user_content)
        self.assertIn("End with recommendations", user_content)

    @patch('gpt_researcher.actions.report_generation.create_chat_completion', new_callable=AsyncMock)
    def test_empty_instructions_not_appended_to_conclusion(self, mock_llm):
        from gpt_researcher.actions.report_generation import write_conclusion
        from gpt_researcher.config.config import Config

        mock_llm.return_value = "Conclusion text"
        config = Config()

        asyncio.run(write_conclusion(
            query="test query",
            context="report body content",
            agent_role_prompt="You are a researcher",
            config=config,
            custom_instructions="",
        ))

        call_kwargs = mock_llm.call_args[1]
        user_content = call_kwargs["messages"][1]["content"]
        self.assertNotIn("Additional writing instructions:", user_content)


if __name__ == "__main__":
    unittest.main()
