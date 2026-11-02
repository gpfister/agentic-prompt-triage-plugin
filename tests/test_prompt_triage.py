"""Tests for dynamic prompt triage hook and Laya Router System 1 model."""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
import pytest

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from hooks.prompt_triage import (
    HookContext,
    HookResult,
    PromptTriageHook,
    prompt_triage_pre_turn,
    session_start_triage,
    extract_prompt_text,
    clean_user_prompt,
    detect_installed_plugins,
    build_dynamic_triage_spec,
    LayaTriageEngine,
    CONFIDENCE_THRESHOLD,
    TRIAGE_QUESTION,
    PRIMARY_PLUGINS,
    LEGACY_PLUGIN_ALIASES,
    KNOWN_PLUGINS,
)


def test_clean_user_prompt():
    raw1 = (
        "<USER_REQUEST>\nExplain this code\n</USER_REQUEST>\n"
        "<ADDITIONAL_METADATA>\nThe current local time is: 2026-10-26T10:55:12+01:00.\n</ADDITIONAL_METADATA>"
    )
    assert clean_user_prompt(raw1) == "Explain this code"

    raw2 = (
        "<USER_REQUEST>\n"
        "Improve prompt_triage.py -> lower the threshold to 0.4\n"
        "/path/to/file1.py\n"
        "/path/to/file2.py\n"
        "</USER_REQUEST>\n"
        "<ADDITIONAL_METADATA>some metadata</ADDITIONAL_METADATA>"
    )
    assert clean_user_prompt(raw2) == "Improve prompt_triage.py -> lower the threshold to 0.4"

    raw3 = "Please refactor the database connector"
    assert clean_user_prompt(raw3) == "Please refactor the database connector"


def test_extract_prompt_text():
    assert extract_prompt_text("simple string") == "simple string"
    assert extract_prompt_text(["part1", "part2"]) == "part1 part2"


def test_detect_installed_plugins_custom():
    plugins = detect_installed_plugins(custom_plugins={"agentic-coding-plugin"})
    assert plugins == {"agentic-coding-plugin"}


def test_detect_installed_plugins_research():
    plugins = detect_installed_plugins(custom_plugins={"agentic-research-plugin"})
    assert plugins == {"agentic-research-plugin"}


def test_detect_installed_plugins_obsidian():
    plugins = detect_installed_plugins(custom_plugins={"agentic-obsidian-plugin"})
    assert plugins == {"agentic-obsidian-plugin"}


def test_detect_installed_plugins_document():
    plugins = detect_installed_plugins(custom_plugins={"agentic-document-plugin"})
    assert plugins == {"agentic-document-plugin"}


def test_detect_installed_plugins_monitor():
    plugins = detect_installed_plugins(custom_plugins={"agentic-monitor-plugin"})
    assert plugins == {"agentic-monitor-plugin"}


def test_detect_installed_plugins_monitoring_legacy():
    plugins = detect_installed_plugins(custom_plugins={"agentic-monitoring-plugin"})
    assert plugins == {"agentic-monitoring-plugin"}


def test_detect_installed_plugins_canonical_task_monitoring():
    plugins = detect_installed_plugins(custom_plugins={"agentic-task-monitoring-plugin"})
    assert plugins == {"agentic-task-monitoring-plugin"}


def test_detect_installed_plugins_canonical_issue_investigation():
    plugins = detect_installed_plugins(custom_plugins={"agentic-issue-investigation-plugin"})
    assert plugins == {"agentic-issue-investigation-plugin"}


def test_detect_installed_plugins_legacy_investigate():
    plugins = detect_installed_plugins(custom_plugins={"agentic-investigate-plugin"})
    assert plugins == {"agentic-investigate-plugin"}


def test_detect_installed_plugins_canonical_topic_researching():
    plugins = detect_installed_plugins(custom_plugins={"agentic-topic-researching-plugin"})
    assert plugins == {"agentic-topic-researching-plugin"}


def test_detect_installed_plugins_canonical_project_documenting():
    plugins = detect_installed_plugins(custom_plugins={"agentic-project-documenting-plugin"})
    assert plugins == {"agentic-project-documenting-plugin"}


def test_detect_installed_plugins_env_canonical(monkeypatch):
    monkeypatch.setenv(
        "INSTALLED_PLUGINS",
        "agentic-coding-plugin, agentic-task-monitoring-plugin, agentic-issue-investigation-plugin, agentic-topic-researching-plugin, agentic-project-documenting-plugin, agentic-project-grooming-plugin"
    )
    plugins = detect_installed_plugins()
    assert plugins == {
        "agentic-coding-plugin",
        "agentic-task-monitoring-plugin",
        "agentic-issue-investigation-plugin",
        "agentic-topic-researching-plugin",
        "agentic-project-documenting-plugin",
        "agentic-project-grooming-plugin",
    }


def test_detect_installed_plugins_env(monkeypatch):
    monkeypatch.setenv("INSTALLED_PLUGINS", "agentic-coding-plugin, agentic-monitoring-plugin, agentic-research-plugin")
    plugins = detect_installed_plugins()
    assert plugins == {"agentic-coding-plugin", "agentic-monitoring-plugin", "agentic-research-plugin"}


def test_detect_installed_plugins_env_primary(monkeypatch):
    monkeypatch.setenv(
        "INSTALLED_PLUGINS",
        "agentic-coding-plugin, agentic-monitor-plugin, agentic-document-plugin, agentic-research-plugin"
    )
    plugins = detect_installed_plugins()
    assert plugins == {
        "agentic-coding-plugin",
        "agentic-monitor-plugin",
        "agentic-document-plugin",
        "agentic-research-plugin",
    }


@pytest.fixture
def workspace_with_mixed_plugin_directories(tmp_path: Path) -> Path:
    ws = tmp_path / "mixed_ws"
    ws.mkdir()
    # Primary plugins with manifest
    for name in ["agentic-monitor-plugin", "agentic-document-plugin"]:
        pdir = ws / name
        pdir.mkdir()
        manifest = pdir / "plugin.json"
        manifest.write_text(json.dumps({"name": name, "version": "0.1.0"}), encoding="utf-8")
    # Legacy plugins with manifest
    for name in ["agentic-monitoring-plugin", "agentic-obsidian-plugin"]:
        pdir = ws / name
        pdir.mkdir()
        manifest = pdir / "plugin.json"
        manifest.write_text(json.dumps({"name": name, "version": "0.1.0"}), encoding="utf-8")
    # Directory without manifest matching KNOWN_PLUGINS
    pdir_no_manifest = ws / "agentic-research-plugin"
    pdir_no_manifest.mkdir()
    return ws


def test_detect_installed_plugins_workspace_fixture(workspace_with_mixed_plugin_directories: Path):
    detected = detect_installed_plugins(workspace_paths=[str(workspace_with_mixed_plugin_directories)])
    assert "agentic-monitor-plugin" in detected
    assert "agentic-document-plugin" in detected
    assert "agentic-monitoring-plugin" in detected
    assert "agentic-obsidian-plugin" in detected
    assert "agentic-research-plugin" in detected


@pytest.fixture
def workspace_with_canonical_plugin_directories(tmp_path: Path) -> Path:
    ws = tmp_path / "canonical_ws"
    ws.mkdir()
    canonical_names = [
        "agentic-coding-plugin",
        "agentic-issue-investigation-plugin",
        "agentic-task-monitoring-plugin",
        "agentic-topic-researching-plugin",
        "agentic-project-documenting-plugin",
        "agentic-project-grooming-plugin",
    ]
    for name in canonical_names:
        pdir = ws / name
        pdir.mkdir()
        manifest = pdir / "plugin.json"
        manifest.write_text(json.dumps({"name": name, "version": "0.1.0"}), encoding="utf-8")
    return ws


def test_detect_installed_plugins_all_canonical_directories(workspace_with_canonical_plugin_directories: Path):
    detected = detect_installed_plugins(workspace_paths=[str(workspace_with_canonical_plugin_directories)])
    assert "agentic-coding-plugin" in detected
    assert "agentic-issue-investigation-plugin" in detected
    assert "agentic-task-monitoring-plugin" in detected
    assert "agentic-topic-researching-plugin" in detected
    assert "agentic-project-documenting-plugin" in detected
    assert "agentic-project-grooming-plugin" in detected


def test_build_dynamic_triage_spec_canonical_investigation():
    spec = build_dynamic_triage_spec({"agentic-issue-investigation-plugin"})
    assert "response 1" in spec.responses
    assert "response 2" in spec.responses
    assert spec.response_to_intent["response 2"] == "investigation"
    assert spec.intent_to_response["investigation"] == "response 2"
    assert "@coding-investigate" in spec.response_to_directive["response 2"]


def test_build_dynamic_triage_spec_canonical_monitoring():
    spec = build_dynamic_triage_spec({"agentic-task-monitoring-plugin"})
    assert "response 1" in spec.responses
    assert "response 2" in spec.responses
    assert spec.response_to_intent["response 2"] == "monitoring"
    assert spec.intent_to_response["monitoring"] == "response 2"
    assert "@coding-monitor" in spec.response_to_directive["response 2"]


def test_build_dynamic_triage_spec_canonical_research():
    spec = build_dynamic_triage_spec({"agentic-topic-researching-plugin"})
    assert "response 1" in spec.responses
    assert "response 2" in spec.responses
    assert spec.response_to_intent["response 2"] == "research"
    assert spec.intent_to_response["research"] == "response 2"
    assert "@coding-research" in spec.response_to_directive["response 2"]


def test_build_dynamic_triage_spec_canonical_documenting():
    spec = build_dynamic_triage_spec({"agentic-project-documenting-plugin"})
    assert "response 1" in spec.responses
    assert "response 2" in spec.responses
    assert spec.response_to_intent["response 2"] == "documentation"
    assert spec.intent_to_response["documentation"] == "response 2"
    assert "@coding-document" in spec.response_to_directive["response 2"]


def test_build_dynamic_triage_spec_all_six_canonical():
    spec = build_dynamic_triage_spec({
        "agentic-coding-plugin",
        "agentic-issue-investigation-plugin",
        "agentic-task-monitoring-plugin",
        "agentic-topic-researching-plugin",
        "agentic-project-documenting-plugin",
        "agentic-project-grooming-plugin",
    })
    assert len(spec.responses) == 8
    assert spec.intent_to_response["question"] == "response 1"
    assert spec.intent_to_response["coding"] == "response 2"
    assert spec.intent_to_response["monitoring"] == "response 3"
    assert spec.intent_to_response["investigation"] == "response 4"
    assert spec.intent_to_response["research"] == "response 5"
    assert spec.intent_to_response["documentation"] == "response 6"
    assert spec.intent_to_response["grooming"] == "response 7"
    assert spec.intent_to_response["mixed"] == "response 8"
    assert spec.response_to_intent["response 2"] == "coding"
    assert spec.response_to_intent["response 3"] == "monitoring"
    assert spec.response_to_intent["response 4"] == "investigation"
    assert spec.response_to_intent["response 5"] == "research"
    assert spec.response_to_intent["response 6"] == "documentation"
    assert spec.response_to_intent["response 7"] == "grooming"
    assert spec.response_to_intent["response 8"] == "mixed"


def test_build_dynamic_triage_spec_all():
    spec = build_dynamic_triage_spec(
        {"agentic-coding-plugin", "agentic-investigate-plugin", "agentic-monitoring-plugin"}
    )
    assert "response 1" in spec.responses
    assert "response 2" in spec.responses
    assert spec.response_to_intent["response 2"] == "coding"
    assert "coding" in spec.intent_to_response
    assert "investigation" in spec.intent_to_response
    assert "monitoring" in spec.intent_to_response
    mixed_key = spec.intent_to_response.get("mixed")
    assert mixed_key is not None
    assert spec.response_to_intent[mixed_key] == "mixed"


def test_build_dynamic_triage_spec_only_coding():
    spec = build_dynamic_triage_spec({"agentic-coding-plugin"})
    assert "response 1" in spec.responses
    assert "response 2" in spec.responses
    assert "investigation" not in spec.intent_to_response
    assert "monitoring" not in spec.intent_to_response
    assert spec.response_to_intent["response 2"] == "coding"
    assert spec.intent_to_response.get("mixed") == "response 3"


def test_build_dynamic_triage_spec_only_investigate():
    spec = build_dynamic_triage_spec({"agentic-investigate-plugin"})
    assert "response 1" in spec.responses
    assert "response 2" in spec.responses
    assert spec.response_to_intent["response 2"] == "investigation"
    assert "coding" not in spec.intent_to_response
    assert spec.intent_to_response.get("mixed") == "response 3"


def test_build_dynamic_triage_spec_only_research():
    spec = build_dynamic_triage_spec({"agentic-research-plugin"})
    assert "response 1" in spec.responses
    assert "response 2" in spec.responses
    assert spec.responses["response 2"] == "one or more topic research or information retrieval requests"
    assert spec.response_to_intent["response 2"] == "research"
    assert spec.intent_to_response["research"] == "response 2"
    assert "coding" not in spec.intent_to_response
    assert "investigation" not in spec.intent_to_response
    assert "monitoring" not in spec.intent_to_response
    assert spec.intent_to_response.get("mixed") == "response 3"
    assert "@coding-research" in spec.response_to_directive["response 2"]
    assert "Topic Researcher" in spec.response_to_directive["response 2"]


def test_build_dynamic_triage_spec_only_obsidian():
    spec = build_dynamic_triage_spec({"agentic-obsidian-plugin"})
    assert "response 1" in spec.responses
    assert "response 2" in spec.responses
    assert spec.responses["response 2"] == "one or more documentation request"
    assert spec.response_to_intent["response 2"] == "documentation"
    assert spec.intent_to_response["documentation"] == "response 2"
    assert "coding" not in spec.intent_to_response
    assert "investigation" not in spec.intent_to_response
    assert "monitoring" not in spec.intent_to_response
    assert "research" not in spec.intent_to_response
    assert spec.intent_to_response.get("mixed") == "response 3"
    assert "@coding-document" in spec.response_to_directive["response 2"]
    assert "Documentation Specialist" in spec.response_to_directive["response 2"]


def test_build_dynamic_triage_spec_only_document():
    spec = build_dynamic_triage_spec({"agentic-document-plugin"})
    assert "response 1" in spec.responses
    assert "response 2" in spec.responses
    assert spec.responses["response 2"] == "one or more documentation request"
    assert spec.response_to_intent["response 2"] == "documentation"
    assert spec.intent_to_response["documentation"] == "response 2"
    assert "coding" not in spec.intent_to_response
    assert "investigation" not in spec.intent_to_response
    assert "monitoring" not in spec.intent_to_response
    assert "research" not in spec.intent_to_response
    assert spec.intent_to_response.get("mixed") == "response 3"
    assert "@coding-document" in spec.response_to_directive["response 2"]
    assert "Documentation Specialist" in spec.response_to_directive["response 2"]


def test_build_dynamic_triage_spec_only_monitor():
    spec = build_dynamic_triage_spec({"agentic-monitor-plugin"})
    assert "response 1" in spec.responses
    assert "response 2" in spec.responses
    assert spec.responses["response 2"] == "one or more task monitoring request"
    assert spec.response_to_intent["response 2"] == "monitoring"
    assert spec.intent_to_response["monitoring"] == "response 2"
    assert "coding" not in spec.intent_to_response
    assert "investigation" not in spec.intent_to_response
    assert "documentation" not in spec.intent_to_response
    assert "research" not in spec.intent_to_response
    assert spec.intent_to_response.get("mixed") == "response 3"
    assert "@coding-monitor" in spec.response_to_directive["response 2"]
    assert "Telemetry Monitor" in spec.response_to_directive["response 2"]


def test_build_dynamic_triage_spec_all_five_primary():
    spec = build_dynamic_triage_spec({
        "agentic-coding-plugin",
        "agentic-investigate-plugin",
        "agentic-monitor-plugin",
        "agentic-research-plugin",
        "agentic-document-plugin",
    })
    assert len(spec.responses) == 7
    assert spec.intent_to_response["question"] == "response 1"
    assert spec.intent_to_response["coding"] == "response 2"
    assert spec.intent_to_response["monitoring"] == "response 3"
    assert spec.intent_to_response["investigation"] == "response 4"
    assert spec.intent_to_response["research"] == "response 5"
    assert spec.intent_to_response["documentation"] == "response 6"
    assert spec.intent_to_response["mixed"] == "response 7"
    assert spec.response_to_intent["response 3"] == "monitoring"
    assert spec.response_to_intent["response 6"] == "documentation"
    assert spec.response_to_intent["response 7"] == "mixed"
    assert "@coding-monitor" in spec.response_to_directive["response 3"]
    assert "@coding-document" in spec.response_to_directive["response 6"]


def test_build_dynamic_triage_spec_all_five():
    spec = build_dynamic_triage_spec({
        "agentic-coding-plugin",
        "agentic-investigate-plugin",
        "agentic-monitoring-plugin",
        "agentic-research-plugin",
        "agentic-obsidian-plugin",
    })
    assert len(spec.responses) == 7
    assert spec.intent_to_response["question"] == "response 1"
    assert spec.intent_to_response["coding"] == "response 2"
    assert spec.intent_to_response["monitoring"] == "response 3"
    assert spec.intent_to_response["investigation"] == "response 4"
    assert spec.intent_to_response["research"] == "response 5"
    assert spec.intent_to_response["documentation"] == "response 6"
    assert spec.intent_to_response["mixed"] == "response 7"
    assert spec.response_to_intent["response 6"] == "documentation"
    assert spec.response_to_intent["response 7"] == "mixed"
    assert "@coding-document" in spec.response_to_directive["response 6"]


def test_heuristic_documentation_request():
    engine = LayaTriageEngine(
        use_router=False,
        installed_plugins={"agentic-obsidian-plugin", "agentic-coding-plugin"},
    )
    choice, conf, probs = engine.evaluate_system1_triage(
        "Please document the project architecture and features in the obsidian vault"
    )
    assert choice == engine.intent_to_response["documentation"]
    assert conf >= 0.90
    assert engine.response_to_intent[choice] == "documentation"


def test_heuristic_documentation_request_document_plugin():
    engine = LayaTriageEngine(
        use_router=False,
        installed_plugins={"agentic-document-plugin", "agentic-coding-plugin"},
    )
    choice, conf, probs = engine.evaluate_system1_triage(
        "Please document the project architecture and features in the vault"
    )
    assert choice == engine.intent_to_response["documentation"]
    assert conf >= 0.90
    assert engine.response_to_intent[choice] == "documentation"


def test_heuristic_monitoring_request_monitor_plugin():
    engine = LayaTriageEngine(
        use_router=False,
        installed_plugins={"agentic-monitor-plugin", "agentic-coding-plugin"},
    )
    choice, conf, probs = engine.evaluate_system1_triage(
        "Please monitor the log stream for failures"
    )
    assert choice == engine.intent_to_response["monitoring"]
    assert conf >= 0.90
    assert engine.response_to_intent[choice] == "monitoring"


def test_heuristic_monitoring_request_monitoring_plugin_legacy():
    engine = LayaTriageEngine(
        use_router=False,
        installed_plugins={"agentic-monitoring-plugin", "agentic-coding-plugin"},
    )
    choice, conf, probs = engine.evaluate_system1_triage(
        "Please monitor the log stream for failures"
    )
    assert choice == engine.intent_to_response["monitoring"]
    assert conf >= 0.90
    assert engine.response_to_intent[choice] == "monitoring"


def test_heuristic_documentation_request_canonical_plugin():
    engine = LayaTriageEngine(
        use_router=False,
        installed_plugins={"agentic-project-documenting-plugin", "agentic-coding-plugin"},
    )
    choice, conf, probs = engine.evaluate_system1_triage(
        "Please document the project architecture and features in the vault"
    )
    assert choice == engine.intent_to_response["documentation"]
    assert conf >= 0.90
    assert engine.response_to_intent[choice] == "documentation"


def test_heuristic_monitoring_request_canonical_plugin():
    engine = LayaTriageEngine(
        use_router=False,
        installed_plugins={"agentic-task-monitoring-plugin", "agentic-coding-plugin"},
    )
    choice, conf, probs = engine.evaluate_system1_triage(
        "Please monitor the log stream for failures"
    )
    assert choice == engine.intent_to_response["monitoring"]
    assert conf >= 0.90
    assert engine.response_to_intent[choice] == "monitoring"


def test_heuristic_investigation_request_canonical_plugin():
    engine = LayaTriageEngine(
        use_router=False,
        installed_plugins={"agentic-issue-investigation-plugin", "agentic-coding-plugin"},
    )
    choice, conf, probs = engine.evaluate_system1_triage(
        "Please investigate why the checkout endpoint returns 500 error"
    )
    assert choice == engine.intent_to_response["investigation"]
    assert conf >= 0.90
    assert engine.response_to_intent[choice] == "investigation"


def test_heuristic_research_request_canonical_plugin():
    engine = LayaTriageEngine(
        use_router=False,
        installed_plugins={"agentic-topic-researching-plugin", "agentic-coding-plugin"},
    )
    choice, conf, probs = engine.evaluate_system1_triage(
        "Please conduct deep research on state of the art retrieval models"
    )
    assert choice == engine.intent_to_response["research"]
    assert conf >= 0.90
    assert engine.response_to_intent[choice] == "research"


def test_build_dynamic_triage_spec_all_four():
    spec = build_dynamic_triage_spec({
        "agentic-coding-plugin",
        "agentic-investigate-plugin",
        "agentic-monitoring-plugin",
        "agentic-research-plugin",
    })
    assert len(spec.responses) == 6
    assert spec.intent_to_response["question"] == "response 1"
    assert spec.intent_to_response["coding"] == "response 2"
    assert spec.intent_to_response["monitoring"] == "response 3"
    assert spec.intent_to_response["investigation"] == "response 4"
    assert spec.intent_to_response["research"] == "response 5"
    assert spec.intent_to_response["mixed"] == "response 6"
    assert spec.response_to_intent["response 5"] == "research"
    assert spec.response_to_intent["response 6"] == "mixed"
    assert "@coding-research" in spec.response_to_directive["response 5"]


def test_build_dynamic_triage_spec_none():
    spec = build_dynamic_triage_spec(set())
    assert "response 1" in spec.responses
    assert len(spec.responses) == 1
    assert "mixed" not in spec.intent_to_response


def test_prompt_triage_hook_allow():
    async def _run():
        context = HookContext()
        context.set_state("invocation_num", 0)

        hook = PromptTriageHook(
            use_router=False,
            installed_plugins={"agentic-coding-plugin", "agentic-investigate-plugin", "agentic-monitoring-plugin"},
        )
        result = await hook.run(context, "How does this code work?")

        assert isinstance(result, HookResult)
        assert result.allow is True
        assert result.message in ("response 1", "question")
        assert context.get_state("triage_intent") == "question"
        assert context.get_state("response_choice") == "response 1"
        inject_steps = context.get_state("inject_steps")
        assert len(inject_steps) > 0
        assert "systemMessage" in inject_steps[0]
        assert "ephemeralMessage" not in inject_steps[0]
        sys_msg = inject_steps[0]["systemMessage"]["systemMessage"]
        assert "What is this prompt about ?" in sys_msg
        assert "Prompt Triage Directive" in sys_msg
        assert "Do not make any changes" in sys_msg

    asyncio.run(_run())


def test_session_start_hook():
    async def _run():
        context = HookContext()
        await session_start_triage(context)
        assert context.get_state("triage_initialized") is True
        assert context.get_state("invocation_num") == 0

    asyncio.run(_run())


def test_triage_response1_inquiries_no_changes():
    engine = LayaTriageEngine(
        use_router=False,
        installed_plugins={"agentic-coding-plugin", "agentic-investigate-plugin", "agentic-monitoring-plugin"},
    )
    result = engine.triage("Can you explain how this module works?", invocation_num=0)
    assert result.response_choice == "response 1"
    assert result.intent == "question"
    assert result.response_confidence >= CONFIDENCE_THRESHOLD
    assert len(result.inject_steps) > 0
    assert "systemMessage" in result.inject_steps[0]
    msg = result.inject_steps[0]["systemMessage"]["systemMessage"]
    assert "=== Prompt Triage Telemetry & Metadata ===" in msg
    assert "What is this prompt about ?" in msg
    assert "Do not make any changes" in msg


def test_triage_response_coding():
    engine = LayaTriageEngine(
        use_router=False,
        installed_plugins={"agentic-coding-plugin", "agentic-investigate-plugin", "agentic-monitoring-plugin"},
    )
    result = engine.triage("Please implement a new login endpoint with JWT auth", invocation_num=0)
    assert result.intent == "coding"
    assert result.response_confidence >= CONFIDENCE_THRESHOLD
    assert len(result.inject_steps) > 0
    assert "systemMessage" in result.inject_steps[0]
    msg = result.inject_steps[0]["systemMessage"]["systemMessage"]
    assert "=== Prompt Triage Telemetry & Metadata ===" in msg
    assert "invoke_subagent" in msg
    assert "coding" in msg
    assert "@coding" in msg


def test_triage_response_monitoring():
    engine = LayaTriageEngine(
        use_router=False,
        installed_plugins={"agentic-coding-plugin", "agentic-investigate-plugin", "agentic-monitoring-plugin"},
    )
    result = engine.triage("@coding-monitor watch the serial console /dev/ttyUSB0", invocation_num=0)
    assert result.intent == "monitoring"
    assert result.response_confidence >= CONFIDENCE_THRESHOLD
    assert len(result.inject_steps) > 0
    assert "systemMessage" in result.inject_steps[0]
    msg = result.inject_steps[0]["systemMessage"]["systemMessage"]
    assert "=== Prompt Triage Telemetry & Metadata ===" in msg
    assert "invoke_subagent" in msg
    assert "monitoring" in msg
    assert "@coding-monitor" in msg


def test_triage_response_investigation():
    engine = LayaTriageEngine(
        use_router=False,
        installed_plugins={"agentic-coding-plugin", "agentic-investigate-plugin", "agentic-monitoring-plugin"},
    )
    result = engine.triage("Please investigate why the checkout endpoint returns 500 error", invocation_num=0)
    assert result.intent == "investigation"
    assert result.response_confidence >= CONFIDENCE_THRESHOLD
    assert len(result.inject_steps) > 0
    assert "systemMessage" in result.inject_steps[0]
    msg = result.inject_steps[0]["systemMessage"]["systemMessage"]
    assert "=== Prompt Triage Telemetry & Metadata ===" in msg
    assert "invoke_subagent" in msg
    assert "investigate" in msg
    assert "@coding-investigate" in msg


def test_triage_response_research():
    engine = LayaTriageEngine(
        use_router=False,
        installed_plugins={
            "agentic-coding-plugin",
            "agentic-investigate-plugin",
            "agentic-monitoring-plugin",
            "agentic-research-plugin",
        },
    )
    result = engine.triage("Please conduct deep research on state of the art retrieval models", invocation_num=0)
    assert result.intent == "research"
    assert result.response_confidence >= CONFIDENCE_THRESHOLD
    assert len(result.inject_steps) > 0
    assert "systemMessage" in result.inject_steps[0]
    msg = result.inject_steps[0]["systemMessage"]["systemMessage"]
    assert "=== Prompt Triage Telemetry & Metadata ===" in msg
    assert "invoke_subagent" in msg
    assert "research" in msg
    assert "@coding-research" in msg
    assert "Topic Researcher" in msg


def test_triage_response_documentation():
    engine = LayaTriageEngine(
        use_router=False,
        installed_plugins={
            "agentic-coding-plugin",
            "agentic-obsidian-plugin",
        },
    )
    result = engine.triage("@coding-document document the project architecture in obsidian", invocation_num=0)
    assert result.intent == "documentation"
    assert result.response_confidence >= CONFIDENCE_THRESHOLD
    assert len(result.inject_steps) > 0
    assert "systemMessage" in result.inject_steps[0]
    msg = result.inject_steps[0]["systemMessage"]["systemMessage"]
    assert "=== Prompt Triage Telemetry & Metadata ===" in msg
    assert "invoke_subagent" in msg
    assert "documentation" in msg
    assert "@coding-document" in msg
    assert "Documentation Specialist" in msg


def test_all_dynamic_triage_directives_use_canonical_handles():
    spec = build_dynamic_triage_spec({
        "agentic-coding-plugin",
        "agentic-investigate-plugin",
        "agentic-monitoring-plugin",
        "agentic-research-plugin",
        "agentic-obsidian-plugin",
    })
    canonical = ["@coding", "@coding-investigate", "@coding-monitor", "@coding-research", "@coding-document"]
    legacy = ["@investigate", "@monitoring", "@document", "@research"]

    all_directives = " ".join(spec.response_to_directive.values())
    for handle in canonical:
        assert handle in all_directives, f"Canonical handle {handle} missing from directives"
    for legacy_handle in legacy:
        assert legacy_handle not in all_directives, f"Legacy handle {legacy_handle} should not be in directives"


def test_triage_response_mixed():
    engine = LayaTriageEngine(
        use_router=False,
        installed_plugins={"agentic-coding-plugin", "agentic-investigate-plugin", "agentic-monitoring-plugin"},
    )
    result = engine.triage(
        "Explain how the server starts and also implement a new fix and monitor the logs",
        invocation_num=0,
    )
    assert result.intent == "mixed"
    assert result.response_confidence >= CONFIDENCE_THRESHOLD
    assert len(result.inject_steps) > 0
    assert "systemMessage" in result.inject_steps[0]
    msg = result.inject_steps[0]["systemMessage"]["systemMessage"]
    assert "=== Prompt Triage Telemetry & Metadata ===" in msg
    assert "cannot process a mix" in msg


def test_triage_response_mixed_with_research():
    engine = LayaTriageEngine(
        use_router=False,
        installed_plugins={
            "agentic-coding-plugin",
            "agentic-investigate-plugin",
            "agentic-monitoring-plugin",
            "agentic-research-plugin",
        },
    )
    result = engine.triage(
        "Research quantum computing algorithms and also implement a prototype in Python",
        invocation_num=0,
    )
    assert result.intent == "mixed"
    assert result.response_confidence >= CONFIDENCE_THRESHOLD
    assert len(result.inject_steps) > 0
    assert "systemMessage" in result.inject_steps[0]
    msg = result.inject_steps[0]["systemMessage"]["systemMessage"]
    assert "=== Prompt Triage Telemetry & Metadata ===" in msg
    assert "cannot process a mix" in msg


def test_triage_response_mixed_research_and_inquiry():
    engine = LayaTriageEngine(
        use_router=False,
        installed_plugins={
            "agentic-coding-plugin",
            "agentic-investigate-plugin",
            "agentic-monitoring-plugin",
            "agentic-research-plugin",
        },
    )
    result = engine.triage(
        "Explain how the current system works and also conduct deep research on alternatives",
        invocation_num=0,
    )
    assert result.intent == "mixed"
    assert result.response_confidence >= CONFIDENCE_THRESHOLD
    assert len(result.inject_steps) > 0
    assert "systemMessage" in result.inject_steps[0]
    msg = result.inject_steps[0]["systemMessage"]["systemMessage"]
    assert "=== Prompt Triage Telemetry & Metadata ===" in msg
    assert "cannot process a mix" in msg


def test_triage_ambiguous_low_confidence_triggers_ask_question():
    engine = LayaTriageEngine(
        use_router=False,
        installed_plugins={"agentic-coding-plugin", "agentic-investigate-plugin", "agentic-monitoring-plugin"},
    )
    result = engine.triage("maybe something about project files", invocation_num=0)
    assert result.requires_user_confirmation is True
    assert result.response_confidence < CONFIDENCE_THRESHOLD
    assert len(result.inject_steps) > 0
    assert "systemMessage" in result.inject_steps[0]
    inject_str = str(result.inject_steps[0])
    assert "=== Prompt Triage Telemetry & Metadata ===" in inject_str
    assert "ask_question" in inject_str
    assert "What is this prompt about ?" in inject_str
    assert "one or more inquiries involving no changes" in inject_str
    assert "one or more code change request" in inject_str


def test_triage_hook_script_cli(tmp_path: Path):
    input_payload = {
        "conversationId": "test-conv-001",
        "invocationNum": 0,
        "lastUserInput": "<USER_REQUEST>Can you review this code?</USER_REQUEST><ADDITIONAL_METADATA>time</ADDITIONAL_METADATA>",
        "artifactDirectoryPath": str(tmp_path),
    }

    proc = subprocess.run(
        ["uv", "run", "hooks/prompt_triage.py"],
        cwd=str(PLUGIN_ROOT),
        input=json.dumps(input_payload),
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0
    output = json.loads(proc.stdout)
    assert "injectSteps" in output
    assert len(output["injectSteps"]) > 0
    step = output["injectSteps"][0]
    assert "systemMessage" in step
    assert "ephemeralMessage" not in step
    msg = step["systemMessage"]["systemMessage"]
    assert "What is this prompt about ?" in msg
    assert "Telemetry & Metadata" in msg


def test_triage_hook_skips_when_invocation_num_greater_than_zero(tmp_path: Path):
    input_payload = {
        "conversationId": "test-conv-001",
        "invocationNum": 1,
        "lastUserInput": "Can you review this code?",
        "artifactDirectoryPath": str(tmp_path),
    }

    proc = subprocess.run(
        ["uv", "run", "hooks/prompt_triage.py"],
        cwd=str(PLUGIN_ROOT),
        input=json.dumps(input_payload),
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0
    output = json.loads(proc.stdout)
    assert output == {"injectSteps": []}


def test_detect_installed_plugins_grooming():
    plugins = detect_installed_plugins(custom_plugins={"agentic-project-grooming-plugin"})
    assert plugins == {"agentic-project-grooming-plugin"}


def test_build_dynamic_triage_spec_only_grooming():
    spec = build_dynamic_triage_spec({"agentic-project-grooming-plugin"})
    assert "response 1" in spec.responses
    assert "response 2" in spec.responses
    assert spec.responses["response 2"] == "one or more project grooming or scaffolding requests"
    assert spec.response_to_intent["response 2"] == "grooming"
    assert spec.intent_to_response["grooming"] == "response 2"
    assert "coding" not in spec.intent_to_response
    assert "investigation" not in spec.intent_to_response
    assert "monitoring" not in spec.intent_to_response
    assert "research" not in spec.intent_to_response
    assert "documentation" not in spec.intent_to_response
    assert spec.intent_to_response.get("mixed") == "response 3"
    assert "/project-grooming <prompt>" in spec.response_to_directive["response 2"]
    assert "project grooming skill" in spec.response_to_directive["response 2"]


def test_build_dynamic_triage_spec_grooming_absent():
    spec = build_dynamic_triage_spec({"agentic-coding-plugin", "agentic-document-plugin"})
    assert "grooming" not in spec.intent_to_response
    assert not any("grooming" in desc for desc in spec.responses.values())
    mixed_directive = spec.response_to_directive.get(spec.intent_to_response.get("mixed", ""), "")
    assert "project grooming requests" not in mixed_directive


def test_build_dynamic_triage_spec_all_six_primary():
    spec = build_dynamic_triage_spec({
        "agentic-coding-plugin",
        "agentic-investigate-plugin",
        "agentic-monitor-plugin",
        "agentic-research-plugin",
        "agentic-document-plugin",
        "agentic-project-grooming-plugin",
    })
    assert len(spec.responses) == 8
    assert spec.intent_to_response["question"] == "response 1"
    assert spec.intent_to_response["coding"] == "response 2"
    assert spec.intent_to_response["monitoring"] == "response 3"
    assert spec.intent_to_response["investigation"] == "response 4"
    assert spec.intent_to_response["research"] == "response 5"
    assert spec.intent_to_response["documentation"] == "response 6"
    assert spec.intent_to_response["grooming"] == "response 7"
    assert spec.intent_to_response["mixed"] == "response 8"
    assert spec.response_to_intent["response 7"] == "grooming"
    assert spec.response_to_intent["response 8"] == "mixed"
    assert "/project-grooming" in spec.response_to_directive["response 7"]
    assert "project grooming requests" in spec.responses["response 8"]


def test_triage_response_grooming():
    engine = LayaTriageEngine(
        use_router=False,
        installed_plugins={"agentic-project-grooming-plugin", "agentic-coding-plugin"},
    )
    test_cases = [
        "Please groom the project repository",
        "Scaffold project structure for the new backend",
        "Initialize project workspace with proper templates",
        "Setup workspace for agentic plugins",
        "/project-grooming configure the directory layout",
        "Run project-grooming to standardize rules",
    ]
    for prompt in test_cases:
        result = engine.triage(prompt, invocation_num=0)
        assert result.intent == "grooming", f"Expected grooming for prompt: {prompt}"
        assert result.response_confidence >= CONFIDENCE_THRESHOLD
        assert len(result.inject_steps) > 0
        assert "systemMessage" in result.inject_steps[0]
        msg = result.inject_steps[0]["systemMessage"]["systemMessage"]
        assert "=== Prompt Triage Telemetry & Metadata ===" in msg
        assert "/project-grooming <prompt>" in msg
        assert "project grooming or scaffolding request" in msg


def test_triage_grooming_absent_fallback():
    engine = LayaTriageEngine(
        use_router=False,
        installed_plugins={"agentic-coding-plugin"},
    )
    assert "grooming" not in engine.intent_to_response
    assert "grooming" not in engine.response_to_intent.values()
    choice, conf, probs = engine.evaluate_system1_triage("Please groom the project repository")
    assert choice != "grooming"
    assert engine.response_to_intent.get(choice) != "grooming"


def test_triage_response_mixed_with_grooming():
    engine = LayaTriageEngine(
        use_router=False,
        installed_plugins={"agentic-coding-plugin", "agentic-project-grooming-plugin"},
    )
    result = engine.triage(
        "Groom the project repository and implement a new payment gateway",
        invocation_num=0,
    )
    assert result.intent == "mixed"
    assert result.response_confidence >= CONFIDENCE_THRESHOLD
    assert len(result.inject_steps) > 0
    assert "systemMessage" in result.inject_steps[0]
    msg = result.inject_steps[0]["systemMessage"]["systemMessage"]
    assert "=== Prompt Triage Telemetry & Metadata ===" in msg
    assert "cannot process a mix" in msg


def test_triage_response_mixed_grooming_and_inquiry():
    engine = LayaTriageEngine(
        use_router=False,
        installed_plugins={"agentic-project-grooming-plugin"},
    )
    result = engine.triage(
        "Explain how the directory structure works and also groom the project",
        invocation_num=0,
    )
    assert result.intent == "mixed"
    assert result.response_confidence >= CONFIDENCE_THRESHOLD
    assert len(result.inject_steps) > 0
    assert "systemMessage" in result.inject_steps[0]
    msg = result.inject_steps[0]["systemMessage"]["systemMessage"]
    assert "=== Prompt Triage Telemetry & Metadata ===" in msg
    assert "cannot process a mix" in msg

