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
    assert "inquiry" in spec.responses
    assert "investigation" in spec.responses
    assert "mixed" not in spec.responses
    assert spec.response_to_intent["investigation"] == "investigation"
    assert spec.intent_to_response["investigation"] == "investigation"
    assert "@coding-investigate" in spec.response_to_directive["investigation"]


def test_build_dynamic_triage_spec_canonical_monitoring():
    spec = build_dynamic_triage_spec({"agentic-task-monitoring-plugin"})
    assert "inquiry" in spec.responses
    assert "monitoring" in spec.responses
    assert "mixed" not in spec.responses
    assert spec.response_to_intent["monitoring"] == "monitoring"
    assert spec.intent_to_response["monitoring"] == "monitoring"
    assert "@coding-monitor" in spec.response_to_directive["monitoring"]


def test_build_dynamic_triage_spec_canonical_research():
    spec = build_dynamic_triage_spec({"agentic-topic-researching-plugin"})
    assert "inquiry" in spec.responses
    assert "research" in spec.responses
    assert "mixed" not in spec.responses
    assert spec.response_to_intent["research"] == "research"
    assert spec.intent_to_response["research"] == "research"
    assert "@coding-research" in spec.response_to_directive["research"]


def test_build_dynamic_triage_spec_canonical_documenting():
    spec = build_dynamic_triage_spec({"agentic-project-documenting-plugin"})
    assert "inquiry" in spec.responses
    assert "documentation" in spec.responses
    assert "mixed" not in spec.responses
    assert spec.response_to_intent["documentation"] == "documentation"
    assert spec.intent_to_response["documentation"] == "documentation"
    assert "@coding-document" in spec.response_to_directive["documentation"]


def test_build_dynamic_triage_spec_all_six_canonical():
    spec = build_dynamic_triage_spec({
        "agentic-coding-plugin",
        "agentic-issue-investigation-plugin",
        "agentic-task-monitoring-plugin",
        "agentic-topic-researching-plugin",
        "agentic-project-documenting-plugin",
        "agentic-project-grooming-plugin",
    })
    assert len(spec.responses) == 7
    assert "mixed" not in spec.responses
    assert spec.intent_to_response["question"] == "inquiry"
    assert spec.intent_to_response["coding"] == "coding"
    assert spec.intent_to_response["monitoring"] == "monitoring"
    assert spec.intent_to_response["investigation"] == "investigation"
    assert spec.intent_to_response["research"] == "research"
    assert spec.intent_to_response["documentation"] == "documentation"
    assert spec.intent_to_response["grooming"] == "grooming"
    assert spec.intent_to_response["mixed"] == "mixed"
    assert spec.response_to_intent["coding"] == "coding"
    assert spec.response_to_intent["monitoring"] == "monitoring"
    assert spec.response_to_intent["investigation"] == "investigation"
    assert spec.response_to_intent["research"] == "research"
    assert spec.response_to_intent["documentation"] == "documentation"
    assert spec.response_to_intent["grooming"] == "grooming"
    assert spec.response_to_intent["mixed"] == "mixed"


def test_build_dynamic_triage_spec_all():
    spec = build_dynamic_triage_spec(
        {"agentic-coding-plugin", "agentic-investigate-plugin", "agentic-monitoring-plugin"}
    )
    assert "inquiry" in spec.responses
    assert "coding" in spec.responses
    assert "mixed" not in spec.responses
    assert spec.response_to_intent["coding"] == "coding"
    assert "coding" in spec.intent_to_response
    assert "investigation" in spec.intent_to_response
    assert "monitoring" in spec.intent_to_response
    mixed_key = spec.intent_to_response.get("mixed")
    assert mixed_key == "mixed"
    assert spec.response_to_intent[mixed_key] == "mixed"


def test_build_dynamic_triage_spec_only_coding():
    spec = build_dynamic_triage_spec({"agentic-coding-plugin"})
    assert "inquiry" in spec.responses
    assert "coding" in spec.responses
    assert "mixed" not in spec.responses
    assert "investigation" not in spec.intent_to_response
    assert "monitoring" not in spec.intent_to_response
    assert spec.response_to_intent["coding"] == "coding"
    assert spec.intent_to_response.get("mixed") == "mixed"


def test_build_dynamic_triage_spec_only_investigate():
    spec = build_dynamic_triage_spec({"agentic-investigate-plugin"})
    assert "inquiry" in spec.responses
    assert "investigation" in spec.responses
    assert "mixed" not in spec.responses
    assert spec.response_to_intent["investigation"] == "investigation"
    assert "coding" not in spec.intent_to_response
    assert spec.intent_to_response.get("mixed") == "mixed"


def test_build_dynamic_triage_spec_only_research():
    spec = build_dynamic_triage_spec({"agentic-research-plugin"})
    assert "inquiry" in spec.responses
    assert "research" in spec.responses
    assert "mixed" not in spec.responses
    assert "research" in spec.responses["research"].lower()
    assert spec.response_to_intent["research"] == "research"
    assert spec.intent_to_response["research"] == "research"
    assert "coding" not in spec.intent_to_response
    assert "investigation" not in spec.intent_to_response
    assert "monitoring" not in spec.intent_to_response
    assert spec.intent_to_response.get("mixed") == "mixed"
    assert "@coding-research" in spec.response_to_directive["research"]
    assert "Topic Researcher" in spec.response_to_directive["research"]


def test_build_dynamic_triage_spec_only_obsidian():
    spec = build_dynamic_triage_spec({"agentic-obsidian-plugin"})
    assert "inquiry" in spec.responses
    assert "documentation" in spec.responses
    assert "mixed" not in spec.responses
    assert "obsidian" in spec.responses["documentation"].lower()
    assert spec.response_to_intent["documentation"] == "documentation"
    assert spec.intent_to_response["documentation"] == "documentation"
    assert "coding" not in spec.intent_to_response
    assert "investigation" not in spec.intent_to_response
    assert "monitoring" not in spec.intent_to_response
    assert "research" not in spec.intent_to_response
    assert spec.intent_to_response.get("mixed") == "mixed"
    assert "@coding-document" in spec.response_to_directive["documentation"]
    assert "Documentation Specialist" in spec.response_to_directive["documentation"]


def test_build_dynamic_triage_spec_only_document():
    spec = build_dynamic_triage_spec({"agentic-document-plugin"})
    assert "inquiry" in spec.responses
    assert "documentation" in spec.responses
    assert "mixed" not in spec.responses
    assert spec.response_to_intent["documentation"] == "documentation"
    assert spec.intent_to_response["documentation"] == "documentation"
    assert "coding" not in spec.intent_to_response
    assert "investigation" not in spec.intent_to_response
    assert "monitoring" not in spec.intent_to_response
    assert "research" not in spec.intent_to_response
    assert spec.intent_to_response.get("mixed") == "mixed"
    assert "@coding-document" in spec.response_to_directive["documentation"]
    assert "Documentation Specialist" in spec.response_to_directive["documentation"]


def test_build_dynamic_triage_spec_only_monitor():
    spec = build_dynamic_triage_spec({"agentic-monitor-plugin"})
    assert "inquiry" in spec.responses
    assert "monitoring" in spec.responses
    assert "mixed" not in spec.responses
    assert "monitor" in spec.responses["monitoring"].lower()
    assert spec.response_to_intent["monitoring"] == "monitoring"
    assert spec.intent_to_response["monitoring"] == "monitoring"
    assert "coding" not in spec.intent_to_response
    assert "investigation" not in spec.intent_to_response
    assert "documentation" not in spec.intent_to_response
    assert "research" not in spec.intent_to_response
    assert spec.intent_to_response.get("mixed") == "mixed"
    assert "@coding-monitor" in spec.response_to_directive["monitoring"]
    assert "Telemetry Monitor" in spec.response_to_directive["monitoring"]


def test_build_dynamic_triage_spec_all_five_primary():
    spec = build_dynamic_triage_spec({
        "agentic-coding-plugin",
        "agentic-investigate-plugin",
        "agentic-monitor-plugin",
        "agentic-research-plugin",
        "agentic-document-plugin",
    })
    assert len(spec.responses) == 6
    assert "mixed" not in spec.responses
    assert spec.intent_to_response["question"] == "inquiry"
    assert spec.intent_to_response["coding"] == "coding"
    assert spec.intent_to_response["monitoring"] == "monitoring"
    assert spec.intent_to_response["investigation"] == "investigation"
    assert spec.intent_to_response["research"] == "research"
    assert spec.intent_to_response["documentation"] == "documentation"
    assert spec.intent_to_response["mixed"] == "mixed"
    assert spec.response_to_intent["monitoring"] == "monitoring"
    assert spec.response_to_intent["documentation"] == "documentation"
    assert spec.response_to_intent["mixed"] == "mixed"
    assert "@coding-monitor" in spec.response_to_directive["monitoring"]
    assert "@coding-document" in spec.response_to_directive["documentation"]


def test_build_dynamic_triage_spec_all_five():
    spec = build_dynamic_triage_spec({
        "agentic-coding-plugin",
        "agentic-investigate-plugin",
        "agentic-monitoring-plugin",
        "agentic-research-plugin",
        "agentic-obsidian-plugin",
    })
    assert len(spec.responses) == 6
    assert "mixed" not in spec.responses
    assert spec.intent_to_response["question"] == "inquiry"
    assert spec.intent_to_response["coding"] == "coding"
    assert spec.intent_to_response["monitoring"] == "monitoring"
    assert spec.intent_to_response["investigation"] == "investigation"
    assert spec.intent_to_response["research"] == "research"
    assert spec.intent_to_response["documentation"] == "documentation"
    assert spec.intent_to_response["mixed"] == "mixed"
    assert spec.response_to_intent["documentation"] == "documentation"
    assert spec.response_to_intent["mixed"] == "mixed"
    assert "@coding-document" in spec.response_to_directive["documentation"]


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
    assert len(spec.responses) == 5
    assert "mixed" not in spec.responses
    assert spec.intent_to_response["question"] == "inquiry"
    assert spec.intent_to_response["coding"] == "coding"
    assert spec.intent_to_response["monitoring"] == "monitoring"
    assert spec.intent_to_response["investigation"] == "investigation"
    assert spec.intent_to_response["research"] == "research"
    assert spec.intent_to_response["mixed"] == "mixed"
    assert spec.response_to_intent["research"] == "research"
    assert spec.response_to_intent["mixed"] == "mixed"
    assert "@coding-research" in spec.response_to_directive["research"]


def test_build_dynamic_triage_spec_none():
    spec = build_dynamic_triage_spec(set())
    assert "inquiry" in spec.responses
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
        assert result.message in ("inquiry", "question")
        assert context.get_state("triage_intent") == "question"
        assert context.get_state("response_choice") == "inquiry"
        inject_steps = context.get_state("inject_steps")
        assert len(inject_steps) > 0
        assert "systemMessage" in inject_steps[0]
        assert "ephemeralMessage" not in inject_steps[0]
        sys_msg = inject_steps[0]["systemMessage"]["systemMessage"]
        assert TRIAGE_QUESTION in sys_msg
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
    assert result.response_choice == "inquiry"
    assert result.intent == "question"
    assert result.response_confidence >= CONFIDENCE_THRESHOLD
    assert len(result.inject_steps) > 0
    assert "systemMessage" in result.inject_steps[0]
    msg = result.inject_steps[0]["systemMessage"]["systemMessage"]
    assert "=== Prompt Triage Telemetry & Metadata ===" in msg
    assert TRIAGE_QUESTION in msg
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
    assert TRIAGE_QUESTION in inject_str
    assert engine.triage_responses["inquiry"] in inject_str
    assert engine.triage_responses["coding"] in inject_str


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
    assert TRIAGE_QUESTION in msg
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
    assert "inquiry" in spec.responses
    assert "grooming" in spec.responses
    assert "mixed" not in spec.responses
    assert "groom" in spec.responses["grooming"].lower()
    assert spec.response_to_intent["grooming"] == "grooming"
    assert spec.intent_to_response["grooming"] == "grooming"
    assert "coding" not in spec.intent_to_response
    assert "investigation" not in spec.intent_to_response
    assert "monitoring" not in spec.intent_to_response
    assert "research" not in spec.intent_to_response
    assert "documentation" not in spec.intent_to_response
    assert spec.intent_to_response.get("mixed") == "mixed"
    assert "/project-grooming <prompt>" in spec.response_to_directive["grooming"]
    assert "project grooming skill" in spec.response_to_directive["grooming"]


def test_build_dynamic_triage_spec_grooming_absent():
    spec = build_dynamic_triage_spec({"agentic-coding-plugin", "agentic-document-plugin"})
    assert "grooming" not in spec.intent_to_response
    assert not any("groom" in desc.lower() for desc in spec.responses.values())


def test_build_dynamic_triage_spec_all_six_primary():
    spec = build_dynamic_triage_spec({
        "agentic-coding-plugin",
        "agentic-investigate-plugin",
        "agentic-monitor-plugin",
        "agentic-research-plugin",
        "agentic-document-plugin",
        "agentic-project-grooming-plugin",
    })
    assert len(spec.responses) == 7
    assert "mixed" not in spec.responses
    assert spec.intent_to_response["question"] == "inquiry"
    assert spec.intent_to_response["coding"] == "coding"
    assert spec.intent_to_response["monitoring"] == "monitoring"
    assert spec.intent_to_response["investigation"] == "investigation"
    assert spec.intent_to_response["research"] == "research"
    assert spec.intent_to_response["documentation"] == "documentation"
    assert spec.intent_to_response["grooming"] == "grooming"
    assert spec.intent_to_response["mixed"] == "mixed"
    assert spec.response_to_intent["grooming"] == "grooming"
    assert spec.response_to_intent["mixed"] == "mixed"
    assert "/project-grooming" in spec.response_to_directive["grooming"]


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


def test_confidence_threshold_is_0_80():
    assert CONFIDENCE_THRESHOLD == 0.80


def test_triage_low_confidence_sorted_with_percentages_and_recommended(monkeypatch):
    engine = LayaTriageEngine(
        use_router=False,
        installed_plugins={"agentic-coding-plugin", "agentic-issue-investigation-plugin"},
    )
    probs = {
        "inquiry": 0.144,        # 14%
        "coding": 0.556,         # 56% -> highest, should be first and (Recommended)
        "investigation": 0.300,  # 30% -> second
    }
    monkeypatch.setattr(
        engine,
        "evaluate_system1_triage",
        lambda prompt: ("coding", 0.556, probs),
    )

    result = engine.triage("ambiguous prompt", invocation_num=0)
    assert result.requires_user_confirmation is True
    msg = result.inject_steps[0]["systemMessage"]["systemMessage"]

    expected_options = [
        f'  1. "(Recommended) {engine.triage_responses["coding"]} (56%)"',
        f'  2. "{engine.triage_responses["investigation"]} (30%)"',
        f'  3. "{engine.triage_responses["inquiry"]} (14%)"',
    ]
    options_block = "\n".join(expected_options)
    assert f"- Options:\n{options_block}" in msg


def test_triage_boundary_below_and_at_0_80(monkeypatch):
    engine = LayaTriageEngine(
        use_router=False,
        installed_plugins={"agentic-coding-plugin"},
    )
    probs_below = {"inquiry": 0.21, "coding": 0.79}
    monkeypatch.setattr(
        engine,
        "evaluate_system1_triage",
        lambda prompt: ("coding", 0.79, probs_below),
    )
    res_below = engine.triage("slightly uncertain", invocation_num=0)
    assert res_below.requires_user_confirmation is True
    assert "ask_question" in res_below.inject_steps[0]["systemMessage"]["systemMessage"]

    probs_at = {"inquiry": 0.20, "coding": 0.80}
    monkeypatch.setattr(
        engine,
        "evaluate_system1_triage",
        lambda prompt: ("coding", 0.80, probs_at),
    )
    res_at = engine.triage("confident enough", invocation_num=0)
    assert res_at.requires_user_confirmation is False
    assert "ask_question" not in res_at.inject_steps[0]["systemMessage"]["systemMessage"]


def test_detect_installed_plugins_design():
    assert "agentic-design-plugin" in PRIMARY_PLUGINS
    assert "agentic-design-plugin" in KNOWN_PLUGINS
    plugins = detect_installed_plugins(custom_plugins={"agentic-design-plugin"})
    assert plugins == {"agentic-design-plugin"}


def test_detect_installed_plugins_design_workspace_directory(tmp_path: Path):
    ws = tmp_path / "design_ws"
    ws.mkdir()
    pdir = ws / "agentic-design-plugin"
    pdir.mkdir()
    (pdir / "plugin.json").write_text(
        json.dumps({"name": "agentic-design-plugin", "version": "0.1.0"}),
        encoding="utf-8",
    )
    detected = detect_installed_plugins(workspace_paths=[str(ws)])
    assert "agentic-design-plugin" in detected


def test_build_dynamic_triage_spec_only_design():
    spec = build_dynamic_triage_spec({"agentic-design-plugin"})
    assert "inquiry" in spec.responses
    assert "design" in spec.responses
    assert "mixed" not in spec.responses
    assert "penpot" in spec.responses["design"].lower()
    assert spec.response_to_intent["design"] == "design"
    assert spec.intent_to_response["design"] == "design"
    assert "coding" not in spec.intent_to_response
    assert "investigation" not in spec.intent_to_response
    assert "monitoring" not in spec.intent_to_response
    assert "research" not in spec.intent_to_response
    assert "documentation" not in spec.intent_to_response
    assert "grooming" not in spec.intent_to_response
    assert spec.intent_to_response.get("mixed") == "mixed"
    assert "invoke_subagent(TypeName='coding-design', Role='UI/UX Designer', Prompt=...)" in spec.response_to_directive["design"]
    assert "@coding-design" in spec.response_to_directive["design"]


def test_build_dynamic_triage_spec_design_absent():
    spec = build_dynamic_triage_spec({"agentic-coding-plugin", "agentic-document-plugin"})
    assert "design" not in spec.intent_to_response
    assert not any("PenPot" in desc or "UI/UX" in desc for desc in spec.responses.values())
    assert not any("coding-design" in d for d in spec.response_to_directive.values())
    mixed_directive = spec.response_to_directive.get(spec.intent_to_response.get("mixed", ""), "")
    assert "UI/UX design requests" not in mixed_directive


def test_triage_response_design():
    engine = LayaTriageEngine(
        use_router=False,
        installed_plugins={"agentic-design-plugin", "agentic-coding-plugin"},
    )
    test_cases = [
        "Create a UI/UX wireframe for the login page",
        "Design a dashboard mockup in PenPot",
        "Extract design tokens from the PenPot file",
        "Update the design-token palette for dark mode",
        "Ask @coding-design to review the user interface prototype",
    ]
    for prompt in test_cases:
        result = engine.triage(prompt, invocation_num=0)
        assert result.intent == "design", f"Expected design for prompt: {prompt}"
        assert result.response_confidence >= CONFIDENCE_THRESHOLD
        assert len(result.inject_steps) > 0
        assert "systemMessage" in result.inject_steps[0]
        msg = result.inject_steps[0]["systemMessage"]["systemMessage"]
        assert "=== Prompt Triage Telemetry & Metadata ===" in msg
        assert "invoke_subagent(TypeName='coding-design', Role='UI/UX Designer', Prompt=...)" in msg
        assert "@coding-design" in msg


def test_triage_design_absent_fallback():
    engine = LayaTriageEngine(
        use_router=False,
        installed_plugins={"agentic-coding-plugin"},
    )
    assert "design" not in engine.intent_to_response
    assert "design" not in engine.response_to_intent.values()
    choice, conf, probs = engine.evaluate_system1_triage("Create a PenPot wireframe for the login screen")
    assert engine.response_to_intent.get(choice) != "design"


def test_triage_response_mixed_with_design():
    engine = LayaTriageEngine(
        use_router=False,
        installed_plugins={"agentic-coding-plugin", "agentic-design-plugin"},
    )
    result = engine.triage(
        "Create a PenPot wireframe and implement the backend authentication API",
        invocation_num=0,
    )
    assert result.intent == "mixed"
    assert result.response_confidence >= CONFIDENCE_THRESHOLD
    assert len(result.inject_steps) > 0
    msg = result.inject_steps[0]["systemMessage"]["systemMessage"]
    assert "cannot process a mix" in msg




class _MockRouter:
    """Mock Laya Router capturing predict calls and returning canned responses."""

    def __init__(self, response: dict):
        self.response = response
        self.calls: list[tuple[dict, dict]] = []

    def predict(self, state: dict, questions: dict, **kwargs) -> dict:
        self.calls.append((state, questions))
        return self.response


def test_triage_question_anchored_to_prompt():
    assert TRIAGE_QUESTION == "Which single agent or action should handle `prompt`?"
    assert "`prompt`" in TRIAGE_QUESTION


def test_parallel_choice_and_noul_single_predict_call():
    mock_router = _MockRouter(
        {
            "answers": {
                "about": {
                    "type": "choice",
                    "choice": "coding",
                    "probabilities": {"inquiry": 0.05, "coding": 0.95},
                    "confidence": 0.92,
                    "answer_confidence": 0.95,
                },
                "requires_changes": {
                    "type": "noul",
                    "noul": 0.91,
                    "confidence": 0.91,
                },
                "is_mixed": {
                    "type": "noul",
                    "noul": 0.08,
                    "confidence": 0.92,
                },
            },
            "usage": {"truncated": False, "truncated_questions": []},
        }
    )
    engine = LayaTriageEngine(
        use_router=False,
        installed_plugins={"agentic-coding-plugin"},
    )
    engine._router = mock_router

    result = engine.triage("Implement a new CLI flag", invocation_num=0)
    assert len(mock_router.calls) == 1
    state, questions = mock_router.calls[0]
    assert state == {"prompt": "Implement a new CLI flag"}
    assert set(questions.keys()) == {"about", "requires_changes", "is_mixed"}
    assert questions["about"]["type"] == "choice"
    assert questions["requires_changes"]["type"] == "noul"
    assert questions["is_mixed"]["type"] == "noul"
    assert result.intent == "coding"
    assert result.response_choice == "coding"
    assert result.response_confidence == 0.92
    assert result.requires_user_confirmation is False


def test_parallel_is_mixed_triggers_mixed_directive():
    mock_router = _MockRouter(
        {
            "answers": {
                "about": {
                    "type": "choice",
                    "choice": "coding",
                    "probabilities": {"inquiry": 0.35, "coding": 0.65},
                    "confidence": 0.60,
                },
                "requires_changes": {"type": "noul", "noul": 0.85, "confidence": 0.85},
                "is_mixed": {"type": "noul", "noul": 0.89, "confidence": 0.89},
            },
            "usage": {"truncated": False},
        }
    )
    engine = LayaTriageEngine(
        use_router=False,
        installed_plugins={"agentic-coding-plugin"},
    )
    engine._router = mock_router

    result = engine.triage("Explain auth and implement rate limiting", invocation_num=0)
    assert result.response_choice == "mixed"
    assert result.intent == "mixed"
    assert result.requires_user_confirmation is False
    msg = result.inject_steps[0]["systemMessage"]["systemMessage"]
    assert "cannot process a mix" in msg


def test_requires_changes_yes_gates_inquiry_to_disambiguation():
    mock_router = _MockRouter(
        {
            "answers": {
                "about": {
                    "type": "choice",
                    "choice": "inquiry",
                    "probabilities": {"inquiry": 0.88, "coding": 0.12},
                    "confidence": 0.88,
                },
                "requires_changes": {"type": "noul", "noul": 0.90, "confidence": 0.90},
                "is_mixed": {"type": "noul", "noul": 0.10, "confidence": 0.90},
            },
            "usage": {"truncated": False},
        }
    )
    engine = LayaTriageEngine(
        use_router=False,
        installed_plugins={"agentic-coding-plugin"},
    )
    engine._router = mock_router

    result = engine.triage("Can you update the config file?", invocation_num=0)
    assert result.response_confidence < CONFIDENCE_THRESHOLD
    assert result.requires_user_confirmation is True
    assert "ask_question" in result.inject_steps[0]["systemMessage"]["systemMessage"]


@pytest.mark.parametrize("mutating_agent,plugin_name", [
    ("coding", "agentic-coding-plugin"),
    ("design", "agentic-design-plugin"),
    ("documentation", "agentic-project-documenting-plugin"),
    ("grooming", "agentic-project-grooming-plugin"),
])
def test_requires_changes_no_gates_mutating_agents(mutating_agent: str, plugin_name: str):
    mock_router = _MockRouter(
        {
            "answers": {
                "about": {
                    "type": "choice",
                    "choice": mutating_agent,
                    "probabilities": {"inquiry": 0.10, mutating_agent: 0.90},
                    "confidence": 0.90,
                },
                "requires_changes": {"type": "noul", "noul": 0.12, "confidence": 0.88},
                "is_mixed": {"type": "noul", "noul": 0.05, "confidence": 0.95},
            },
            "usage": {"truncated": False},
        }
    )
    engine = LayaTriageEngine(use_router=False, installed_plugins={plugin_name})
    engine._router = mock_router

    result = engine.triage("What does this component do?", invocation_num=0)
    assert result.response_confidence < CONFIDENCE_THRESHOLD
    assert result.requires_user_confirmation is True


@pytest.mark.parametrize("non_mutating_agent,plugin_name", [
    ("monitoring", "agentic-task-monitoring-plugin"),
    ("investigation", "agentic-issue-investigation-plugin"),
    ("research", "agentic-topic-researching-plugin"),
])
def test_requires_changes_no_does_not_penalize_non_mutating_agents(non_mutating_agent: str, plugin_name: str):
    mock_router = _MockRouter(
        {
            "answers": {
                "about": {
                    "type": "choice",
                    "choice": non_mutating_agent,
                    "probabilities": {"inquiry": 0.08, non_mutating_agent: 0.92},
                    "confidence": 0.91,
                },
                "requires_changes": {"type": "noul", "noul": 0.10, "confidence": 0.90},
                "is_mixed": {"type": "noul", "noul": 0.05, "confidence": 0.95},
            },
            "usage": {"truncated": False},
        }
    )
    engine = LayaTriageEngine(use_router=False, installed_plugins={plugin_name})
    engine._router = mock_router

    result = engine.triage("Read-only operational or research request", invocation_num=0)
    assert result.intent == non_mutating_agent
    assert result.response_confidence == 0.91
    assert result.requires_user_confirmation is False


def test_calibrated_confidence_preferred_with_fallback_when_none():
    engine = LayaTriageEngine(use_router=False, installed_plugins={"agentic-coding-plugin"})

    # 1. Calibrated confidence (0.72) is preferred over probabilities["coding"] (0.94)
    engine._router = _MockRouter(
        {
            "answers": {
                "about": {
                    "type": "choice",
                    "choice": "coding",
                    "probabilities": {"inquiry": 0.06, "coding": 0.94},
                    "confidence": 0.72,
                    "answer_confidence": 0.94,
                },
                "requires_changes": {"type": "noul", "noul": 0.88, "confidence": 0.88},
            },
        }
    )
    choice, conf, _ = engine.evaluate_system1_triage("Implement feature")
    assert choice == "coding"
    assert conf == 0.72

    # 2. Fallback to probs[choice] when confidence is None
    engine._router = _MockRouter(
        {
            "answers": {
                "about": {
                    "type": "choice",
                    "choice": "coding",
                    "probabilities": {"inquiry": 0.06, "coding": 0.94},
                    "confidence": None,
                    "answer_confidence": 0.94,
                },
            },
        }
    )
    choice, conf, _ = engine.evaluate_system1_triage("Implement feature")
    assert choice == "coding"
    assert conf == 0.94


def test_usage_inspection_logs_warnings_on_truncation_and_collision(caplog):
    engine = LayaTriageEngine(use_router=False, installed_plugins={"agentic-coding-plugin"})
    engine._router = _MockRouter(
        {
            "answers": {
                "about": {
                    "type": "choice",
                    "choice": "coding",
                    "probabilities": {"inquiry": 0.1, "coding": 0.9},
                    "confidence": 0.9,
                },
            },
            "usage": {
                "truncated": True,
                "truncated_questions": ["about"],
                "options": ["inquiry"],
            },
        }
    )
    with caplog.at_level("WARNING"):
        engine.evaluate_system1_triage("Long prompt")
    assert "truncated" in caplog.text.lower()
    assert "mismatch/collision" in caplog.text.lower()


def test_single_plugin_inquiry_only_omits_is_mixed():
    mock_router = _MockRouter(
        {
            "answers": {
                "about": {
                    "type": "choice",
                    "choice": "inquiry",
                    "probabilities": {"inquiry": 1.0},
                    "confidence": 0.95,
                },
                "requires_changes": {"type": "noul", "noul": 0.10, "confidence": 0.90},
            },
        }
    )
    engine = LayaTriageEngine(use_router=False, installed_plugins=set())
    engine._router = mock_router

    result = engine.triage("What is Python?", invocation_num=0)
    _, questions = mock_router.calls[0]
    assert "is_mixed" not in questions
    assert result.intent == "question"
    assert result.requires_user_confirmation is False
