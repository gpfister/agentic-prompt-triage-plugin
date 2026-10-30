"""Tests for agentic-triage-plugin structure and specifications."""

import json
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent


def test_plugin_manifest_valid():
    manifest_file = PLUGIN_ROOT / "plugin.json"
    assert manifest_file.exists(), "plugin.json must exist"
    data = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert data.get("name") == "agentic-triage-plugin"
    assert data.get("license") == "MIT"
    assert "version" in data


def test_license_exists():
    license_file = PLUGIN_ROOT / "LICENSE"
    assert license_file.exists(), "LICENSE must exist"
    content = license_file.read_text(encoding="utf-8")
    assert "MIT License" in content
    assert "Greg PFISTER" in content


def test_contributing_exists():
    contrib_file = PLUGIN_ROOT / "CONTRIBUTING.md"
    assert contrib_file.exists(), "CONTRIBUTING.md must exist"
    content = contrib_file.read_text(encoding="utf-8")
    assert "agentic-triage-plugin" in content


def test_hooks_json_valid():
    hooks_file = PLUGIN_ROOT / "hooks.json"
    assert hooks_file.exists(), "hooks.json must exist"
    data = json.loads(hooks_file.read_text(encoding="utf-8"))
    assert "prompt-triage" in data
    assert data["prompt-triage"].get("PreInvocation") is not None


def test_rules_exist():
    rules_file = PLUGIN_ROOT / "rules" / "AGENTS.md"
    assert rules_file.exists(), "rules/AGENTS.md must exist"
    content = rules_file.read_text(encoding="utf-8")
    assert "Prompt Triage Standards & Guidelines" in content


def test_hook_script_exists():
    hook_file = PLUGIN_ROOT / "hooks" / "prompt_triage.py"
    assert hook_file.exists(), "hooks/prompt_triage.py must exist"


def test_canonical_handles_in_documentation_and_rules():
    canonical = ["@coding", "@coding-investigate", "@coding-monitor", "@coding-research", "@coding-document"]
    legacy = ["@investigate", "@monitoring", "@document", "@research"]

    for doc_name in ["README.md", "rules/AGENTS.md", "CONTRIBUTING.md"]:
        doc_path = PLUGIN_ROOT / doc_name
        assert doc_path.exists(), f"{doc_name} must exist"
        text = doc_path.read_text(encoding="utf-8")
        for handle in canonical:
            assert handle in text, f"Canonical handle {handle} missing in {doc_name}"
        for legacy_handle in legacy:
            assert legacy_handle not in text, f"Legacy handle {legacy_handle} found in {doc_name}"


def test_hook_keywords_contain_no_legacy_aliases():
    hook_file = PLUGIN_ROOT / "hooks" / "prompt_triage.py"
    text = hook_file.read_text(encoding="utf-8")
    for legacy_handle in ['"@investigate"', '"@monitoring"', '"@document"', '"@research"']:
        assert legacy_handle not in text, f"Legacy handle {legacy_handle} should be removed from keyword tuples"

