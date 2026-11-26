# Prompt Triage Standards & Guidelines — agentic-prompt-triage-plugin

This rule set applies whenever the `agentic-prompt-triage-plugin` is active in an Antigravity session.

---

## 1. Triage Architecture & Principles
- **Fast Non-Autoregressive Classification**: Prompt triage uses the Laya Router System 1 model (`Router(preload=False)` default) to classify user intent within a single fast forward pass using parallel `choice` (`about` over semantic criteria keys) + `noul` (`requires_changes`, `is_mixed`) decomposition, calibrated `confidence` (`>= 0.80`), and `usage` (`truncated` / `options`) diagnostics.
- **Dynamic Plugin-Aware Routing**:
  - Evaluate code change requests (`coding`) only if `agentic-coding-plugin` is installed (routes to `@coding`).
  - Evaluate issue investigation requests (`investigation`) only if `agentic-issue-investigation-plugin` (or `agentic-investigate-plugin`) is installed (routes to `@coding-investigate`).
  - Evaluate task monitoring requests (`monitoring`) only if `agentic-task-monitoring-plugin` (or `agentic-monitor-plugin`, `agentic-monitoring-plugin`) is installed (routes to `@coding-monitor`).
  - Evaluate topic research requests (`research`) only if `agentic-topic-researching-plugin` (or `agentic-research-plugin`) is installed (routes to `@coding-research`).
  - Evaluate documentation requests (`documentation`) only if `agentic-project-documenting-plugin` (or `agentic-document-plugin`, `agentic-obsidian-plugin`) is installed (routes to `@coding-document`).
  - Evaluate project grooming requests (`grooming`) only if `agentic-project-grooming-plugin` is installed (routes to `/project-grooming <prompt>`).
  - Evaluate UI/UX design requests (`design`) only if `agentic-design-plugin` is installed (routes to `@coding-design`).
  - Informational inquiries (`inquiry`) involving no changes are always supported.
  - Mixed prompts (`is_mixed`) combining multiple workflows are rejected with guidance to separate requests.
  - If a requested capability is not supported by installed plugins, do not evaluate or include that response; respond directly with guidance.

---

## 2. Hook Execution Safety
- **Single-Fire Guard**: Only triage the user's initial prompt (`invocationNum == 0`). Do not re-fire during multi-step tool iterations (`invocationNum > 0`).
- **Sanitization**: Strip XML wrappers, `<USER_REQUEST>`, `<ADDITIONAL_METADATA>`, and trailing file attachments before classifying intent.
- **Fail-Safe Operation**: If Laya Router is unavailable, fall back seamlessly to deterministic heuristic intent evaluation.
