# agentic-prompt-triage-plugin

Dynamic prompt intent triage lifecycle hook for [Google Antigravity](https://antigravity.google) powered by [Laya Router](https://github.com).

## Overview
`agentic-prompt-triage-plugin` automatically triages incoming user prompts before model execution using the non-autoregressive Laya Router System 1 model. It dynamically adapts its semantic `choice` criteria keys (`inquiry`, `coding`, `monitoring`, `investigation`, `research`, `documentation`, `grooming`, `design`) based on which plugins are installed:
- Evaluates code change requests (`coding`) if `agentic-coding-plugin` is installed (routes to `@coding`).
- Evaluates issue investigations (`investigation`) if `agentic-issue-investigation-plugin` (or `agentic-investigate-plugin`) is installed (routes to `@coding-investigate`).
- Evaluates task monitoring (`monitoring`) if `agentic-task-monitoring-plugin` (or `agentic-monitor-plugin`, `agentic-monitoring-plugin`) is installed (routes to `@coding-monitor`).
- Evaluates topic and web research (`research`) if `agentic-topic-researching-plugin` (or `agentic-research-plugin`) is installed (routes to `@coding-research`).
- Evaluates documentation requests (`documentation`) if `agentic-project-documenting-plugin` (or `agentic-document-plugin`, `agentic-obsidian-plugin`) is installed (routes to `@coding-document`).
- Evaluates project grooming requests (`grooming`) if `agentic-project-grooming-plugin` is installed (routes to `/project-grooming <prompt>`).
- Evaluates UI/UX design requests (`design`) if `agentic-design-plugin` is installed (routes to `@coding-design`).
- Handles informational inquiries (`inquiry`) directly without code modifications.
- Rejects conflicting mixed prompts (`is_mixed`), asking users to separate their requests.

### Triage Architecture
- **Single-Pass Parallel `choice` + `noul` Decomposition**: Submits `"about"` (`choice` over active semantic keys anchored to `` `prompt` ``), `"requires_changes"` (`noul` mutating-intent gate), and `"is_mixed"` (`noul` multi-workflow detector) in a single `Router.predict` forward pass.
- **Calibrated Confidence & Usage Diagnostics**: Prefers Laya's calibrated `confidence` field (`>= 0.80` threshold), gates mutating agents vs. `inquiry` using `requires_changes` (exempting read-only `monitoring`, `investigation`, and `research`), and inspects `usage` (`truncated` / `options`) for anomalies.

## Installation & Activation

```bash
# Enable in Antigravity session:
agy plugin enable agentic-prompt-triage-plugin
```

## Contributing
See [CONTRIBUTING.md](file:///Users/gpfister/Development/Projects/gpfister/agentic/agentic-prompt-triage-plugin/CONTRIBUTING.md) for contribution guidelines and testing practices.

## License
This project is licensed under the [MIT License](file:///Users/gpfister/Development/Projects/gpfister/agentic/agentic-prompt-triage-plugin/LICENSE).
