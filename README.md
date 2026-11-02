# agentic-prompt-triage-plugin

Dynamic prompt intent triage lifecycle hook for [Google Antigravity](https://antigravity.google) powered by [Laya Router](https://github.com).

## Overview
`agentic-prompt-triage-plugin` automatically triages incoming user prompts before model execution using the non-autoregressive Laya Router System 1 model. It dynamically adapts its intent criteria based on which plugins are installed:
- Evaluates code change requests if `agentic-coding-plugin` is installed (routes to `@coding`).
- Evaluates issue investigations if `agentic-issue-investigation-plugin` (or `agentic-investigate-plugin`) is installed (routes to `@coding-investigate`).
- Evaluates task monitoring if `agentic-task-monitoring-plugin` (or `agentic-monitor-plugin`, `agentic-monitoring-plugin`) is installed (routes to `@coding-monitor`).
- Evaluates topic and web research if `agentic-topic-researching-plugin` (or `agentic-research-plugin`) is installed (routes to `@coding-research`).
- Evaluates documentation requests if `agentic-project-documenting-plugin` (or `agentic-document-plugin`, `agentic-obsidian-plugin`) is installed (routes to `@coding-document`).
- Evaluates project grooming requests if `agentic-project-grooming-plugin` is installed (routes to `/project-grooming <prompt>`).
- Handles informational inquiries directly without code modifications.
- Rejects conflicting mixed prompts, asking users to separate their requests.

## Installation & Activation

```bash
# Enable in Antigravity session:
agy plugin enable agentic-prompt-triage-plugin
```

## Contributing
See [CONTRIBUTING.md](file:///Users/gpfister/Development/Projects/gpfister/agentic/agentic-prompt-triage-plugin/CONTRIBUTING.md) for contribution guidelines and testing practices.

## License
This project is licensed under the [MIT License](file:///Users/gpfister/Development/Projects/gpfister/agentic/agentic-prompt-triage-plugin/LICENSE).
