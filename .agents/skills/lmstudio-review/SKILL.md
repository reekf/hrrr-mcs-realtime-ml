---
name: lmstudio-review
description: Use this skill when a local LM Studio model should review or summarize code, logs, diffs, website automation, WPC ERO logic, v33 verification/statistics changes, cron scripts, or when the user asks to offload token usage to a local model.
---

# LM Studio Local Review Skill

This skill calls a local LM Studio OpenAI-compatible server.

Use only when running locally on the same machine as LM Studio.

Default endpoint:

```text
http://localhost:1234/v1/chat/completions
