---
name: antigravity-review
description: Optional second-opinion review using Google Antigravity CLI (`agy`) if installed. Use when the user asks for Gemini, Antigravity, Google-model review, or an external second-opinion review.
---

# Antigravity Review Skill

This skill is optional. Use it only if the `agy` command is installed and authenticated in the same environment where Codex is running.

Rules:
- Do not assume `agy` is installed.
- Do not assume the old `gemini` CLI is available.
- Do not block the main task if Antigravity is unavailable.
- Do not let Antigravity directly edit this repository.
- Run Antigravity only as a second-opinion reviewer from a temporary directory.
- Never use `--dangerously-skip-permissions`.

Use:

```bash
.agents/skills/antigravity-review/scripts/agy_review.sh
