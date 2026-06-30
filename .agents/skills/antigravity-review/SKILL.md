---
name: antigravity-review
description: Use this skill when a task would benefit from a second-opinion model review, external Gemini/Google/Antigravity review, large-diff review, bug-risk review, verification logic review, WPC ERO risk review, website automation review, cron/publishing review, or when the user asks to save Codex tokens by offloading review/summarization to Antigravity CLI.
---

# Antigravity Review Skill

Use this skill as an optional second-opinion reviewer through Google Antigravity CLI (`agy`).

## When to use automatically

Use this skill when any of the following are true:

- The user asks for Gemini, Google, or Antigravity review.
- The user asks to offload token usage.
- The current diff is large or touches multiple files.
- The task involves WPC ERO risk ingest, rasterization, or verification metrics.
- The task involves GitHub Pages publishing, cron automation, or website verification uploads.
- The task involves checking whether a bug may affect past v33 statistics.
- The task involves a high-risk code change where an independent review would help.

## When not to use

Do not use this skill for simple one-line edits, formatting-only changes, or obvious shell commands.

Do not block the main task if `agy` is unavailable.

## Required checks before using

Run:

```bash
command -v agy
agy --version
