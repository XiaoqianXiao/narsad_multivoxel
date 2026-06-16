## Purpose

This file defines how AI assistants should interact with this project.

The goal is to maximize reproducibility, minimize hallucination, and maintain a consistent development workflow across ChatGPT, Codex, and future AI agents.

---

# Required Startup Procedure

Before performing any work, read:

1. PROJECT_CONTEXT.md
2. PROJECT_STATUS.md
3. CODEBASE_MAP.md
4. Latest SESSION_NOTE

Do not rely on chat history as the primary source of project knowledge.

If any of these files are missing or outdated, state the limitation before proceeding.

---

# Planning Before Coding

Always create a plan before modifying code.

The plan should include:

1. Objective
2. Files likely affected
3. Dependencies
4. Risks
5. Validation strategy

Do not begin implementation until the plan is reviewed.

---

# Modification Rules

## Allowed

* Add new functionality
* Refactor existing code when justified
* Improve documentation
* Improve reproducibility
* Add tests
* Add logging
* Improve error handling

## Not Allowed Without Explicit Approval

* Delete scripts
* Rename files
* Change output formats
* Change directory structure
* Remove analyses
* Change statistical models
* Change inclusion/exclusion criteria
* Change primary outcomes
* Change predefined hypotheses

Always explain proposed changes before making them.

---

# Scientific Integrity Rules

Never alter scientific hypotheses.

Never reinterpret results to match expected outcomes.

Never convert exploratory findings into primary findings.

Maintain separation between:

* Primary analyses
* Secondary analyses
* Sensitivity analyses

If a proposed analysis conflicts with PROJECT_CONTEXT.md, flag the conflict.

---

# Reproducibility Requirements

Every analysis should be reproducible.

Whenever creating or modifying code:

* Use configurable paths
* Avoid hard-coded directories
* Preserve random seeds when applicable
* Log major analysis decisions
* Document assumptions

Always report:

* Input files
* Output files
* Key parameters
* Random seed
* Subject counts

---

# Statistical Guardrails

Do not introduce:

* Data leakage
* Circular analysis
* Double dipping
* Post-selection inference without documentation

Ensure:

* Cross-validation remains subject-aware
* Scaling occurs within training folds only
* Feature selection occurs within the correct training structure
* Held-out data remain fully independent

If leakage risk exists, stop and explain the issue.

---

# Coding Standards

Prefer:

* Readable code
* Small functions
* Explicit variable names
* Modular design
* Configuration-driven execution

Avoid:

* Magic numbers
* Hard-coded paths
* Hidden dependencies
* Unused code
* Silent failures

Add comments only when they clarify reasoning.

Do not add comments that simply restate code.

---

# Analysis Workflow

When implementing a new analysis:

Step 1:
Review PROJECT_CONTEXT.md.

Step 2:
Identify relevant files using CODEBASE_MAP.md.

Step 3:
Create implementation plan.

Step 4:
Implement changes.

Step 5:
Run validation.

Step 6:
Summarize:

* Files modified
* Key changes
* Validation performed
* Remaining issues

---

# Session-End Requirements

At the end of a work session generate:

## Session Summary

Completed work.

## Files Modified

List all modified files.

## Decisions Made

Important implementation decisions.

## Remaining Issues

Open problems.

## Recommended Next Step

Highest-priority next action.

Format output as markdown suitable for SESSION_NOTES/YYYY-MM-DD.md.

---

# Communication Style

Be concise.

Prioritize facts over speculation.

When uncertain:

* State uncertainty clearly.
* Explain available evidence.
* Recommend verification steps.

Do not claim that code was executed unless it was actually executed.

Do not claim that files were reviewed unless they were actually reviewed.

Do not invent results.

---

# Failure Recovery

If the task becomes unclear:

1. Stop.
2. Summarize current understanding.
3. List assumptions.
4. Request clarification.

If context becomes too large:

1. Generate session summary.
2. Update SESSION_NOTE.
3. Start a fresh session.

Do not continue a degraded conversation indefinitely.

---

# Project Philosophy

The objective is not merely to produce working code.

The objective is to produce:

* Reproducible analyses
* Scientifically defensible results
* Maintainable code
* Transparent decisions
* Efficient collaboration between human researchers and AI assistants

