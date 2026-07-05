# hermes-plugin-context-pruner

Agent-driven context pruning middleware for [Hermes Agent](https://hermes-agent.nousresearch.com).

## Problem

Hermes appends every tool output and assistant message to the conversation history. Over a long session, this accumulation can bloat context to **80K–150K+ tokens**, most of which is obsolete tool output (logs, SQL results, SSH output, etc.) from previous subtasks.

## Solution: Always-On 3-Tier Smart Pruning (v2.0)

Unlike fixed-window or checkpoint-based approaches, this plugin runs **before every LLM call** and evaluates each message individually using a scoring system:

| Tier | Role | Behavior |
|:----|:----|:---------|
| 🔒 **Protected** | `system` + recent `user`(×3) + recent `assistant`(×5) | **Always kept** — untouched |
| 🔧 **Tool chain** | Tool outputs paired with their triggering assistant call | Kept as atomic groups to prevent API errors |
| 📊 **Scored candidates** | Everything else | Scored by value; lowest-scoring ones are pruned up to 60% cap |

### Scoring Factors

| Factor | Score | Description |
|:-------|:-----:|:------------|
| Active file path reference | **+4** | Message references a file that's likely still relevant |
| Error/reference keywords | **+3** | Message contains tracebacks, error codes, or failure patterns |
| Large tool output | **-1** | Output > 2000 chars gets a size penalty |
| Default | **0** | Baseline for unremarkable messages |

### Safety Caps

- **MAX_PRUNED_PCT = 60%**: never prune more than 60% of total messages
- **Tool call chain integrity**: assistant + tool pairs are kept or dropped together
- **System prompt**: always protected
- **Recent user messages**: last 3 always kept

### Archive (v2.0+)

Every pruning decision is logged to `{hermes-root}/logs/context-pruner.jsonl` with:
- Session ID, timestamps
- Total before/after, dropped count
- Protected group sizes
- Dropped count by role (assistant vs tool)
- Automatic rotation at 10,000 records

## Installation

```bash
git clone https://github.com/849506054/hermes-plugin-context-pruner.git
ln -s $(pwd)/hermes-plugin-context-pruner ~/.hermes/plugins/context-pruner
```

## Configuration

No configuration needed — plugin auto-registers as an `llm_request` middleware.

## Tuning Parameters

Edit `__init__.py` constants:

| Constant | Default | Description |
|:---------|:-------:|:------------|
| `SCORE_KEEP` | 2.0 | Messages scoring ≥ this stay; below may be pruned |
| `LARGE_OUTPUT` | 2000 | Tool output above this (chars) gets a size penalty |
| `MAX_PRUNED_PCT` | 0.60 | Max fraction of messages pruned per pass |
| `ARCHIVE_SILENT` | False | Set True to disable archive writing when stable |

## File Layout

```
├── __init__.py      # Plugin implementation (middleware + scoring + archive)
├── plugin.yaml      # Plugin manifest
├── EVALUATION.md    # Evaluation report & optimization roadmap
├── README.md        # This file
└── .gitignore
```

## License

MIT
