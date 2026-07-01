"""Context pruner plugin for Hermes — Always-On smart pruning.

Two tiers:

1. **Protected** — system prompts + recent 3 user + recent 5 assistant messages.
   Never removed.

2. **Reducible** — older messages with no active file/error references.
   Deleted as a contiguous block from the conversation start.

Runs before every LLM call, regardless of checkpoint markers.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────
PROTECTED_USER = 3      # keep this many most-recent user messages
PROTECTED_ASST = 5      # keep this many most-recent assistant messages
SCORE_KEEP = 2.0        # messages scoring >= this stay; below may be pruned
LARGE_OUTPUT = 2000     # tool output above this gets a size penalty
MAX_PRUNED_PCT = 0.60   # never prune more than 60% of total messages

# ── Archive ─────────────────────────────────────────────────────────────
ARCHIVE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "logs", "context-pruner.jsonl",
)
ARCHIVE_MAX_RECORDS = 10000    # rotate when exceeded
ARCHIVE_SILENT = False          # set True to stop archive writes (stable stage)
_archive_lock = threading.Lock()

# ── Skip cache ──────────────────────────────────────────────────────────
_skip_cache: tuple[int, int] | None = None  # (count, content_hash)
_skip_lock = threading.Lock()

# ── Content-hash helper ────────────────────────────────────────────────
def _quick_hash(messages: list[dict]) -> tuple[int, int]:
    """Hash first 3 + last 3 messages as a fingerprint.

    Only considers the last 80 chars of each sampled message's content.
    When messages have the same count and same content boundaries,
    pruning would produce the same result → safe to skip.
    """
    sample: list[str] = []
    indices = [0, 1, 2, -3, -2, -1]
    for i in indices:
        if i < len(messages):
            content = str(messages[i].get("content", ""))[-80:]
            sample.append(f"{i}:{content}")
    return (len(messages), hash("".join(sample)))


# ═════════════════════════════════════════════════════════════════════════
#  Plugin entry point
# ═════════════════════════════════════════════════════════════════════════

def register(ctx):
    """Register the llm_request middleware."""
    ctx.register_middleware("llm_request", pruner_middleware)
    logger.info("context-pruner: registered llm_request middleware (Always-On)")


# ═════════════════════════════════════════════════════════════════════════
#  Middleware
# ═════════════════════════════════════════════════════════════════════════

def pruner_middleware(request, **context):
    """Rewrite ``request["messages"]`` before each API call."""
    messages = request.get("messages")
    if not isinstance(messages, list):
        alt = request.get("input")
        if isinstance(alt, list):
            messages = alt
        else:
            return

    if len(messages) <= 3:
        return

    msg_count = len(messages)

    # ── Skip cache (content-hash) ──────────────────────────────────────
    _cache_key = _quick_hash(messages)
    global _skip_cache
    with _skip_lock:
        cached = _skip_cache is not None and _skip_cache == _cache_key
    if cached:
        return
    with _skip_lock:
        _skip_cache = _cache_key

    # ── Run pruning ─────────────────────────────────────────────────
    new_messages, dropped, stats = _prune(messages)

    if dropped <= 0:
        return

    kept = len(new_messages)
    logger.info(
        "context-pruner: pruned %d messages (kept %d, always-on)",
        dropped, kept,
    )

    # ── Archive ──────────────────────────────────────────────────────
    if not ARCHIVE_SILENT:
        _archive_prune_result(
            session_id=context.get("session_id"),
            total_before=msg_count,
            total_after=kept,
            dropped=dropped,
            stats=stats,
        )

    target = "messages" if "messages" in request else "input"
    new_request = dict(request)
    new_request[target] = new_messages
    return {"request": new_request}


# ═════════════════════════════════════════════════════════════════════════
#  Pruning engine
# ═════════════════════════════════════════════════════════════════════════

# Regexes for extracting references
_PATH_RE = re.compile(r"(?:/[\w.\-]+)+[\w./\-]*\.[a-zA-Z]{1,4}(?:\:\d+)?")
_REF_RE = re.compile(
    r"(?:"
    r"(?:Error|Exception|Traceback|ERROR|FAIL|status=\d{3})\s*(?::\s*(\w+))?"
    r"|"
    r"def\s+(\w+)"
    r"|"
    r"\b([A-Z_]{4,})\b"
    r")"
)


def _prune(messages: list[dict]) -> tuple[list[dict], int, dict]:
    """Score all non-protected messages and drop the lowest-value ones.

    Unlike the old contiguous-block-from-start approach, this scores every
    message independently and removes the lowest-scoring ones regardless
    of position.  This handles interleaved garbage — resolved investigation
    cycles, stale tool outputs, self-correction noise — that would otherwise
    be protected by a single high-score anchor message.

    Returns:
        (new_messages, dropped_count, stats_dict)
    """
    n = len(messages)
    stats: dict = {
        "total": n,
        "protected_system": 0,
        "protected_user": 0,
        "protected_assistant": 0,
        "protected_tool": 0,
        "protected_other": 0,
        "candidates": 0,
        "max_droppable": 0,
        "drop_role_breakdown": {},
        "retention_ratio": 1.0,
    }
    # Don't bother pruning small sessions: scoring needs enough samples
    # to be meaningful. A session must have at least 50 unprotected
    # messages beyond the protected zone before we attempt pruning.
    if n <= PROTECTED_USER + PROTECTED_ASST + 50:
        return list(messages), 0, stats

    # ── Identify protected zone ─────────────────────────────────────
    protected_indices: set[int] = set()

    # System prompts always protected
    for i, msg in enumerate(messages):
        if msg.get("role") == "system":
            protected_indices.add(i)
            stats["protected_system"] += 1

    # Recent user + assistant messages protected
    user_found = 0
    asst_found = 0
    for i in range(n - 1, -1, -1):
        if user_found >= PROTECTED_USER and asst_found >= PROTECTED_ASST:
            break
        role = messages[i].get("role", "")
        if role == "user" and user_found < PROTECTED_USER:
            protected_indices.add(i)
            user_found += 1
            stats["protected_user"] += 1
        elif role == "assistant" and asst_found < PROTECTED_ASST:
            protected_indices.add(i)
            asst_found += 1
            stats["protected_assistant"] += 1

    # ── Build reference index from protected zone ───────────────────
    recent_msgs = [messages[i] for i in range(n) if i in protected_indices]
    recent_paths = _collect_paths(recent_msgs)
    recent_refs = _collect_refs(recent_msgs)

    # ── Score all non-protected messages ────────────────────────────
    candidates: list[tuple[int, float]] = []
    for i in range(n):
        if i in protected_indices:
            continue
        score = _score_message(messages[i], recent_paths, recent_refs)
        if score < SCORE_KEEP:
            candidates.append((i, score))

    if not candidates:
        return list(messages), 0, stats

    # ── Sort by score ascending (worst first) → pick what to drop ───
    candidates.sort(key=lambda x: x[1])
    max_droppable = int(n * MAX_PRUNED_PCT)
    to_drop = min(len(candidates), max_droppable)
    stats["candidates"] = len(candidates)
    stats["max_droppable"] = max_droppable
    drop_indices = set(c[0] for c in candidates[:to_drop])

    # ── Tool call chain integrity ───────────────────────────────────
    # DeepSeek enforces: an assistant message with tool_calls must be
    # immediately followed by tool responses for each tool_call_id.
    # If the pruner splits a tool_call chain, the API returns 400.
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            # Find end of the tool response block
            j = i + 1
            while j < len(messages) and messages[j].get("role") == "tool":
                j += 1
            if i not in drop_indices:
                # Assistant kept → all tool responses must also stay
                for k in range(i + 1, j):
                    drop_indices.discard(k)
            else:
                # Assistant dropped → all tool responses must also drop
                for k in range(i + 1, j):
                    drop_indices.add(k)
            i = j  # Skip past the tool block
        else:
            i += 1

    # ── Build pruned list (preserve chronological order) ────────────
    new_messages: list[dict] = []
    for i, msg in enumerate(messages):
        if msg.get("role") == "system" or i not in drop_indices:
            new_messages.append(msg)

    dropped = len(messages) - len(new_messages)

    # ── Role breakdown of dropped messages ─────────────────────────
    drop_roles: dict[str, int] = {}
    for i in drop_indices:
        role = messages[i].get("role", "unknown")
        drop_roles[role] = drop_roles.get(role, 0) + 1
    stats["drop_role_breakdown"] = drop_roles
    stats["retention_ratio"] = round(len(new_messages) / len(messages), 4)

    return new_messages, dropped, stats


def _score_message(msg: dict, recent_paths: set, recent_refs: set) -> float:
    """Score a single message for usefulness. Higher = keep."""
    role = msg.get("role", "")
    content = msg.get("content", "")
    if not isinstance(content, str):
        content = ""

    score = 1.5 if role == "assistant" else 0.5 if role == "tool" else 3.0 if role == "user" else 1.0  # baseline by role

    # Size penalty for tool outputs (biggest bloat culprits)
    if role == "tool" and len(content) > LARGE_OUTPUT:
        excess = (len(content) - LARGE_OUTPUT) // 1000
        score -= min(3.0, excess * 0.5)

    # Active file path references → highly useful
    content_paths = _collect_paths([msg])
    if content_paths & recent_paths:
        score += 4.0

    # Active error/function/constant references
    content_refs = _collect_refs([msg])
    if content_refs & recent_refs:
        score += 3.0

    return score


def _collect_paths(msgs: list[dict]) -> set[str]:
    """Extract file paths from a batch of messages."""
    paths: set[str] = set()
    for msg in msgs:
        content = msg.get("content", "")
        if isinstance(content, str):
            for m in _PATH_RE.finditer(content):
                paths.add(m.group(0).rstrip(":"))
    return paths


def _collect_refs(msgs: list[dict]) -> set[str]:
    """Extract reference strings (error names, function defs, constants)."""
    refs: set[str] = set()
    for msg in msgs:
        content = msg.get("content", "")
        if isinstance(content, str):
            for m in _REF_RE.finditer(content):
                val = m.group(1) or m.group(2) or m.group(3)
                if val and len(val) > 1:
                    refs.add(val.upper())
    return refs


# ═════════════════════════════════════════════════════════════════════════
#  Archive
# ═════════════════════════════════════════════════════════════════════════

def _archive_prune_result(
    session_id: str | None,
    total_before: int,
    total_after: int,
    dropped: int,
    stats: dict,
) -> None:
    """Write one prune record to the JSONL archive.

    Each line is a self-contained JSON object.  File is rotated when
    it exceeds ARCHIVE_MAX_RECORDS.
    """
    if ARCHIVE_SILENT:
        return

    try:
        os.makedirs(os.path.dirname(ARCHIVE_PATH), exist_ok=True)
    except OSError:
        return  # can't create dir, skip silently

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "total_before": total_before,
        "total_after": total_after,
        "dropped": dropped,
        "retention_ratio": stats.get("retention_ratio", 1.0),
        "drop_pct": round(dropped / total_before, 4) if total_before > 0 else 0.0,
        "protected_system": stats.get("protected_system", 0),
        "protected_user": stats.get("protected_user", 0),
        "protected_assistant": stats.get("protected_assistant", 0),
        "protected_other": stats.get("protected_other", 0),
        "candidates": stats.get("candidates", 0),
        "max_droppable": stats.get("max_droppable", 0),
        "drop_role_breakdown": stats.get("drop_role_breakdown", {}),
    }

    with _archive_lock:
        try:
            # Rotate if oversized
            try:
                if os.path.isfile(ARCHIVE_PATH):
                    with open(ARCHIVE_PATH, "r") as f:
                        line_count = sum(1 for _ in f)
                    if line_count >= ARCHIVE_MAX_RECORDS:
                        base = ARCHIVE_PATH
                        bak = base + ".bak"
                        if os.path.isfile(bak):
                            os.remove(bak)
                        os.rename(base, bak)
            except OSError:
                pass  # proceed to append anyway

            with open(ARCHIVE_PATH, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass  # best-effort archiving, never crash the middleware
