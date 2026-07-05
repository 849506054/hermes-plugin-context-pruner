"""Context pruner plugin for Hermes — Session health maintenance.

Designed for LONG sessions (500+ msgs) where progressive decay
(dead subtopics, superseded data, self-corrected errors) degrades
session quality and triggers premature compression.

Three layers:

1. **Health gate** — skip when the session is still young (< 500 msgs
   or hasn't grown enough since last cleanup).  Preserves prompt-cache
   prefix stability for the vast majority of calls.

2. **Staleness detection** — identify messages that are truly dead:
   ended subtopics, superseded tool outputs, self-corrected content.
   Score baseline raised to 5.0 so most messages pass; only demonstrably
   stale ones enter the candidate pool.

3. **Conservative action** — max 20 % of messages dropped per pass,
   protected + active zones untouched.  Conversation prefix stays stable.

Runs before every LLM call, but returns immediately (near-zero cost)
when the health gate says "not yet needed".
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
# ── Protection zones
PROTECTED_USER = 3         # keep this many most-recent user messages
PROTECTED_ASST = 5         # keep this many most-recent assistant messages
PROTECTED_LEADING = 20     # first N messages (system + initial context) always protected

# ── Health gate (Layer A) ────────────────────────────────────────────────
HEALTH_THRESHOLD_BASE = 180    # msgs — no pruning below this count (v3 optimization: was 500 → 250 → 180)
GROWTH_INTERVAL = 50           # LLM calls — minimum gap between cleanups
GROWTH_THRESHOLD = 30          # msgs — minimum new messages since last cleanup

# ── Scoring (Layer B)
SCORE_KEEP = 4.5           # messages scoring >= this stay; below may be pruned
LARGE_OUTPUT = 2000        # tool output above this gets a size penalty
MAX_PRUNED_PCT = 0.20      # never prune more than 20% of total messages (v3: down from 60%)

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
_skip_cache_lock = threading.Lock()

# ── Layer A state ───────────────────────────────────────────────────────
_call_counter = 0                # monotonic LLM call counter
_last_cleanup_call = 0           # call counter value at last actual cleanup
_last_cleanup_msgs = 0           # message count at last actual cleanup
_skipped_count = 0               # total times health gate skipped pruning
_layer_a_lock = threading.Lock()

# ── Reference extraction patterns ──────────────────────────────────────
_FILE_PATH_RE = re.compile(
    # Unix absolute paths
    r"(?:/[a-zA-Z0-9_./-]+){2,}"
    # Windows paths (drive letter + colon + backslash)
    r"|[a-zA-Z]:\\(?:[a-zA-Z0-9_ .-]+\\)*[a-zA-Z0-9_ .-]+"
    # home-relative paths
    r"|~/[a-zA-Z0-9_./-]+"
    # Python dotted module imports (potential file references)
    r"|(?:from|import)\s+[a-zA-Z_][\w.]*(?:\s+import\s+[\w*]+)?",
)

_REF_RE = re.compile(
    r"(?:Error|Exception|Traceback|ERROR|FAIL|status=\d{3})\s*(?::\s*(\w+))?"
    r"|def\s+(\w+)\("
    r"|class\s+(\w+)"
    r"|([A-Z][A-Z_]{2,})\s*="   # CONSTANT_NAME = ...
    r"|(?:\"([A-Z_]{3,})\"|'([A-Z_]{3,})')"  # quoted constants
)


# ══════════════════════════════════════════════════════════════════════════
#  Middleware entry point
# ══════════════════════════════════════════════════════════════════════════

def register(ctx) -> None:
    ctx.register_middleware("llm_request", pruner_middleware)


def pruner_middleware(request: dict, **context) -> dict | None:
    """Wrap an LLM request, pruning stale messages before pass-through.

    Returns a modified request dict (``{"request": new_request}``), or
    ``None`` to skip the call entirely.
    """
    # Module-level mutable state
    global _call_counter, _last_cleanup_call, _last_cleanup_msgs
    global _skipped_count, _skip_cache

    # Locate the messages key
    messages: list[dict] | None = None
    for key in ("messages", "input", "prompt"):
        val = request.get(key)
        if isinstance(val, list):
            messages = val
            break
    if messages is None:
        return None

    n = len(messages)
    if n <= PROTECTED_USER + PROTECTED_ASST + 10:
        return None  # too tiny to bother

    # ── Layer A: Health gate ─────────────────────────────────────────
    with _layer_a_lock:
        _call_counter += 1
        calls_since_cleanup = _call_counter - _last_cleanup_call
        msgs_since_cleanup = n - _last_cleanup_msgs

        should_prune = (
            n >= HEALTH_THRESHOLD_BASE
            and calls_since_cleanup >= GROWTH_INTERVAL
            and msgs_since_cleanup >= GROWTH_THRESHOLD
        )
        if not should_prune:
            _skipped_count += 1
            return None  # healthy session, preserve cache prefix

        # Reset state for this cleanup run
        _last_cleanup_call = _call_counter
        _last_cleanup_msgs = n
        _skipped_snapshot = _skipped_count

    # ── Skip cache ───────────────────────────────────────────────────
    with _skip_cache_lock:
        h = _quick_hash(messages)
        if h is not None and h == _skip_cache:
            return None  # boundaries unchanged since last prune
        _skip_cache = h

    # ── Prune ────────────────────────────────────────────────────────
    new_messages, dropped, stats = _prune(messages)
    if not dropped:
        return None

    new_request = dict(request)
    for key in ("messages", "input", "prompt"):
        if key in request:
            new_request[key] = new_messages
            break

    logger.info(
        "pruned %d/%d msgs (%.1f%%)  skipped=%d  session=%s",
        dropped, n, dropped / n * 100,
        _skipped_snapshot,
        context.get("session_id", "?"),
    )

    session_id = context.get("session_id") or "?"
    _archive_prune_result(n, len(new_messages), dropped, stats,
                          session_id, skipped=_skipped_snapshot)

    return {"request": new_request}


# ══════════════════════════════════════════════════════════════════════════
#  Layer A helper: state reset
# ══════════════════════════════════════════════════════════════════════════

def reset_health_state() -> None:
    """Reset Layer A counters (useful for testing or forced cleanup)."""
    global _call_counter, _last_cleanup_call, _last_cleanup_msgs, _skipped_count
    with _layer_a_lock:
        _call_counter = 0
        _last_cleanup_call = 0
        _last_cleanup_msgs = 0
        _skipped_count = 0


# ══════════════════════════════════════════════════════════════════════════
#  Core pruning logic
# ══════════════════════════════════════════════════════════════════════════

def _quick_hash(messages: list[dict]) -> tuple[int, int] | None:
    """Hash first 3 + last 3 messages as a fingerprint for skip cache."""
    sample: list[str] = []
    indices = [0, 1, 2, -3, -2, -1]
    for i in indices:
        if i < len(messages):
            content = str(messages[i].get("content", ""))[-80:]
            sample.append(f"{i}:{content}")
    return (len(messages), hash("".join(sample)))


def _prune(messages: list[dict]) -> tuple[list[dict], int, dict]:
    """Return (new_messages, dropped_count, stats_dict).

    Three-layer pruning:
    1. Structural zoning — core + protected + aging
    2. Chain-connected flood-fill (existing, keep for ref-chain awareness)
    3. Staleness scoring (rewritten v3: 5.0 baseline, 4.5 keep threshold)
    """
    # ── Build zone maps ──────────────────────────────────────────────
    protected_indices: set[int] = set()
    all_system_indices: list[int] = []

    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        if role == "system":
            all_system_indices.append(i)
            protected_indices.add(i)

    # Core zone (first N messages)
    for i in range(min(PROTECTED_LEADING, len(messages))):
        protected_indices.add(i)

    # Active user zone (most recent user messages)
    user_indices = [i for i, m in enumerate(messages)
                    if m.get("role") == "user"]
    for i in user_indices[-PROTECTED_USER:]:
        protected_indices.add(i)

    # Active assistant zone (most recent assistant messages)
    asst_indices = [i for i, m in enumerate(messages)
                    if m.get("role") == "assistant"]
    for i in asst_indices[-PROTECTED_ASST:]:
        protected_indices.add(i)

    # System messages anywhere in the conversation
    for i in all_system_indices:
        protected_indices.add(i)

    # ── Layer 2: Chain-connected flood-fill ──────────────────────────
    # Seed from protected zone
    active_paths: set[str] = set()
    active_refs: set[str] = set()
    for i in protected_indices:
        p = _collect_paths([messages[i]])
        r = _collect_refs([messages[i]])
        active_paths.update(p)
        active_refs.update(r)

    chain_connected: set[int] = set()
    # Walk backwards from protected zone through aging zone
    aging_range = list(range(min(PROTECTED_LEADING, len(messages)),
                             len(messages) - max(PROTECTED_USER, PROTECTED_ASST)))
    for i in reversed(aging_range):
        if i in protected_indices:
            continue
        msg_paths = _collect_paths([messages[i]])
        msg_refs = _collect_refs([messages[i]])
        if msg_paths & active_paths or msg_refs & active_refs:
            chain_connected.add(i)
            active_paths.update(msg_paths)
            active_refs.update(msg_refs)

    # ── Layer 3: Data coverage detection ─────────────────────────────
    superseded_map: dict[str, int] = {}   # path → last_idx
    for i, msg in enumerate(messages):
        if msg.get("role") == "tool":
            for p in _collect_paths([msg]):
                superseded_map[p] = i

    # ── Scoring ─────────────────────────────────────────────────────
    candidate_indices: list[int] = []

    stats = {
        "total": len(messages),
        "protected_system": len(all_system_indices),
        "protected_user": min(PROTECTED_USER, len(user_indices)),
        "protected_assistant": min(PROTECTED_ASST, len(asst_indices)),
        "protected_other": 0,
        "protected_leading": PROTECTED_LEADING,
        "chain_connected": len(chain_connected),
        "superseded_hit": 0,
        "candidates": 0,
        "max_droppable": 0,
    }

    for i, msg in enumerate(messages):
        if i in protected_indices:
            continue

        role = msg.get("role", "")
        score = _score_message(
            msg,
            msg_idx=i,
            chain_connected=(i in chain_connected),
            superseded_map=superseded_map,
            messages=messages,
        )

        if score < SCORE_KEEP:
            candidate_indices.append(i)
        else:
            # Protected by score
            protected_indices.add(i)

    stats["candidates"] = len(candidate_indices)

    # ── Cap pruning ──────────────────────────────────────────────────
    max_droppable = int(len(messages) * MAX_PRUNED_PCT)
    stats["max_droppable"] = max_droppable

    # Sort candidates by score ascending (worst first)
    candidate_scores = []
    for idx in candidate_indices:
        s = _score_message(
            messages[idx],
            msg_idx=idx,
            chain_connected=(idx in chain_connected),
            superseded_map=superseded_map,
            messages=messages,
        )
        candidate_scores.append((s, idx))
    candidate_scores.sort(key=lambda x: x[0])

    # ── Select candidates with tool_call chain budget ──
    # If an assistant with tool_calls is dropped, its tool outputs must
    # also be dropped. Account for the full (assistant + tools) cost so
    # the MAX_PRUNED_PCT cap is respected.
    # Build tool chain map: assistant_idx → [tool_idx, ...]
    assistant_tools: dict[int, list[int]] = {}
    for idx, msg in enumerate(messages):
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            tools = []
            j = idx + 1
            while j < len(messages) and messages[j].get("role") == "tool":
                tools.append(j)
                j += 1
            assistant_tools[idx] = tools

    final_drop: set[int] = set()
    remaining_budget = max_droppable

    for score, idx in candidate_scores:
        if remaining_budget <= 0:
            break

        msg = messages[idx]
        role = msg.get("role", "")

        # Tool messages are not independently droppable — skip;
        # they're handled via their parent assistant's cost.
        if role == "tool":
            continue

        # Compute real cost: assistant + any tool outputs that must follow
        cost = 1
        unkept_tools: list[int] = []
        for t in assistant_tools.get(idx, []):
            if t not in protected_indices:
                unkept_tools.append(t)
        cost += len(unkept_tools)

        if cost <= remaining_budget:
            final_drop.add(idx)
            final_drop.update(unkept_tools)
            remaining_budget -= cost

    # ── Build result ─────────────────────────────────────────────────
    new_messages = [m for idx, m in enumerate(messages) if idx not in final_drop]
    dropped_count = len(final_drop)

    # Drop role breakdown
    drop_roles: dict[str, int] = {}
    for idx in final_drop:
        role = messages[idx].get("role", "unknown")
        drop_roles[role] = drop_roles.get(role, 0) + 1

    stats["drop_role_breakdown"] = drop_roles
    stats["chain_connected"] = len(chain_connected & final_drop)

    # Count superseded hits among dropped
    superseded_dropped = 0
    for idx in final_drop:
        if idx in chain_connected:
            continue  # chain-connected msg dropped = edge case
        msg_paths = _collect_paths([messages[idx]])
        for p in msg_paths:
            last_idx = superseded_map.get(p)
            if last_idx is not None and last_idx > idx:
                superseded_dropped += 1
                break
    stats["superseded_hit"] = superseded_dropped

    return new_messages, dropped_count, stats


# ══════════════════════════════════════════════════════════════════════════
#  v3 scoring: focus on "has this message been rendered irrelevant?"
# ══════════════════════════════════════════════════════════════════════════

def _score_message(
    msg: dict,
    msg_idx: int = -1,
    chain_connected: bool = False,
    superseded_map: dict[str, int] | None = None,
    messages: list[dict] | None = None,
) -> float:
    """Score a single message (higher = more valuable to keep).

    v3 design:
      - Role baselines raised to 4-6 range (vs 0-2 in v2.x)
      - SCORE_KEEP = 4.5 → most messages pass at baseline
      - Only messages with clear "staleness signals" drop below threshold
      - Focus on being REFERENCED BY later messages (not just having refs)
    """
    role = msg.get("role", "")
    content = str(msg.get("content", "") or "")

    # ── Role baseline ────────────────────────────────────────────────
    baseline: float = {
        "user": 6.0,
        "assistant": 5.0,
        "tool": 4.0,
        "system": 7.0,
    }.get(role, 4.0)

    score = baseline

    # ── Staleness: referenced by later messages? ─────────────────────
    # This is the KEY v3 metric — a message is valuable if subsequent
    # messages refer back to it.
    if messages is not None and msg_idx >= 0:
        if _referenced_by_later_messages(msg, msg_idx, messages):
            score += 3.0

    # ── Chain-connected bonus (Layer 2 legacy keep) ──────────────────
    if chain_connected:
        score += 1.5

    # ── Active file-path reference bonus ─────────────────────────────
    content_paths = _collect_paths([msg])
    content_refs = _collect_refs([msg])

    # Still useful: contains concrete paths/definitions
    if len(content_paths) >= 1:
        score += 0.5
    if len(content_refs) >= 1:
        score += 0.5

    # ── Data coverage penalty ────────────────────────────────────────
    if role == "tool" and superseded_map and msg_idx >= 0:
        for p in content_paths:
            last_idx = superseded_map.get(p)
            if last_idx is not None and last_idx > msg_idx:
                score -= 1.5  # this copy of data is stale
                break

    # ── Size penalty (only for tool output) ──────────────────────────
    if role == "tool" and content:
        excess = (len(content) - LARGE_OUTPUT) // 1000
        score -= min(2.0, excess * 0.5)

    # ── Self-corrected content penalty ───────────────────────────────
    # If later messages explicitly correct this one, it's stale
    if messages is not None and msg_idx >= 0:
        if _corrected_by_later_messages(msg, msg_idx, messages):
            score -= 2.0

    # ── Empty/null content penalty ──────────────────────────────────
    if not content or content in ("", " ", None):
        score -= 2.0

    return score


def _referenced_by_later_messages(
    msg: dict, msg_idx: int, messages: list[dict]
) -> bool:
    """Check if any message AFTER msg_idx explicitly references this message.

    Heuristic: look for quotation patterns, "as you said", file-path
    mentions, and error-name mentions in subsequent messages that match
    content in this message.
    """
    if msg_idx >= len(messages) - 1:
        return False  # last message can't be referenced

    content = str(msg.get("content", "") or "")
    if not content or len(content) < 50:
        return False  # short messages rarely get referenced

    # Extract distinctive tokens from this message
    # 1. File paths
    paths = _collect_paths([msg])
    # 2. Error/function/constant names
    refs = _collect_refs([msg])
    # 3. Key phrases (first meaningful sentence)
    key_phrases: list[str] = []
    for line in content.split("\n"):
        stripped = line.strip()
        if stripped and len(stripped) > 30 and len(stripped) < 200:
            # Grab the first ~50 chars of each meaningful line
            key_phrases.append(stripped[:80])

    # Check subsequent messages for references
    check_window = min(msg_idx + 20, len(messages))  # check up to 20 messages ahead
    for j in range(msg_idx + 1, check_window):
        later_content = str(messages[j].get("content", "") or "")

        # Check path references
        if paths:
            later_paths = _collect_paths([messages[j]])
            if paths & later_paths:
                return True

        # Check error/function/constant references
        if refs:
            later_refs = _collect_refs([messages[j]])
            if refs & later_refs:
                return True

        # Check for explicit quotation patterns
        for phrase in key_phrases[:5]:  # limit to first 5 key phrases
            if phrase and phrase in later_content:
                return True

    return False


def _corrected_by_later_messages(
    msg: dict, msg_idx: int, messages: list[dict]
) -> bool:
    """Check if subsequent messages explicitly correct this one.

    Heuristic: look for correction markers like "actually", "correction",
    "that's wrong", "I was mistaken", etc. within the next few messages.
    """
    if msg_idx >= len(messages) - 2:
        return False

    check_window = min(msg_idx + 5, len(messages))
    correction_patterns = re.compile(
        r"(actually|correction|wait|sorry|my mistake|"
        r"that.*(?:wrong|incorrect|error)|"
        r"let me (?:correct|fix|retry)|"
        r"previous.*(?:wrong|incorrect)|"
        r"instead of|not correct|misunderstood)",
        re.IGNORECASE,
    )

    for j in range(msg_idx + 1, check_window):
        later_content = str(messages[j].get("content", "") or "")
        if correction_patterns.search(later_content):
            return True

    return False


# ══════════════════════════════════════════════════════════════════════════
#  Reference extraction helpers
# ══════════════════════════════════════════════════════════════════════════

def _collect_paths(messages: list[dict]) -> set[str]:
    """Extract unique absolute file paths from messages."""
    paths: set[str] = set()
    for msg in messages:
        content = str(msg.get("content", "") or "")
        for m in _FILE_PATH_RE.finditer(content):
            raw = m.group(0).strip()
            if len(raw) > 5 and "/" in raw[1:]:
                paths.add(raw)
    return paths


def _collect_refs(messages: list[dict]) -> set[str]:
    """Extract reference strings (error names, function defs, constants)."""
    refs: set[str] = set()
    for msg in messages:
        content = str(msg.get("content", "") or "")
        for m in _REF_RE.finditer(content):
            for group in m.groups():
                if group:
                    refs.add(group)
    return refs


# ══════════════════════════════════════════════════════════════════════════
#  Archive
# ══════════════════════════════════════════════════════════════════════════

def _archive_prune_result(
    total_before: int,
    total_after: int,
    dropped: int,
    stats: dict,
    session_id: str,
    skipped: int = 0,
) -> None:
    """Append a pruning record to the JSONL archive (best-effort)."""
    if ARCHIVE_SILENT:
        return

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "total_before": total_before,
        "total_after": total_after,
        "dropped": dropped,
        "retention_ratio": round(total_after / total_before, 4) if total_before else 0.0,
        "drop_pct": round(dropped / total_before, 4) if total_before else 0.0,
        "skipped_since_last": skipped,  # v3: health gate skips since last cleanup
        "protected_system": stats.get("protected_system", 0),
        "protected_user": stats.get("protected_user", 0),
        "protected_assistant": stats.get("protected_assistant", 0),
        "protected_other": stats.get("protected_other", 0),
        "protected_leading": stats.get("protected_leading", 0),
        "candidates": stats.get("candidates", 0),
        "max_droppable": stats.get("max_droppable", 0),
        "chain_connected": stats.get("chain_connected", 0),
        "superseded_hit": stats.get("superseded_hit", 0),
        "drop_role_breakdown": stats.get("drop_role_breakdown", {}),
        "v3_health_threshold": HEALTH_THRESHOLD_BASE,
        "v3_growth_interval": GROWTH_INTERVAL,
        "v3_growth_threshold": GROWTH_THRESHOLD,
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
