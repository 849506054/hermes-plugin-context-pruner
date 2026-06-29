# context-pruner v2.1 — 优化设计方案

**基于 v2.0 运行数据（202 条存档记录）和源码审计的分析建议。**

---

## 目录

1. [问题优先级矩阵](#1-问题优先级矩阵)
2. [A: 评分系统重校（Critical）](#a-评分系统重校critical)
3. [B: 修复 Skip Cache（Bugfix）](#b-修复-skip-cachebugfix)
4. [C: 移除死代码（Cleanup）](#c-移除死代码cleanup)
5. [D: 自适应裁剪（Nice-to-have）](#d-自适应裁剪nice-to-have)
6. [实现计划](#6-实现计划)

---

## 1. 问题优先级矩阵

| 编号 | 问题 | 严重度 | 影响范围 | 风险 | 工作量 | 优先级 |
|:---:|:----|:-----:|:--------:|:---:|:-----:|:-----:|
| A | `SCORE_KEEP` 阈值过低，评分退化 | 中 | 全部裁剪决策 | 低 | 小 | **P0** |
| B | Skip Cache 冗余（msg_count 即 key） | 低 | 短会话重复裁剪 | 低 | 极小 | **P0** |
| C | `_CHECKPOINT_RE` 死代码 | 低 | 无 | 无 | 极小 | **P1** |
| D | 始终触及 `max_droppable` 上限 | 中 | 裁剪效果 | 中 | 大 | P2 |

---

## 2. A: 评分系统重校（Critical）

### 当前问题

```python
# 当前（v2.0）
score = 1.0  # 所有消息统一基线
# 只有活跃路径引用(+4)或错误引用(+3)能让分数 > SCORE_KEEP=2.0
# 结果：100% 的非保护消息都成为候选
```

### 根因

基线 `1.0` 对所有角色一视同仁，而 `SCORE_KEEP=2.0` 只需超过 2.0 即可豁免。实际上：

- 一个空的 tool 输出得 1.0 → 候选（正确）
- 一个 200 行的重要 assistant 分析也得 1.0 → 候选（不合理）
- 唯一能免裁的是有活跃文件/错误引用的消息

### 优化方案

**按角色区分基线：**

| 角色 | 新基线 | 理由 |
|:----|:-----:|:-----|
| `user` | 3.0 | 用户意图驱动，已在保护层，冗余但无害 |
| `assistant` | **2.0** | 包含推理、代码、分析和结构化回复，有内在价值 |
| `tool` | **0.5** | 原始输出是膨胀主因，无引用时应优先裁剪 |

同时保留 SCORE_KEEP=2.0 不变。

### 预期效果

| 场景 | 旧分 | 新分 | 是否候选 |
|:----|:----:|:----:|:--------:|
| 普通 assistant 回复 | 1.0 | **2.0** | ❌（刚好留） |
| 有路径引用的 assistant | 5.0 | **6.0** | ❌（强留） |
| 普通 tool 输出 | 1.0 | **0.5** | ✅（优先裁） |
| 大体积 tool 输出 (>2000字) | -2.0~1.0 | **-2.5~0.5** | ✅（最优先裁） |
| 有路径引用的 tool | 5.0 | **4.5** | ❌（留，引用有价值） |
| user 消息 | 3.0 | 3.0 | ❌（已在保护层） |

**核心改变：** assistant 消息默认脱离候选池，tool 输出成为主要裁剪对象。这符合直觉——assistant 的推理/代码应保留，tool 的原始日志应优先清理。

### 代码变更

```diff
 # _score_message()
-    score = 1.0  # baseline
+    score = 2.0 if role == "assistant" else 0.5 if role == "tool" else 3.0 if role == "user" else 1.0
```

---

## 3. B: 修复 Skip Cache（Bugfix）

### 当前问题

```python
# 当前（v2.0）— 有 Bug
cache_key = f"prune:{msg_count}"          # 仅基于消息数量
if _skip_cache[0] == msg_count and _skip_cache[1] == cache_key:
    return                                 # 只要数量相同就跳过
_skip_cache = (msg_count, cache_key)       # 两个字段都是 msg_count
```

**两个字段完全冗余** (`cache_key` 仅仅是 `"prune:" + str(msg_count)`)，所以当 LLM 请求数不变但上下文内容变化时，不会重新裁剪。

### 优化方案

替换为基于内容哈希的 skip cache：

```python
def _quick_hash(messages: list[dict]) -> int:
    """Hash first 3 + last 3 messages as a fingerprint."""
    sample = []
    for i in [0, 1, 2, -3, -2, -1]:
        if i < len(messages):
            sample.append(str(i) + ":" + str(messages[i].get("content", ""))[-80:])
    return hash("".join(sample))

# 在 middleware 中
cache_key = _quick_hash(messages)
```

**效果：** 只有当上下文内容实质变化时，才重新执行裁剪判定。

---

## 4. C: 移除死代码（Cleanup）

### 当前
```python
_CHECKPOINT_RE = re.compile(r"<!--\s*CHECKPOINT:\s*(.+?)\s*-->")
```

定义在第 25 行，从未在任何地方被引用或调用。

### 优化方案

直接删除 `_CHECKPOINT_RE` 行。该变量对应于 v1.0 的 checkpoint-based 策略，v2.0 已彻底改为 Always-On 评分模式，不再需要。

---

## 5. D: 自适应裁剪（Nice-to-have）

### 背景

即使在 A 优化后，仍然可能存在以下情况：
- 会话中大段高价值内容（代码审查、架构讨论）不应过度裁剪
- 会话中大量低价值内容（批量工具输出）可以激进裁剪

### 优化方案

引入**动态上限**概念替代固定 `MAX_PRUNED_PCT=0.60`：

```python
# 根据候选池的分数分布调整裁剪上限
avg_candidate_score = sum(s for _, s in candidates) / len(candidates)
if avg_candidate_score > 1.0:
    # 候选池整体分不低 → 手下留情
    dynamic_cap = min(MAX_PRUNED_PCT, avg_candidate_score * 0.3)
else:
    # 候选池整体很低 → 可激进裁
    dynamic_cap = MAX_PRUNED_PCT
max_droppable = int(n * dynamic_cap)
```

### 优先级

P2 — 可在后续版本实现，先验证 A+B+C 的效果。

---

## 6. 实现计划

| 步骤 | 变更 | 影响 |
|:----|:----|:----|
| **A** 评分基线重校 ~5 行 | `_score_message` | **核心效果提升** |
| **B** 修复 Skip Cache ~10 行 | `pruner_middleware` | 消除冗余计算 |
| **C** 移除死代码 ~1 行 | 删除 `_CHECKPOINT_RE` | 干净 |
| **D** 自适应裁剪 ~15 行 | `_prune` 动态上限 | P2 延后 |
| 验证 | 运行 1 小时 → 检查存档 | 确认效果 |

### 验证指标

| 指标 | v2.0 基线 | v2.1 预期目标 |
|:----|:---------:|:--------------|
| 触及 max_droppable 占比 | 100% | **< 60%** |
| 候选池实际裁剪率 | 75% | **40-80%（自适应）** |
| 平均保留率 | 35.9% | **40-50%** |
| assistant 裁剪占比 | 44% | **< 20%** |
| tool 裁剪占比 | 56% | **> 80%** |
| archive 记录频率 | 每 LLM 请求 | 仅内容变化时 |

---

**结论：** 通过 A+B+C 三项精准修改，可在 ~15 行代码变动内使裁剪器从"按数量上限机械截断"转变为"真正的按价值智能裁剪"。
