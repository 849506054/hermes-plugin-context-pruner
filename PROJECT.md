# context-pruner

## 📋 项目卡片

| 字段 | 值 |
|------|-----|
| **领域** | Hermes 插件 — 中间件 |
| **性质** | 🔴 自有开发 |
| **目的** | Always-On 智能会话裁剪，每轮 LLM 调用前降低上下文长度 |
| **阶段** | 维护优化 |
| **状态** | 🟢 v2.1.1 已部署运行中 |
| **源码位置** | `/opt/data/plugins/context-pruner/` |
| **Git remote** | `github.com/849506054/context-pruner` |
| **依赖项目** | Hermes Gateway（llm_request 中间件） |
| **相关项目** | 同属 Hermes 自有插件生态 |

## 🎯 里程碑

- [x] **v1.0** — 基于 CHECKPOINT 标记裁剪（已废弃）
- [x] **v2.0** — Always-On 3-tier 评分裁剪 + 存档功能
- [x] **v2.1** — 评分按角色分基线、Skip Cache 内容哈希、死代码清理（本地 tag `v2.1-implemented`）
- [ ] **v2.1.1** — 修正评分基线：assistant 2.0→1.5，解除 tool_call chain 死锁（❌ v2.1 实际不裁剪）。已部署运行中
- [ ] **v2.1 D 项（自适应裁剪）** — 可选延后

## 📝 决策日志

| 日期 | 决策 | 理由 |
|------|------|------|
| 2026-06-29 | 添加存档功能 (`_archive_prune_result`) | 用户要求裁剪记录可分析 |
| 2026-06-29 | 存档路径 `/opt/data/logs/context-pruner.jsonl` | 统一日志目录 |
| 2026-06-30 | v2.1 评分按角色区分基线 (assistant 2.0, tool 0.5, user 3.0) | 避免重要 assistant 回复被误裁 |
| 2026-06-30 | v2.1 Skip Cache 由 msg_count 改为内容哈希 | 原缓存逻辑两字段冗余 |
| 2026-06-30 | v2.1 删除 `_CHECKPOINT_RE` 死代码 | v2.0 不再使用 checkpoint 模式 |
| 2026-06-30 | OPTIMIZATION.md D 项延后 | 自适应裁剪非当前优先级 |
| 2026-06-30 | **v2.1.1: assistant 基线 2.0→1.5** | 实测定死锁：asst=2.0≥SCORE_KEEP→assistant 永不入候选→chain 保护锁定所有 tool 响应→裁不动。288msg session 只裁 1 条。改为 1.5 后恢复裁剪（首轮裁 64/168 条） |

## 📌 活跃事项

- [x] **[P1] v2.1 部署** — ✅ 已部署。发现评分基线 2.0 导致死锁，修复为 1.5（v2.1.1）。Gateway 已重启生效
- [ ] **[P1] 监控 v2.1.1 裁剪效果** — 观察 1-2 天，确认保留率 80-90% 稳定，不再出现上下文膨胀
- [ ] **[P2] 存档数据分析** — 已有 295+ 条记录，可对比 v2.0 vs v2.1.1 效果
- [ ] **[P3] D 项（自适应裁剪）** — 延后评估
