---
status: accepted
---

# Responses 远程压缩从 `/responses/compact` 迁移到 compaction_trigger

Codex 上游弃用了 unary `POST /responses/compact`（仅 Bedrock 保留），OpenAI/Azure
路由改用「普通流式 Responses 请求 + `input` 末尾追加 `{"type":"compaction_trigger"}`」
机制（codex @ `1bb6384`，feature `remote_compaction_v2` 已 Stable）。grok-build 的
远程压缩整体迁移到该契约：transport 统一进普通流式路径，专用 compact client 退役；
checkpoint 形状从「opaque 整段历史」改为「保留前缀 + 单个不透明压缩项收尾」。
契约 id `responses-compact-grok`、资格规则、回退链、持久化/提交机器、配置开关链
全部不变；旧会话零兼容（该功能从未正式使用，旧 checkpoint 由结构校验 fail-closed）。

理由：与上游 wire 逐字段对齐是本项目既有策略；transport 统一消除双管线的
header/路由/重试漂移（旧 compact 路径剥离 `x-codex-turn-state` 等头部在 v2 下有害）。

## Considered Options

- **保留独立 compact client，仅改打 `/responses` + SSE** —— diff 更局部，但要手工
  对齐两套 header/重试/路由行为，与上游「普通请求」语义漂移时无人兜底。拒绝。
- **契约 id 更名** —— 无生产旧会话需要区分，且项目刚完成去版本化命名；保持
  `responses-compact-grok` 不变。拒绝更名。
- **资格收缩为仅官方 ChatGPT 路由** —— per-endpoint 探测回退已覆盖通用端点，
  本次是契约迁移而非资格策略重构。拒绝。
- **保留前缀只留 user 角色（pi-codex-compact 插件的简化版）** —— 会丢失 developer
  指令上下文并与上游行为漂移；镜像上游（user/developer/system、64k token 预算）。
  拒绝。

## Consequences

- 「端点不支持」的判定从 404/405/501 扩展为：400/422 及「completed 但无恰好一个
  compaction item」（重试预算耗尽后），均记入 negative capability cache。
- 头部最小集（`x-codex-beta-features: remote_compaction_v2`、
  `x-codex-turn-metadata` 是否必需）以 live probe 结论为准，回写
  `provider-backend-debug/references/codex-backend-quirks.md`。
- `docs/plans/responses-server-compaction-cache-fix.md` 由本 ADR 与
  `docs/plans/responses-compaction-trigger-migration.md` 取代。
