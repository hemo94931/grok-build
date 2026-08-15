# Responses 远程压缩：compaction_trigger 迁移计划

## 状态

- Phase 1–5（sampling-types、sampler transport、shell 编排、storage/replay、
  测试重写）已实现并通过测试；review 阶段额外修复：compaction item 改从
  `response.output_item.done` 流式帧收集（真实后端 terminal 帧 output 为空，
  见 codex-backend-quirks quirk 1）、compact 请求头部恢复 turn 管线 allowlist
  并补 codex 路由集成测试。
- Phase 0（live probe）**已完成**（2026-08-14，生产 ChatGPT 路由实测）：
  trigger 契约、replay、recompact 形状全部 HTTP 200 通过；`x-codex-beta-features`/
  `x-codex-turn-metadata`/`openai-beta` 三个头均验证**非必需**（保留发送以对齐上游）；
  blob 经 `output_item.done` 到达、terminal 帧 output 为空已实测确认；`usage`
  字段齐全。详见 codex-backend-quirks.md v2 节。
- Phase 6（文档收尾）：quirks/旧计划已更新；grok 端到端 e2e **已通过**
  （`tests/responses_compaction_live_e2e.rs`，`GROK_LIVE_CODEX_E2E=1` 门控）：
  真实后端完成 seed → `/compact` → checkpoint 落盘（`kind: responses_server`，
  retained 前缀 + 真实 `cmp_…` blob）→ 后续 turn replay 被接受。e2e 发现并修复了
  一个 mock 期不可见的缺陷：retained 前缀必须丢弃携带 tool call 的 assistant 项
  （其 function_call_output 随 ToolResult 不保留，否则 replay 400），已带回归测试。

## 背景

Codex 上游已弃用 unary `POST /responses/compact`：当前代码中该端点仅为 Amazon
Bedrock 保留（`RemoteCompactionSupport::V1`），OpenAI 与 Azure-Responses 一律走
**V2 —— 基于 `compaction_trigger` 的普通流式 Responses 请求**（上游测试断言
"v2 should not call /responses/compact"）。grok-build 现有远程压缩全部建立在旧端点上，
对 ChatGPT 路由已不可用，需整体迁移。

调研依据（交叉印证一致）：

- 上游 codex @ `1bb6384c1971579c74194a3cc832847480de470d`（2026-08-14），
  关键文件：`core/src/compact_remote_v2.rs`、`core/src/compact_remote_v2_attempt.rs`、
  `protocol/src/models.rs`（`ResponseItem::CompactionTrigger` / `Compaction`）、
  `codex-api/src/common.rs`（`ResponsesApiRequest`）、`core/src/responses_metadata.rs`；
- pi 插件 pi-codex-compact v0.50.1（commit `e3203bd`，
  github.com/hemo94931/pi-codex-compact），其 `docs/codex-compaction-mechanism.md`
  为同机制的另一份独立分析；
- 本仓库现状映射见 ADR-0001 与下文变更映射。

## 决策清单（grill-with-docs 会话锁定）

| # | 决策 | 结论 |
|---|---|---|
| D1 | 旧会话兼容 | **零兼容**：该功能从未正式使用，不写迁移/双契约代码；旧 checkpoint 由结构校验 fail-closed 转 builtin continuity migration |
| D2 | 资格规则 | 不变：`server_compaction` 开关 + 走 Responses backend，与 base_url 无关；不支持的端点靠 negative capability cache 探测兜底 |
| D3 | 正交行为 | 全部冻结：触发点（auto 阈值 + `/compact`）、远程优先 + builtin 回退 + two-pass/prefire 复用、开关链（`GROK_SERVER_COMPACTION` > `[features] server_compaction` > remote setting > 默认 true）、`GROK_COMPACT_MODEL`、did-not-shrink 判定、提交顺序、checkpoint quota 与 GC |
| D4 | 交付物 | 本计划 + CONTEXT.md 词条 + ADR-0001 |
| D5 | transport 形态 | **统一进普通 Responses 流式路径**：普通请求构造 + `input` 末尾追加 trigger + 「恰好一个 compaction item」收集模式；专用 `compact_http` client 退役 |
| D6 | negative cache 分类 | 400/404/405/501/422 记入；「completed 但无恰好一个 compaction item」在重试预算耗尽后记入；流错误/超时/取消不记 |
| D7 | 契约标识 | `responses-compact-grok` 不变 |
| D8 | 保留前缀语义 | 镜像上游：user/developer/system（非 final 且不携带 tool call 的 agent 消息 ≤10k token），newest-first 截断至 64k token，compaction item 收尾 |

## 目标 wire 契约

### 请求

一次**普通流式** Responses 请求（ChatGPT 路由：
`POST https://chatgpt.com/backend-api/codex/responses`），与 turn 请求的唯一区别是
`input` 末尾追加一个裸控制 item：

```json
{"type": "compaction_trigger"}
```

body 其余字段与 turn 请求完全一致：`model`、`instructions`（base instructions +
memory 提升规则不变）、`tools`、`tool_choice: "auto"`、`parallel_tool_calls`、
`reasoning`、`store: false`、`stream: true`、`include: ["reasoning.encrypted_content"]`、
`service_tier`、`prompt_cache_key`、`text`。不使用 `previous_response_id`
（该字段仅存在于上游 WS transport）。

**压缩触发器是请求级控制项**：只存在于冻结请求体中，从不进入 history，从不持久化
（上游 rollout policy 同样不持久化 trigger）。

### 响应

SSE 流，要求：

1. 收到 `response.completed`；
2. 全部输出中**恰好一个** `{"type":"compaction"|"compaction_summary", id?, encrypted_content}`
   item，`encrypted_content` 非空（体积上限沿用现有 10 MiB，除非 probe 发现更小限制）；
3. 其他 output item 容忍并忽略。

### 历史重建与 replay/recompact

- **替换历史** = 保留前缀（retained prefix）+ 不透明压缩项收尾：
  - 保留前缀 = 压缩前 history 中的 user/developer/system 消息（非 final 的 agent
    消息须 ≤10k token 且**不携带 tool call** —— tool call 载体的
    function_call_output 随 ToolResult 不保留，replay 会 400，live 实测），**newest-first** 截断至 **64k token** 预算（镜像上游
    `RETAINED_MESSAGE_TOKEN_BUDGET`，与服务端默认一致）；
  - 不透明压缩项（opaque compaction item）作为最后一个元素；reasoning/assistant/tool
    项不保留在 typed 段，只存在于服务端加密的 blob 内。
- **后续普通请求（replay）**：`[保留前缀…, blob, typed tail…]`。
- **连续再压缩（recompact）**：先展开先前 checkpoint（含旧 blob）为输入，再末尾追加
  trigger。链式语义与现有 `from_validated_recompact` 同构。
- `usage` 字段读取不变（did-not-shrink 判定所需字段在 `response.completed` 中同样可得）。

### 头部

compaction 请求继承 turn 请求管线（D5），因此自动获得：`authorization`、
`chatgpt-account-id`、`originator`、`session-id`、`x-client-request-id`、
`x-codex-turn-state`（粘性路由 token，旧 compact 路径曾剥离，v2 下必须照发）、
关联头（`x-grok-conv-id/req-id/...`）。

上游另发：`x-codex-beta-features`（总是包含 `remote_compaction_v2`）与
`x-codex-turn-metadata: {"request_kind":"compaction", compaction:{...,
"implementation":"responses_compaction_v2","strategy":"memento"}}`。
HTTP 上**不再**设 `OpenAI-Beta: responses=experimental`。

### 待 live probe 锁定项（已于 Phase 0 实测解决，结论见 quirks v2 节）

按 provider-backend-debug 流程（`.agents/skills/provider-backend-debug/scripts/probe_params.py`
+ 反向代理抓包）对真实 ChatGPT 路由验证，结果回写
`.agents/skills/provider-backend-debug/references/codex-backend-quirks.md`：

1. `x-codex-beta-features: remote_compaction_v2` 是否为后端接受 trigger 的**必需**头 → **非必需**（已实测）；
2. `x-codex-turn-metadata` 的 `request_kind:"compaction"` 是否必需或仅为遥测 → **非必需**（已实测），保留发送以对齐上游遥测；
3. 不支持 trigger 的通用 Responses 端点的实际错误码（400 vs 422 vs completed-无-item），
   校准 D6 分类器 → codex 官方路由已实测支持；通用第三方端点按现有 turn 契约表
   （400/422 schema 拒绝）+ completed-无-item 覆盖，逐个端点无法穷举，运行时由
   negative cache 探测兜底；
4. `encrypted_content` 实际上限与 `usage` 字段位置 → blob 实测 1.7–2.5 KB（远低于 10 MiB
   上限），`usage.output_tokens/total_tokens` 在 `response.completed` 中（已实测）。

## 变更映射

1. **`xai-grok-sampler/src/client/responses_compact.rs`** —— 重写。删独立端点
   `self.endpoint("responses/compact")`、专用 `compact_http` client
   （`RESPONSES_COMPACT_CONNECT_TIMEOUT`、120s deadline、408/5xx 重试策略）与 unary
   校验器；改为在普通流式 Responses 管线上加「compaction 收集模式」：SSE 事件中收集
   恰好一个 compaction item。传输层超时/重试随普通管线（compaction 重试预算沿用 2 次，
   与上游 `MAX_REMOTE_COMPACTION_V2_STREAM_RETRIES` 对齐）。
2. **`xai-grok-sampler/src/provider_wire.rs`** —— 删 `endpoint_path()` 的
   `responses/compact` → `codex/responses/compact` 特例（L239-241）与
   `sanitize_compact_body()`（L441）；compaction 请求走普通 body 构造 + trigger 追加；
   `strip_compact_denied_headers()`（client L590）退役，头部与 turn 请求一致。
3. **`xai-grok-sampling-types/src/conversation/resolved.rs`** ——
   `ResolvedCompactRequest`（L517）冻结体改为完整流式请求形状；
   `try_normal`（首次 compact，仍拒绝 checkpoint 输入）与
   `from_validated_recompact`（连续 compact，展开已验证 checkpoint）双构造器结构保留，
   密封构造安全边界不动。trigger 仅由这两个构造器追加。
4. **`xai-grok-sampling-types/src/conversation/responses_compaction.rs`** ——
   checkpoint/sidecar 数据形状改为「retained typed 前缀 + 单 blob」；
   `RESPONSES_COMPACTION_CONTRACT`（L14）保持 `responses-compact-grok` 不变（D7）；
   `CheckpointIdentity` 绑定维度不变。旧形状数据反序列化失败即 fail-closed（D1），
   不写兼容 reader。
5. **`xai-grok-shell/src/session/compaction/responses.rs`** +
   **`session/compaction.rs` ServerFirst 分支**（L1437+）—— `prepare_server_request`
   （L193）适配新冻结体；`classify_compact_failure`
   （`responses_server_compaction.rs` L321）按 D6 扩展：`completed`-无-item 归入
   unsupported（重试预算耗尽后记入 negative cache），400/422 与 404/405/501 同列；
   流错误/超时/取消维持「回退 builtin 但不记 cache」。successor 构造适配新替换形状。
6. **`session/storage/responses_compaction.rs` + `responses_recovery.rs` +
   `jsonl/responses_recovery.rs` + `helpers/replay/responses.rs`** —— sidecar/marker/
   journal/staging/GC 机器原样保留，内部持久化形状随 (4) 改变；replay 注入由
   「opaque output 作为 compaction_summary items」改为「retained 前缀 + blob inline +
   typed tail」。digest/identity/强绑定校验规则不变。
7. **测试** —— `xai-grok-sampler/tests/responses_compact.rs`：axum mock 从 unary
   `/v1/responses/compact` 改为 SSE `/responses`，fixture 改为流式事件序列 +
   compaction item；body key-order 与头部契约断言按新形状重写，新增「trigger 在 input
   末尾」「trigger 不出现在持久化」「completed 含两个/零个 compaction item 被拒绝」用例。
   `xai-grok-shell/tests/responses_server_compaction.rs` 与
   `sampling-types/conversation/responses_compaction_tests.rs`：negative cache 新分类、
   retained 前缀截断（64k/10k 边界）、replay/recompact 形状、旧形状 fail-closed。
8. **文档** —— `docs/plans/responses-server-compaction-cache-fix.md` 标记 superseded
   （指向本计划）；`codex-backend-quirks.md` 按 probe 结果更新；
   `multi-provider-oauth-wire-cards.md` 中 `responses-compact-grok` 相关卡片同步措辞。

## 不变量（迁移前后必须同时成立）

- checkpoint quota、GC 可达性规则、sidecar→CAS→marker→typed-tail 提交顺序、
  prefire/two-pass 复用关系、did-not-shrink 基线算法、配置开关链 —— 全部冻结（D3）；
- 密封构造器安全边界：真实请求只接受 `ResolvedCompactRequest` /
  `ValidatedResponsesReplay` 产物，任何校验失败不向 provider 发送 checkpoint 内容；
- 契约 id、negative cache 键形（endpoint+model+principal+contract）不变。

## 实施顺序

1. **Phase 0 — live probe**：锁定上述 4 个待验证项，回写 quirks 参考文档；
2. **Phase 1 — sampling-types**：checkpoint 数据形状 + 流式 `ResolvedCompactRequest`；
3. **Phase 2 — sampler transport**：统一进普通流式路径 + 收集模式，退役旧 client；
4. **Phase 3 — shell 编排**：失败分类器、successor 构造；
5. **Phase 4 — storage/replay**：持久化与重建形状；
6. **Phase 5 — 测试重写 + e2e**（复用 `grok-e2e-codex-oauth-compact` 体系）；
7. **Phase 6 — 文档收尾**。

每个 phase 独立可编译、独立过测；Phase 0 的结论若有意外（如必需头集合与预期不符），
回到本计划修订后再继续。
