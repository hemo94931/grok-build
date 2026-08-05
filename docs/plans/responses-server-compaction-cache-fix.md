# Responses 远程压缩：当前单一实现

## 状态

Responses 远程压缩现在只有一套实现，不再包含旧 checkpoint writer、reader、迁移灰度或兼容反序列化路径。

默认行为：

- Responses backend 默认优先调用 `POST /responses/compact`；
- 资格规则只看两条：`server_compaction` 开关开启 + 当前模型走 Responses backend，与 base_url 无关（自定义端点同样适用）；
- 服务端不支持、请求失败或返回无效结果时，自动回退到 builtin compaction（不支持的端点会被 negative capability cache 记住，1 小时内直接走 builtin）；
- 不需要额外的 writer 开关或会话百分比环境变量；
- 旧会话中的历史远程 checkpoint 格式不再兼容。

唯一功能开关按以下优先级解析：

1. `GROK_SERVER_COMPACTION`
2. `[features] server_compaction`（`config.toml`）
3. remote setting `server_compaction_enabled`
4. 默认值 `true`

例如显式关闭：

```toml
[features]
server_compaction = false
```

环境变量仍可作为临时覆盖：

```text
GROK_SERVER_COMPACTION=false
```

已删除的旧灰度/版本开关不会再被读取：

- `GROK_RESPONSES_V1_SERVER_COMPACTION`
- `GROK_RESPONSES_V2_SERVER_COMPACTION`
- `GROK_V1_MIGRATION_PERCENT`
- `GROK_V2_WRITER_PERCENT`

Responses backend 也不再发送 inline `x-compaction-at` /
`x-compactions-remaining` 来暗中启用另一套服务端压缩。显式关闭唯一开关后，
会直接使用 grok-build builtin compaction。

## 单一契约与类型

当前契约标识：

```text
responses-compact-grok
```

核心类型均使用无版本语义名称：

- `ConversationItem::ResponsesCompactionCheckpoint`
- `ServerResponsesCheckpoint`
- `CheckpointIdentity`
- `TrustedPromptEnvelope`
- `CheckpointReplayMaterial`
- `ValidatedResponsesReplay`
- `CompactionCheckpointFile`
- `ResponsesCompactionSegmentStaging`
- `ConversationAppendPrepared` / `ConversationAppendCommitted`

Checkpoint 持久化记录不再包含数值 `schema_version`，也不存在按版本号分派 reader/writer 的逻辑。
共享 checkpoint 目录和 marker 只使用语义 `kind` 区分当前两种用途：

- `builtin`：grok-build builtin compaction；
- `responses_server`：Responses 远程压缩。

Responses wrapper 由 `ConversationItem::ResponsesCompactionCheckpoint` 的 serde tag
确定类型；sidecar、marker 和 segment staging 均标记为 `responses_server`。缺失或
未知 `kind` 会 fail closed。带有旧 `schema_version` 字段的 Responses wrapper、
sidecar、marker 或 staging 会被拒绝，不会进入兼容 reader。

旧 serde variant、旧 contract、旧 sidecar 格式和旧 tail tag 均不提供 alias。

## 模块边界与上游隔离

Responses 实现按职责拆分，避免把本地逻辑继续堆入上游高频修改文件：

- `xai-grok-sampling-types/conversation/responses_compaction.rs`：sealed checkpoint、identity、replay 与 resolved request 契约；
- `xai-grok-sampler/client/responses_compact.rs`：`/responses/compact` transport、认证快照和响应校验；
- `xai-grok-shell/session/compaction/responses.rs`：Responses 专属请求准备、checkpoint gate、提交和 fallback；
- `xai-grok-shell/session/helpers/replay/responses.rs`：checkpoint replay；
- `xai-grok-shell/session/storage/responses_recovery.rs`：会话重建；
- `xai-grok-shell/session/storage/jsonl/responses_recovery.rs`：JSONL marker 与 journal 修复；
- `xai-grok-shell/session/storage/responses_compaction.rs`：sidecar、staging、digest 与持久化安全边界。

原有 `compaction.rs`、`helpers/replay.rs` 和 `storage/*/mod.rs` 只保留通用流程、
分派点及上游 builtin 行为。后续同步 `origin/main` 时，Responses 逻辑应优先在上述
子模块内演进；除接口接线外，避免再次扩大对这些上游热点文件的修改。

## 请求安全边界

真实 Responses POST 不接受调用方提供的任意 checkpoint JSON。

### 普通请求

无 checkpoint 的普通请求走 typed conversion。普通 POST 构造器发现 checkpoint 时会拒绝请求。

### 首次远程压缩

首次 compact 必须通过：

```text
ResolvedCompactRequest::try_normal
```

该构造器：

1. 验证输入中没有 checkpoint；
2. 按 `SystemSource` 提升基础 instructions 和 memory；
3. 生成冻结的 compact body；
4. 由 `ResponsesCompactRequest::from_resolved` 交给 transport。

### Checkpoint replay

已有 checkpoint 的下一次普通模型请求必须先完成：

1. identity 与当前 provider、endpoint、model、principal、base instructions、prompt envelope 和 cache route 的绑定；
2. live wrapper 与 sidecar 的强绑定；
3. portable history、完整 wrapper digest、branch 和 prior-checkpoint chain 的重新校验；
4. 完整 digest 同时绑定 opaque output、token seed、seed source 与 output item count；
5. `ValidatedResponsesReplay::verify` 生成不实现 `Deserialize` 的已验证值；
6. `ResolvedResponsesRequest::from_validated_replay` 冻结实际请求体。

Sampler 已移除接受 `Vec<serde_json::Value>` 的 compact 构造器；真实 compact
transport 只接受 `ResolvedCompactRequest` 转换出的私有字段请求。

任何校验失败都不会向 provider 发送 checkpoint 内容，而是转为 builtin continuity migration。

### 连续远程压缩

已有 checkpoint 再次触发 compact 时走：

```text
ResolvedCompactRequest::from_validated_recompact
```

输入为：

```text
先前 opaque compact output + 当前 typed tail
```

`prior_checkpoint_id` 将新 checkpoint 与前一个 checkpoint 绑定。Provider output 始终作为不透明数据保存，不扫描角色、不重排，也不从中提取 system 或 memory。

## Prompt envelope 与 memory

`TrustedPromptEnvelope` 显式记录：

- `base_instructions_sha256`：兼容性 identity 的一部分；
- `envelope_fingerprint`：tools、tool choice、reasoning、text、parallel tool calls、service tier 和 cache options 等非 transcript 语义；
- `wire_prompt_sha256`：完整 wire instructions 的诊断摘要；
- `memory_revision`：仅用于元数据，不使已有 checkpoint 失效。

基础 instructions 和 memory 由 `SystemSource` 区分：

- `BaseInstructions`
- `MemoryContext`
- `Runtime`
- `LegacyUnclassified`

只有 `BaseInstructions` 与 `MemoryContext` 可以提升到 Responses 顶层 `instructions`。Runtime system 信息保持在其原始语义位置，不会被提升为基础控制信息。

## Cache continuity

普通请求、远程 compact、checkpoint replay 和连续 compact 使用同一逻辑 cache route。

Checkpoint identity 保存 `cache_route_fingerprint`，但 `prompt_cache_key` 本身不进入兼容性 identity：key 是路由细节，route fingerprint 才用于检测 provider、deployment、model family 或 principal 漂移。

## 持久化与提交顺序

远程结果只有在确认压缩后 token 数严格小于压缩前 token 数时才能提交。
压缩前基线使用“最近一次 provider total + 此后新增 user/tool 的估算 delta”，
而不是可能过期的最近 provider total，因此 preflight/tool-output 场景不会误判为
`DidNotShrink`。

提交顺序固定为：

```text
持久化 sidecar
→ 可选：持久化 segment staging
→ chat-state 双 generation CAS 替换 live history
→ durable-ack checkpoint marker
→ durable-ack replacement typed-tail prepared/committed baseline
→ 发布 staged segment
→ 后台 GC
```

关键性质：

- sidecar 先于 live wrapper，避免 wrapper 指向不存在的数据；
- CAS 同时验证 history revision 和 request identity generation；
- marker 只在 live history 已成功替换后写入；
- marker 前崩溃可由 live wrapper + sidecar 修复；
- marker 不再使用“普通 update + flush”假确认，而是等待该记录自身的 durable commit 分类；
- marker 修复后会把 authoritative typed-tail baseline 重写在新 marker 之后，旧 marker 前的 journal 不会被误当成恢复依据；
- typed tail 使用 prepared/committed journal，并在 append 前锁定文件、修复唯一可能的 torn final line；
- 任何 session copy 都在复制来的旧 journal 之后写入 target marker 与完整 tail baseline；filtered/truncated fork 和 rewind 还会旋转 active branch，同一进程内再次 rewind 也可重建；
- Responses checkpoint 之后即使又发生 builtin compaction，rewind 到两次边界之间仍会选择历史 Responses marker，经 sidecar 强校验后用 typed journal 恢复 wrapper；写入 RewindMarker 后的再次 replay/resume 不会断链；
- segment staging 允许仅 branch 旋转；fork 未复制可选 segment archive 时不会阻止 checkpoint replay；
- GC 保护 live wrapper、prior chain、marker、sidecar、staging、journal 和已发布 segment 的可达对象；任何非 `NotFound` 的 prior-chain 读取错误都会使 GC fail closed；
- quota 判定前会先运行 fail-closed GC，使已过 grace period 的 pre-CAS orphan 不会永久锁死远程压缩。

## 失败与回退

远程路径出现以下情况时回退 builtin：

- endpoint 返回 `404`、`405` 或 `501`；
- auth、quota、rate limit、timeout、transport 或服务端错误；
- compact 请求或响应超过限制；
- 响应结构无效；
- token seed 无法计算；
- compact output 没有使会话严格缩小；
- checkpoint identity 或 continuity 校验不再匹配；
- session checkpoint quota 超限。

不支持的 endpoint 会写入进程内 negative capability cache，TTL 内直接回退 builtin，避免每次压缩重复请求同一不支持的接口。

## 与 two-pass / prefire 的兼容方式

远程压缩和 builtin two-pass 不是互斥实现，而是“远程优先、prefire 作为可复用回退工作”的关系。

### Prefire 输入

Prefire 在达到 `auto_compact_threshold - lead_percent` 时提前启动 pass 1。

- 普通会话直接使用 typed history；
- checkpoint 会话先通过 `portable_history_for_request` 展开 sidecar portable history，再附加 typed tail；
- pass 1 与 builtin pass 2 对同一展开结果计算 prefix length 和 fingerprint；
- misplaced 或重复 checkpoint 会 fail closed，不启动 prefire。

因此 checkpoint wrapper 本身不会进入 builtin summarizer，也不会再出现 pass 1 使用展开 history、pass 2 却使用 wrapper 导致的 fingerprint 假失效。

### 远程失败

Prefire cache 在远程请求期间保持可用。以下结果不会清除 NOTE1：

- unsupported / HTTP / transport / timeout；
- request build failure；
- response validation failure；
- token seed 表明没有缩小；
- 最终 committed token 检查表明没有缩小。

控制流会继续进入 builtin fallback，`try_two_pass_pass2_apply` 可以消费 NOTE1；若 cache 已过期、model 变化或 fingerprint 不匹配，再退化为 single-pass builtin。

### 远程成功

只有远程输出同时通过：

1. token seed shrink 校验；
2. 最终 replacement committed-token shrink 校验；
3. cancellation 检查；
4. sidecar/staging 持久化；
5. 双 generation CAS 成功安装 replacement；

才会调用 `discard_prefire`。因此无效响应、pre-CAS 持久化失败、取消或 CAS
superseded 都不会提前丢失 NOTE1；只有已经提交的远程 successor 才使它失效。

### 取消

等待后台 prefire handle 和执行 pass 2 时都监听当前 turn cancellation。取消会 abort 未完成的 prefire，并阻止继续提交远程或 builtin 结果。特别地，pass 2 因取消返回后会再次检查 cancellation，不会把 `None` 当作 cache miss 再启动一次 single-pass 付费请求。

### 配置关系

`two_pass_compaction` 仍是独立功能，默认关闭；开启方式：

```toml
[features]
two_pass_compaction = true
```

远程压缩默认开启并不要求 two-pass 开启。组合行为为：

| 远程压缩 | two-pass | 行为 |
|---|---|---|
| 开 | 关 | 远程优先，失败后 single-pass builtin |
| 开 | 开 | 远程优先，失败后优先复用 prefire 执行 builtin pass 2 |
| 关 | 关 | single-pass builtin |
| 关 | 开 | builtin two-pass/prefire |

## 验证范围

测试覆盖：

- checkpoint serde 使用无版本 tag；
- 普通请求拒绝 checkpoint；
- replay/recompact 的 resolved constructor，以及 raw compact constructor 已移除；
- prompt envelope、memory drift 和 cache route identity；
- sidecar digest、opaque output/token-seed 完整绑定、branch rotation 和 prior chain；
- marker durable acknowledgement、repair ordering、chat rebuild、resume、fork、rewind；
- replacement typed-tail baseline、prepared/committed 幂等、sequence gap 和 torn-tail healing；
- checkpoint quota 和 GC；
- remote failure classification 与 negative capability cache；
- did-not-shrink 时保留 prefire；
- checkpoint portable expansion 在 prefire pass 1/pass 2 间保持一致；
- builtin `kind = "builtin"` 路径与 Responses marker 分派互不混用。
