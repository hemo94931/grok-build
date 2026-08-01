# Responses 远程压缩连续性与缓存修复计划

## 文档状态

- 目标变更：`86f02e6 - feat: add Responses server compaction`
- 状态：已完成多轮源码级架构审查并修订，尚未开始编码
- 范围：Responses `/responses/compact`、checkpoint 回放、连续压缩、提示词连续性和 prompt cache affinity
- 安全原则：无法证明 checkpoint 可安全回放时，必须在发出 provider HTTP 请求前 fail closed

## 背景与根因

当前普通 Responses 请求把基础 System 放在 `input` 中，并且生产 request 默认：

```rust
instructions: None,
prompt_cache_key: None,
prompt_cache_options: None,
prompt_cache_retention: None,
service_tier: None,
```

`ResponsesCompactRequest::from_final` 会扫描整个 `input`，删除所有 `role == "system"` 的 item，并把内容移动到顶层 `instructions`。compact 完成后，live history 被替换为 checkpoint wrapper 和 typed tail；下一次 Responses 请求则仅把 `checkpoint.output` 拼在 typed tail 前面，没有恢复相同的顶层 prompt envelope，也没有稳定的 `prompt_cache_key`。

因此 normal、compact 和 post-compact 三种请求在控制前缀与缓存路由上不一致，造成：

1. post-compact 请求丢失基础 instructions；
2. compact 前后的 System/Memory 表示不同；
3. provider output、可信本地 prompt 和 runtime context 的信任边界不清；
4. 压缩后请求不具备稳定的 prompt-cache 命中条件；
5. 连续第二次 remote compact 仍依赖 serializer 隐式 flatten checkpoint。

禁止采用“把 `canonical_prompt_projection` 直接拼回 `checkpoint.output` 前面”的快速修复。V1 projection 来自对整个 input 的角色扫描，无法可靠区分基础 prompt、memory、中途 System 和 provider output，存在重复、错序和控制信息提升风险。

---

## 目标架构

```text
ConversationRequest / Checkpoint
              │
              ▼
     CheckpointResolver
              │
              ▼
┌───────────────────────────────────┐
│ ResponsesReplayPlan               │
│                                   │
│ Normal                            │
│ ReplayV2                          │
│ RecompactV2                       │
│ MigrateV1                         │
│ Unrecoverable                     │
└───────────────────────────────────┘
              │
              ▼
      ResolvedResponsesRequest
              │
       ┌──────┴──────┐
       ▼             ▼
 /responses      /responses/compact
 serializer        serializer
       │             │
       └──────┬──────┘
              ▼
       HTTP client gate
```

稳态下只有 `Normal`、`ReplayV2` 和 `RecompactV2` 可以生成网络请求。

`MigrateV1` 必须先完成本地历史迁移；`Unrecoverable` 禁止请求 provider。D1b 灰度期间允许一个受严格校验、禁止新建 checkpoint 的临时 `ValidatedLegacyReplayV1` 兼容路径，迁移达到 100% 后删除。

跨 crate 的安全边界不能依赖 `pub(crate)`。`xai-grok-shell` 负责读取 sidecar 和选择 ReplayPlan；`xai-grok-sampling-types` 提供字段私有、只能通过校验构造函数创建的 opaque resolved/validated 类型；`xai-grok-sampler` 只接受这些已满足结构不变量的类型或不含 checkpoint 的普通请求。

---

## 阶段 1：建立 V1 可恢复迁移基础

在改变线上 V1 回放行为前，先实现分级恢复结果：

```rust
enum V1RecoveryOutcome {
    Lossless {
        history: Vec<ConversationItem>,
        source: LosslessRecoverySource,
        original_digest: String,
    },
    LossySalvageAvailable {
        history: Vec<ConversationItem>,
        salvage_digest: String,
        omissions: Vec<SalvageOmission>,
    },
    Unrecoverable(RecoveryError),
}

enum LosslessRecoverySource {
    SidecarV2,
    SegmentStagingV1,
}
```

### 无损自动恢复

自动 V1 migration 只接受能够逐项恢复原 portable history、并匹配 live wrapper 原始 digest 的来源：

1. 读取并校验 `CompactionCheckpointFileV2`；sidecar wrapper 必须与当前 live conversation 首项的 checkpoint ID、operation ID、prompt index、digest 及其他 immutable replay fields 一致；
2. branch 使用现有 `read_checkpoint_for_wrapper` 的 rotation-aware 规则：rewind/fork 可以只轮换 live wrapper 的 `branch_id`，校验时先把 stored wrapper 的 branch 归一化为 live branch，再比较其余字段，不能要求 sidecar branch 字面相等；
3. sidecar 不可用时，读取仍存在的 `ResponsesCompactionSegmentStagingV1.items`；该 staging 只在相应 compaction mode 产生 `segment_detail` 时存在，且发布后通常会被删除；
4. V1 staging schema 没有 `operation_id`，只能以当前 live wrapper 的 `checkpoint_id` 为锚、重新计算 portable digest 并校验 prompt/system 结构，不能声称已从 staging 验证 operation ID；
5. staging 只有在当前 live wrapper 已经存在且 checkpoint ID 匹配时才能作为恢复源。CAS history commit 之前由取消、失败或 superseded 操作留下的孤儿 staging 必须忽略并可由 GC 清理；
6. 两者都不可用时，不得声称已经无损恢复。

无损结果必须校验：

- checkpoint ID；
- prompt index；
- sidecar 路径上的 operation ID；
- rotation-aware branch binding；
- 原始 portable history digest；
- live wrapper 与 sidecar/staging 的绑定；
- leading/base System 是否存在；
- typed tail 是否可以合法追加；
- history 是否满足 builtin compaction 的输入约束。

已发布的 Markdown compaction segment 不能作为精确恢复源。

### `updates.jsonl`：pre-compact 有损、checkpoint tail 可无损

当前 `helpers/replay.rs` 的 legacy `ReplayState` 主要重建 pre-compact User/Assistant 文本，不能保证保留 ToolCall、ToolResult、Reasoning、BackendToolCall、图片或其他 typed item，因此该部分不能匹配原 portable history digest，也不能进入 `Lossless` 分支。

但 checkpoint 后的 `ConversationAppendPreparedV2`/`ConversationAppendCommittedV2` journal 携带完整 `ConversationItem`，应通过 `rebuild_updates_only_v2` 或其 V3 等价实现恢复为无损 typed tail。salvage 只允许把 pre-compact 段标记为有损，不能把 journal tail 也降级成纯文本。

当 sidecar 和 staging 均不可用时，可以返回 `LossySalvageAvailable`，其组装顺序为：

```text
可信 base instructions
  + 有损 pre-compact User/Assistant 文本
  + 经 journal 校验恢复的无损 typed tail
```

必须遵守：

- 不复用旧 checkpoint output；
- 不复用旧 checkpoint identity 或原 portable digest；
- 对组合后的新历史计算 `salvage_digest`；
- 记录 pre-compact 丢失的 item 类别和 `recovery_fidelity = lossy_precompact_text` telemetry；
- journal tail 必须通过 branch、sequence、prepared/committed 配对和 conversation integrity 校验；无法闭合的 tail item 转为 omission，而不是伪称无损；
- 默认不静默自动迁移，必须由显式 repair 操作或明确部署策略确认；
- trusted base 优先来自当前 agent 渲染的 base prompt，并要求与 session `system_prompt.txt` 及持久化 prompt context canonical-equal；三者不一致时禁止自动 salvage；
- salvage 后执行 builtin compaction，并生成全新的 identity/digest。

如果无法获得可信 base instructions，或有损恢复策略未获允许，则进入 `Unrecoverable`。

主要涉及：

- `crates/codegen/xai-grok-shell/src/session/compaction.rs::portable_history_for_request`
- `crates/codegen/xai-grok-shell/src/session/helpers/replay.rs`
- `crates/codegen/xai-grok-shell/src/session/storage/responses_compaction.rs`

### 防止每 turn 无限重试

恢复状态通过新的 checkpoint recovery session update 持久化到 `updates.jsonl`，以 `checkpoint_id + operation_id` 为键，而不是写回可能已经缺失的 sidecar：

```rust
enum CheckpointRecoveryStatus {
    Pending,
    Migrating { operation_id: String },
    Migrated,
    SalvageRequired {
        salvage_digest: String,
        omissions: Vec<SalvageOmission>,
    },
    Salvaged { operation_id: String },
    Unrecoverable { reason_code: String },
}
```

要求：

- migration/salvage 使用幂等 `operation_id`；
- 同一个 checkpoint 和同一恢复源状态只自动尝试一次；
- 保存 sidecar/staging/updates 的恢复源 fingerprint；只有 fingerprint 变化或用户显式 repair 才重试；
- `SalvageRequired` 和 `Unrecoverable` 返回稳定、可诊断错误；
- 禁止每个用户 turn 都重新执行 builtin compaction。

---

## 阶段 2：V1 安全收口与 HTTP fail closed

### 停止创建新的 V1 checkpoint

远程 V1 compaction 暂时关闭，回落到 builtin compaction，避免继续产生不安全 checkpoint。

### 存量 V1 迁移

自动迁移仅消费 `V1RecoveryOutcome::Lossless`：

```text
无损恢复 portable history 并验证原 digest
  → 追加 checkpoint 后 typed tail
  → 根据 context budget 决定是否 builtin compact
  → 原子替换 live history
  → 标记 Migrated
  → 重新构造请求
```

恢复后的 history 未超过 context threshold 时，可以直接继续，避免无必要的升级期 summarization。

`LossySalvageAvailable` 不进入上述自动路径：只有显式 repair 或部署策略确认后，才使用当前可信 base instructions 和有损文本历史建立全新本地会话连续性，再运行 builtin compaction；旧 wrapper、output、identity 和 digest 全部作废。

### HTTP client 最终安全门

`conversation_stream_responses` 和 `conversation_responses` 的检查只作为第一层防御。最终 gate 必须下沉到真实 POST 点：

- `crates/codegen/xai-grok-sampler/src/client.rs::create_response`
- `crates/codegen/xai-grok-sampler/src/client.rs::create_response_stream`
- `crates/codegen/xai-grok-sampler/src/client/responses_compact.rs::compact_responses`

`CreateResponseWrapper.raw_body` 必须私有化，或替换成 sampler 内部不可任意赋值的 sealed body enum。任何公开调用方都不能直接注入任意 raw JSON 绕过 checkpoint 校验。

普通 API：

```rust
conversation_*_responses(ConversationRequest)
```

只能接受不含 checkpoint 的 normal 请求；POST 层再次验证其 body 来自 typed normal conversion。

跨 crate 不能用 `pub(crate)` 封印 resolved API。ReplayPlanner 位于 `xai-grok-shell`，发送入口位于 `xai-grok-sampler`；因此在双方共同依赖的 `xai-grok-sampling-types` 中定义字段私有的 opaque 类型：

```rust
pub struct ValidatedResponsesReplayV2 {
    // private fields; MUST NOT implement Deserialize
}

pub struct ResolvedResponsesRequest {
    // private body + correlation/trace/revision metadata
    // MUST NOT implement Deserialize
}

pub struct ResolvedCompactRequest {
    // private body + correlation/trace/revision metadata
    // MUST NOT implement Deserialize
}
```

同时定义可持久化、允许 `Deserialize`、但始终处于未校验状态的低层 material：

```rust
pub struct CheckpointReplayMaterialV2 {
    // trusted prompt envelope、wrapper/portable digests、branch、contract 等
    // private fields; deserialization does not imply validation
}
```

公开构造函数至少包括：

```rust
ResolvedResponsesRequest::try_normal(&ConversationRequest)
ResolvedCompactRequest::try_normal(&ConversationRequest, user_context)
CheckpointReplayMaterialV2::try_new(...)
ValidatedResponsesReplayV2::verify(
    &ServerResponsesCheckpointV2,
    &CheckpointReplayMaterialV2,
    &[ConversationItem], // portable history
    &ResolvedPromptEnvelope,
    &[ConversationItem], // typed tail
    history_revision,
    request_identity_generation,
)
ResolvedResponsesRequest::from_validated_replay(ValidatedResponsesReplayV2)
ResolvedCompactRequest::from_validated_recompact(ValidatedResponsesReplayV2)
```

resolved 类型必须原样携带现有发送链所需的：

- `x_grok_conv_id`、`x_grok_req_id`、`x_grok_session_id`、turn、agent、deployment、user 等 correlation fields；
- `trace: Option<Box<dyn TraceContext>>`；
- `history_revision`；
- `request_identity_generation`；
- checkpoint/branch binding。

`verify` 是纯数据校验：重新计算 wrapper/portable digest，检查 checkpoint identity、当前 compatibility envelope、branch、typed tail 和 revision binding，不读取文件系统。Shell 负责读取 sidecar、执行 `bind_request_identity_at_revision` 并在紧邻 dispatch 前确认 history revision/identity generation 仍新鲜；sampling-types 负责构造不变量；sampler 保持纯 HTTP 层。

三个真实 POST 点只接受 typed normal body、opaque resolved body或迁移期临时的 `ValidatedLegacyReplayV1`。`ValidatedLegacyReplayV1` 只能由 Shell 在 `ensure_checkpoint_replayable_for_request` 已返回 `CheckpointReplayStatus::Replayable`、并且 sidecar/live-wrapper 校验成功后签发；禁止其他辅助调用方自行构造。它仅用于 V1 migration 灰度中的非 cohort 存量会话，禁止创建新 V1，并在迁移达到 100% 后删除。

### Recap 与辅助 Responses 请求

`recap.rs` 当前直接读取 `get_conversation()`，经过 `session_recap::build_recap_items` 和 `conversation_collect` 进入 Responses；checkpoint wrapper 会被旧 serializer 隐式 flatten，且该路径不经过 `ensure_checkpoint_replayable_for_request`。新 gate 上线前必须显式改造。

选择保留 recap 功能，但永远把它转换为不含 checkpoint 的 typed normal 请求：

```text
读取当前 conversation
  → 无 checkpoint：沿用现有 recap budgeting
  → 有 V1/V2 checkpoint：校验 live wrapper + sidecar/replay material
  → 取 lossless portable history
  → 追加经 journal/active history 校验的 typed tail
  → budget_recap_items
  → 构造 typed normal ConversationRequest
```

要求：

- recap 不得使用 checkpoint.output，也不得签发 `ValidatedLegacyReplayV1` 或 `ValidatedResponsesReplayV2`；
- portable history/sidecar 无法无损读取时，recap fail closed，返回现有 unavailable UX 并记录原因，不能自动触发有损 salvage；
- recap 只读，不修改 live conversation、checkpoint identity、memory revision 或 recovery status；
- budget/trailing-tool normalization 在展开 portable history 后执行，以避免超出 recap model context；
- `recap.rs`、`session_recap.rs::build_recap_items`、`conversation_collect` 纳入 Responses 发送路径审计；
- 其他 side query 必须证明其 items 是 fresh typed normal；任何读取 live conversation 的辅助请求都要使用同一 checkpoint-aware history resolver。

本阶段只保证正确性和安全性，不设置主会话 cache hit 验收目标。

---

## 阶段 3：显式建模 prompt 和 memory 来源

### System 来源与 V1 digest 稳定性

在 `SystemItem` 中增加来源字段时，必须确保旧 V1 item 反序列化后再序列化不会新增字段，否则 `portable_history_digest`、marker 校验和 append 幂等比较都会失效：

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
enum SystemSource {
    BaseInstructions,
    MemoryContext,
    Runtime,
    #[default]
    LegacyUnclassified,
}

impl SystemSource {
    fn is_default(value: &Self) -> bool {
        matches!(value, Self::LegacyUnclassified)
    }
}

struct SystemItem {
    content: Arc<str>,

    #[serde(
        default,
        skip_serializing_if = "SystemSource::is_default"
    )]
    source: SystemSource,
}
```

旧记录反序列化为 `LegacyUnclassified`，并且 canonical JSON 中继续省略 `source`。必须用真实 V1 sidecar fixture 证明 deserialize→serialize 后 portable digest 和 wrapper/marker 比较保持不变；若无法保证，必须按 checkpoint schema 钉死 V1 digest 算法，不能改变存量 digest。

新增明确构造函数：

```rust
ConversationItem::base_instructions(...)
ConversationItem::memory_context(...)
ConversationItem::runtime_system(...)
```

普通 `ConversationItem::system(...)` 默认得到 `LegacyUnclassified`，不得自动取得 `BaseInstructions` 权限。SystemSource 写入行为只在 V1 migration 达到 100% 后启用；如果提前部署类型读取能力，也必须保持上述 digest-stable 序列化。

### 分离 memory 和 base System

修改：

- `request_builder.rs::inject_memory_reminder`
- `request_builder.rs::persist_checkpoint_memory_reminder`

新规则：

- `BaseInstructions` 独立保存；
- `MemoryContext` 独立保存；
- 更新 memory 时只替换 `SystemSource::MemoryContext`；
- 不再把 `<memory-context>` 拼到首个 System 字符串；
- 不再为 checkpoint 追加未标记的 trailing System；
- memory 更新继续使用现有 `replace_history_and_ack` 原子持久化边界，不能退化为仅修改内存。

必须同步审计所有 System 原地修改和继承路径：

- `acp_session_impl/prompt_build.rs::install_system_prompt` 插入或替换时设为 `BaseInstructions`，不能只改 `content` 而留下错误 source；
- `acp_session_impl/model_switch.rs` 与 `session_mode.rs` 更换 prompt 时保留或重新建立正确 source；
- `session_setup.rs`、spawn/resume 初始化使用明确 constructor；
- `xai-chat-state/src/conversation_util.rs::replace_or_insert_system_head` 只替换合法 base head，并同步 source；
- 全仓库 `SystemItem { ... }` 字面构造点通过编译错误机械迁移，禁止依赖遗漏后的隐式默认语义；
- `replace_conversation`、append journal 等 serde 全值 unchanged/幂等比较加入 V1 fixture 验证；
- `agent/subagent/mod.rs` 的 leading System 继承不能继续按“所有 System”计数，应只继承允许的 base/runtime preamble，默认不得把 `MemoryContext` 复制进子代理。

### Legacy memory 规范化

仅在以下条件全部满足时拆分旧 System：

- 唯一 leading System；
- memory open/close 标签完整；
- 标签外内容可无歧义视为 base prompt；
- 没有其他冲突 System；
- 内容来自本地 history，而不是 provider output。

否则 V2 remote compaction 不可用，走 builtin migration。

### V2 wire 规则

```text
instructions =
    base_instructions
    + 固定分隔模板
    + current_memory_context
```

只有 `BaseInstructions` 和 `MemoryContext` 能进入顶层 `instructions`。其他 System 保留在 `input` 原位置。

typed tail 中已被渲染到 `instructions` 的 Memory item 必须按 `SystemSource` 剔除，禁止只凭文本标签剔除。

### Memory identity 策略

- `base_instructions`：进入 checkpoint compatibility identity；
- `memory_context`：不进入 compatibility identity；
- `memory_revision`：进入 checkpoint metadata 和 telemetry；
- 当前完整 instructions：进入 exact wire prompt hash。

这意味着 memory 更新后允许继续使用旧 compact output。V2 compact 请求不会把 authored memory 放进 compactable `input`，而是放在顶层 `instructions`；provider output 即使出现相似文本也保持不透明，禁止扫描或删除。

---

## 阶段 4：实现 ReplayPlan 和合法 RecompactV2

```rust
enum ResponsesReplayPlan {
    Normal {
        prompt: ResolvedPromptEnvelope,
        transcript: Vec<ConversationItem>,
    },

    ReplayV2 {
        prompt: ResolvedPromptEnvelope,
        compacted_output: Vec<serde_json::Value>,
        typed_tail: Vec<ConversationItem>,
    },

    RecompactV2 {
        prompt: ResolvedPromptEnvelope,
        prior_output: Vec<serde_json::Value>,
        typed_tail: Vec<ConversationItem>,
        portable_history: Vec<ConversationItem>,
    },

    MigrateV1 {
        portable_history: Vec<ConversationItem>,
        reason: MigrationReason,
    },

    Unrecoverable {
        reason: RecoveryError,
    },
}
```

### ReplayV2

```text
input = opaque compact output + serialized typed tail
```

### RecompactV2

第二次 `/responses/compact` 使用：

```text
input = prior opaque compact output + serialized typed tail
```

新 checkpoint 的 portable history 必须是：

```text
previous portable history + typed tail
```

而不是 compact output。V2 identity 建议增加：

```rust
prior_checkpoint_id: Option<String>
```

### Resolved dispatch、actor 与 retry 契约

`ResponsesReplayPlan` 产出的 resolved dispatch 除 wire body 外，还必须冻结 correlation headers、trace、history revision、request identity generation 和 branch binding。首次 normal compact 使用 `ResolvedCompactRequest::try_normal`；只有已有 V2 checkpoint 的连续压缩使用 `from_validated_recompact`。

Sampler actor 管道改为显式 dispatch enum，而不是所有请求都装进 `ConversationRequest`：

```rust
enum SamplingDispatch {
    Normal(Box<ConversationRequest>),
    ResolvedResponses(Arc<ResolvedResponsesRequest>),
    LegacyV1(Arc<ValidatedLegacyReplayV1>),
}
```

`SamplerCommand::Submit`、`SamplerHandle::submit_and_collect`、`request_task::run_one_attempt` 和 completion/retry 管道均按该 enum 传递。resolved/legacy 类型必须 `Clone + Send + Sync`；`TraceContext::clone_box` 用于复制 trace，不能通过 serde 克隆。

采用 **plan-once, body-frozen retry**：

- Shell 在紧邻 `submit_and_collect` 前校验 history revision、request identity generation、branch 和 auth principal fingerprint；
- sampler 内部每次 retry 复用同一个 immutable resolved body、cache key 和 correlation metadata，不重新扫描 conversation，也不尝试读取 Shell 状态；
- auth refresh 后 principal fingerprint 改变时，不得继续 replay，返回 continuity mismatch 交由 Shell 重新规划/migrate；
- 请求被新 history、interjection、rewind 或 cancellation supersede 时取消整个 retry task，下一次提交重新生成 ReplayPlan；
- `run_one_attempt` 只对 `Normal` 应用 conversation defaults，禁止修改 resolved body。

在任何 async credential/capability 准备之后、首次提交之前，如果当前 history revision 或 request identity generation 已变化，则丢弃 resolved request 并重新规划，禁止发送 stale replay。

### Serializer 职责

改为：

```rust
FinalResponsesRequest::from_resolved(&ResolvedResponsesRequest)
ResponsesCompactRequest::from_resolved(&ResolvedCompactRequest)
```

Serializer 只负责：

- typed item 序列化；
- opaque output 拼接；
- extra tools；
- reasoning/text 修正；
- 确定性 JSON 生成。

Serializer 不能扫描 system/developer、决定 migration、读取 sidecar 或修改 identity。

不能在 `FinalResponsesRequest::try_from` 中对所有 checkpoint 一刀切拒绝，因为合法的连续 remote compact 需要显式 `RecompactV2` 分支。

---

## 阶段 5：Checkpoint V2、sidecar V3 全链路版本化

新增：

```rust
ServerResponsesCheckpointV2
CheckpointIdentityV2
TrustedPromptEnvelopeV2
CheckpointReplayMaterialV2 // 位于 xai-grok-sampling-types
CompactionCheckpointFileV3 // 位于 xai-grok-shell storage
RESPONSES_COMPACTION_CONTRACT_V2
```

为避免改变现有 V1 JSON 形状，增加独立 ConversationItem variant：

```rust
ResponsesCompactionCheckpoint(...)
ResponsesCompactionCheckpointV2(...)
```

### 读取能力先于写入能力

该阶段拆成两个发布单元：

- **5a Reader-first**：所有节点先获得 V2 wrapper、V3 sidecar、schema-3 marker 的读取、验证、resume/replay/fork/rewind、chat rebuild 和 builtin-migration 能力；V2 写入 feature flag 保持关闭；
- **5b Writer enablement**：只有确认所有可承载会话的二进制都能读取 V2 后，才允许灰度写入 V2 wrapper/marker。

旧二进制无法反序列化新的 tagged `ConversationItem` variant。因此一旦产生 V2 history，回滚目标必须是“仍具备 V2/V3 读取能力但关闭写入 flag”的新二进制，禁止回滚到完全不认识 V2 variant 的版本。

### V3 sidecar 至少包含

- V2 wrapper；
- `CheckpointReplayMaterialV2`；
- portable history；
- recovery/mode metadata。

`CheckpointReplayMaterialV2` 负责承载 trusted prompt envelope、portable history digest、wrapper digest、memory revision at compaction、current branch、prior checkpoint ID 和 contract version。它可以 `Deserialize`，但反序列化只产生 unvalidated material；sidecar 读取后仍必须调用 `ValidatedResponsesReplayV2::verify` 重新计算摘要。

### V3 segment staging

新 V2/V3 写入不再复用缺少 operation binding 的 `ResponsesCompactionSegmentStagingV1`，新增：

```rust
struct ResponsesCompactionSegmentStagingV2 {
    schema_version: u32,
    checkpoint_id: String,
    operation_id: String,
    branch_id: String,
    wrapper_digest: String,
    items: Vec<ConversationItem>,
    // summary/detail/timestamp...
}
```

V1 recovery 仍按 V1 staging 的能力边界只验证 checkpoint ID 和 portable digest；V3 crash recovery 则使用 staging V2 的 operation ID、branch 和 wrapper digest 做强绑定。

### 原子提交与 crash recovery

沿用现有范式并为 V3 明确提交顺序：

```text
durably write V3 sidecar
  → durably stage optional compaction segment
  → CAS replace live history with V2 wrapper
  → persist schema-3 marker/recovery binding
  → publish staged segment
```

恢复规则：

- CAS 前崩溃：sidecar/staging 没有对应 live wrapper，视为 orphan，不得激活，可异步 GC；
- CAS 后 marker 前崩溃：以 live wrapper + V3 sidecar 为锚修复 marker；
- marker 后 publish 前崩溃：幂等 publish segment；
- marker 存在但 live wrapper/checkpoint ID 不匹配：不得激活该 checkpoint；
- 每一步都要有 crash-injection 测试。

### Marker、replay 与 chat rebuild 同步升级

实现：

- marker schema 3；
- `marker_for_wrapper_v3`；
- `validate_marker_for_wrapper_v3`；
- `helpers/replay.rs::try_replay_responses_v3`；
- `storage/mod.rs::chat_rebuild::try_rebuild_responses_v3`；
- V3 resume；
- V3 rewind；
- V3 fork；
- V3 branch rotation；
- future schema fail closed。

replay 和 chat rebuild 必须共享一个 marker-schema dispatcher：

```text
schema 2 → V2 typed replay/rebuild
schema 3 → V3 typed replay/rebuild
未知 Responses schema → InvalidData / unsupported-reader hard error
没有 Responses server marker → 才允许 legacy reducer
```

禁止 `marker.schema_version != 2/3` 时返回 `Ok(None)` 并静默回落 legacy 文本重建。尤其 `chat_rebuild` 不得在忽略 V3 marker/tail journal 后 rename 覆盖 `chat_history.jsonl`。

V3 marker 应比较稳定标识和摘要，避免继续使用脆弱的 wrapper 全字段 JSON 相等比较；可附带 minimum reader schema/version 作为第二道诊断信息，但不能替代 hard error。

### Checkpoint variant 扇出审计

集中提供：

```rust
ConversationItem::as_responses_checkpoint()
ConversationItem::is_responses_checkpoint()
```

然后审计所有直接匹配 checkpoint variant 的位置，至少包括：

- `conversation.rs::validate_for_backend`
- `mutations.rs::bind_request_identity`，包括当前 `schema_version == 1` 和 `CheckpointIdentityV1` 绑定；
- `mutations.rs::replace_system_head`
- `responses_compaction.rs::recover_history_entries`
- history replacement/append/rewind
- `jsonl/mod.rs::fork_filter_chat`
- `helpers/replay.rs`
- `storage/mod.rs::chat_rebuild::try_rebuild_responses_v2/v3`
- `spawn.rs::find_latest_compaction_checkpoint`
- `goal_evaluator.rs`
- `acp_session_impl/laziness_classifier.rs`
- `agent/subagent/context.rs` 和 `agent/subagent/mod.rs`
- `sampler_turn.rs`
- `acp_session_impl/prompt_build.rs`
- integrity/prune/item-kind/token-accounting

该阶段不允许只更新 wrapper 和 sidecar，而遗漏 marker/replay/resume、chat rebuild 或辅助请求路径。

### Sidecar、segment GC 与 retention

每次 RecompactV2 都保存全量 portable history，磁盘占用会随 checkpoint 数量线性增长。初版必须定义引用驱动的保留策略：

- 始终保留 live wrapper、active marker、pending recovery/migration 和 retained rewind/fork branch 可达的 sidecar/segment；
- CAS 前 orphan sidecar/staging 在 grace period 后可 GC；
- branch rotation、rewind retention 或会话清理后，只删除经 `chat_history.jsonl + updates.jsonl + marker/branch index` 全量扫描确认不可达的对象；
- 绝不在 marker/journal 仍可能引用时删除 portable history；
- 记录每 session checkpoint bytes、orphan bytes、GC outcome 和 quota pressure；
- 设置保守 retention horizon 和 per-session quota，超限时停止新 remote checkpoint 并回落 builtin，而不是删除活跃恢复数据；
- content-addressed dedup/delta sidecar 可作为后续优化，不是初版正确性的依赖。

---

## 阶段 6：统一 Canonical Responses Envelope

```rust
struct CanonicalResponsesContext {
    model: String,
    base_instructions: String,
    memory_context: Option<String>,
    tools: Value,
    tool_choice: Option<Value>,
    reasoning: Option<Value>,
    text: Option<Value>,
    parallel_tool_calls: bool,

    prompt_cache_key: Option<String>,
    prompt_cache_options: Option<Value>,
    prompt_cache_retention: Option<String>,
    service_tier: Option<String>,
}
```

normal、compact、post-compact 和 recompact 均从该 context 派生。

删除 `xai-grok-sampler/src/client/responses_compact.rs::from_final` 扫描全部 `role == "system"` 并移动到 instructions 的做法。

### Compact endpoint allowlist

必须从 canonical context 同源：

- model；
- instructions；
- tools；
- reasoning；
- text；
- parallel tool calls；
- prompt cache key；
- service tier。

仅在 provider 支持时发送：

- prompt cache options；
- prompt cache retention。

可能属于 create-only、compact endpoint 不接受的字段：

- tool choice；
- response-only metadata。

即使 compact endpoint 不接受某字段，该字段仍可进入 compatibility identity，防止 checkpoint 在不兼容的后续配置下回放。

`parallel_tool_calls` 不再使用 `unwrap_or(true)`；必须由 canonical context 显式提供。

### Compact-only user context

当前 `USER_CONTEXT_DELIMITER` 会改变 compact 的完整 instructions。处理优先级：

1. provider 有独立 compaction context 字段时使用独立字段；
2. 否则作为明确的 compact-only instruction suffix；
3. 记录 `compact_directive_hash`；
4. 存在 compact-only suffix 时不承诺完整首请求 cache hit，只承诺 canonical base prefix 一致。

---

## 阶段 7：稳定缓存路由

增加独立于 typed-tail `branch_id` 的持久化 `logical_cache_namespace_id`：

```text
prompt_cache_key = hash(
    provider_id
    + normalized_base_url_or_deployment_fingerprint
    + model_cache_family
    + logical_cache_namespace_id
)
```

缓存 key 的路由指纹必须归一化到 base URL、deployment/account 和 provider，禁止包含 `/responses` 与 `/responses/compact` 的具体路径；否则 compact 与 post-compact 会得到不同 key。

| 场景 | cache key |
|---|---|
| 普通连续请求 | 相同 |
| resume 同一 logical branch | 相同 |
| `/responses/compact` | 与相邻 `/responses` 相同 |
| post-compact | 与 compact 相同 |
| `RecompactV2` | 与当前 logical branch 相同 |
| 普通 fork | 分配新的 logical cache namespace |
| subagent | 新 namespace |
| mirror fork | 默认新 namespace，即使继承了 typed-tail branch ID |
| request/turn ID | 禁止使用 |

`logical_cache_namespace_id` 在 session/branch 创建时持久化；fork/mirror/subagent 显式分配，而不是从当前 tail `branch_id` 临时推导。选择 fork 新 key 是用缓存隔离换取放弃共享前缀命中的保守策略，需在文档和 telemetry 中注明。

辅助请求不能复用主会话 namespace。为 recap 等 side query 派生稳定但隔离的 key：

```text
aux_cache_namespace = hash(
    logical_cache_namespace_id
    + auxiliary_kind // e.g. "recap"
)
```

`recap.rs` 当前直接使用 `session_id` 作为 `prompt_cache_key`，必须改为 `auxiliary_kind = "recap"` 的独立 namespace；其缓存命中和 token 统计不得计入主会话 cache-affinity SLO。

`prompt_cache_key` 不进入 checkpoint compatibility identity，但进入独立的 `cache_route_fingerprint`。

同时保持：

- auth principal；
- account/deployment routing；
- cache retention/options；
- provider capability version。

`compact_seeds_prompt_cache` 当前没有可靠静态 provider 信号时，按 normalized deployment + model 记录观测能力状态；只有明确配置或稳定观测为 true 时，第一条 post-compact 命中才是硬门槛，否则保留第二请求兜底。

---

## 测试计划

### V1 恢复

- sidecar 正常、按 rotation-aware branch 规则绑定 live wrapper 且匹配原 digest，进入 `Lossless`；
- rewind 仅轮换 live branch 时，stored sidecar branch 字面不同但其他 immutable fields 相同，仍可通过校验；
- sidecar 缺失、V1 staging 完整、按 checkpoint ID 绑定 live wrapper且匹配原 digest，进入 `Lossless`，但测试不得声称验证了 staging 中不存在的 operation ID；
- CAS commit 前取消或 superseded 操作留下的 orphan staging 被忽略并可 GC；
- sidecar/staging 均缺失时，legacy pre-compact updates 只能进入 `LossySalvageAvailable`；
- `ConversationAppendPreparedV2/CommittedV2` journal tail 按完整 `ConversationItem` 无损恢复，不能降级为文本；
- journal branch/sequence/prepared/committed 不闭合时明确列入 omissions；
- updates salvage 明确列出 ToolCall/ToolResult/Reasoning/图片等 pre-compact omissions；
- salvage 不得通过原 portable digest 校验，不得复用旧 output/identity；
- current rendered base、`system_prompt.txt` 和 prompt context 不一致时禁止自动 salvage；
- 未经显式 repair 或部署策略允许，不自动执行有损 salvage；
- 所有来源均不可用时进入 `Unrecoverable`；
- `SalvageRequired`/`Unrecoverable` 不会每 turn 重试；
- migration 和 salvage operation 均幂等；
- 恢复源 fingerprint 改变后可以安全重试；
- 无损恢复后 context 超限和未超限两种路径。

### SystemSource 与 Memory

- 真实 V1 sidecar fixture 在加入 `SystemSource` 后 deserialize→serialize 的 canonical bytes 和 portable digest 完全不变；
- `LegacyUnclassified` 默认值不序列化，显式 source 才写入 JSON；
- `install_system_prompt`、model switch、session mode、session setup、spawn/resume、`replace_or_insert_system_head` 后 source 正确；
- 所有 `SystemItem { ... }` 字面构造点均显式迁移，编译和 fixture 无遗漏；
- `replace_conversation` 和 journal 幂等比较不因默认 source 改变；
- subagent 只继承允许的 base/runtime preamble，不继承 `MemoryContext`；
- memory 更新继续通过 `replace_history_and_ack` 持久化，ack 丢失和重试保持幂等；
- normal/compact/post-compact/recompact 中 authored memory 只出现一次；
- checkpoint tail memory 不重复进入 input；
- legacy combined System 正确拆分；
- 歧义 legacy System 触发 migration；
- provider output 中伪 memory tag 不被扫描；
- memory revision 更新后 checkpoint 仍可回放，并记录预期 staleness；
- exact wire hash 随 memory 改变。

### Recompact

至少覆盖：

```text
normal
→ compact #1
→ response
→ compact #2
→ response
```

断言：

- 首次 compact 通过 `ResolvedCompactRequest::try_normal`，连续 compact 通过 validated recompact constructor；
- prior output 原样成为 recompact input 前缀；
- typed tail 顺序正确；
- base/memory 不进入 compactable input；
- portable history 保持完整；
- checkpoint token seed 和 token-seed source 更新为最新 compact 结果；
- checkpoint chain、branch 和 prior checkpoint ID 正确；
- request identity generation/history revision 绑定正确；
- 在 credential/capability async 准备期间制造 revision 变化时，请求被丢弃且 mock HTTP 计数为零；
- 不重复 system/memory。

### 版本化持久化与 crash recovery

- V1/V2 wrapper 共存；
- sidecar V2/V3 共存；
- segment staging V1/V2 共存，V2 staging 强校验 operation ID/branch/wrapper digest；
- marker schema 2/3；
- `try_replay_responses_v3` 与 `chat_rebuild::try_rebuild_responses_v3` 对同一 fixture 生成一致 typed history；
- schema 3 和未知 future Responses schema 不会返回 `Ok(None)` 回落 legacy reducer；
- chat rebuild 遇到未知 schema 时不会覆盖现有 `chat_history.jsonl`；
- reader-only、writer-disabled 的二进制可读取并迁移 V2/V3；
- V2 写入 flag 关闭时绝不产生新 V2 ConversationItem；
- resume；
- fork/mirror fork；
- rewind 到 compact 前后；
- branch rotation；
- future schema 明确拒绝；
- marker/sidecar 摘要不匹配；
- 在 sidecar write、segment stage、CAS history commit、marker write、segment publish 每一步之后注入 crash；
- CAS 前 orphan sidecar/staging 不会被激活；
- CAS 后缺 marker 可以从 live wrapper + sidecar 修复；
- marker 与 live wrapper 不匹配时 fail closed；
- 回滚测试只使用具备 V2/V3 reader 的 flag-off 二进制，明确禁止 pre-V2 reader。

### HTTP gate 与跨 crate 构造不变量

- `conversation_stream_responses` 和 `conversation_responses` 直接收到 V1/V2 wrapper 时拒绝；
- `create_response`、`create_response_stream` 和 `compact_responses` 分别尝试注入任意 raw checkpoint body 时拒绝，mock server 请求数为零；
- `CreateResponseWrapper.raw_body` 在 sampler 外不可直接赋值；
- `ResolvedResponsesRequest`、`ResolvedCompactRequest` 和 `ValidatedResponsesReplayV2` 字段不可直接构造或修改，且 compile-fail 证明它们没有 `Deserialize`；
- `ResolvedResponsesRequest::try_normal` 和 `ResolvedCompactRequest::try_normal` 收到任意 checkpoint variant 时拒绝；
- corrupted `CheckpointReplayMaterialV2` 无法通过 `verify`；
- V3 sidecar 虽能反序列化，但 digest/identity 不匹配时仍无法得到 validated token；
- correlation fields、trace、history revision 和 request identity generation 从 shell 到 sampler 完整透传；
- side query 继续走 typed normal 路径，不能构造 resolved replay；
- 临时 `ValidatedLegacyReplayV1` 只能在 `ensure_checkpoint_replayable_for_request` 判定 Replayable 后签发；
- `SamplingDispatch` 在 actor channel、retry 和 completion 路径中保持 variant，不会退化回 raw `ConversationRequest`；
- resolved dispatch 可 `Clone + Send + Sync`，trace 通过 `clone_box` 保留；
- retry 每次发送相同 frozen body；principal 改变或 supersede 时零额外 HTTP 并回到 Shell 重规划；
- sampler 的三个 POST API 只接受 typed normal、opaque resolved 或临时 legacy permit。

### Recap 与辅助请求

- 无 checkpoint recap 保持现有行为；
- V1/V2 checkpoint recap 先验证并展开 lossless portable history + typed tail，wire 中没有 checkpoint wrapper/output；
- 展开后再执行 budget 和 trailing-tool normalization；
- sidecar/replay material 损坏或仅有有损 salvage 时，recap unavailable 且 mock HTTP 计数为零；
- recap 不修改 live history、identity、memory 或 recovery status；
- recap 使用独立 auxiliary cache namespace，不复用主会话 key，缓存指标不计入主会话 SLO；
- 扫描其他读取 live conversation 的辅助请求，全部证明为 fresh typed normal 或使用同一 checkpoint-aware resolver。

### Wire parity

捕获并比较：

1. normal `/responses`；
2. `/responses/compact`；
3. 第一条 post-compact `/responses`；
4. 第二条 post-compact `/responses`；
5. recompact `/responses/compact`。

断言：

- canonical instructions 一致；
- `/responses` 与 `/responses/compact` 的 prompt cache key 字节完全相同；
- normalized cache fingerprint 不包含具体 endpoint path；
- resume 复用 logical cache namespace，fork/subagent/mirror 分配新 namespace，即使 mirror 继承 typed-tail branch ID；
- model/tools/reasoning/text/parallel 同源，`parallel_tool_calls` 不发生默认值漂移；
- prompt cache options/retention/service tier 按 provider capability 一致；
- compact-only user context 的 directive hash 和预期缓存降级可观测；
- compact allowlist 符合 provider capability；
- 没有提示词丢失或重复。

### Telemetry 要求

只记录摘要和分类，不记录原始 prompt、memory、portable history 或 provider output：

- recovery source/fidelity、source fingerprint、orphan staging、salvage omissions；
- replay plan、checkpoint/schema/contract、prior checkpoint ID；
- history revision、request identity generation、stale-dispatch abort；
- memory revision 和允许的 old-memory staleness；
- normalized cache route fingerprint、logical cache namespace 类别；
- compact directive hash；
- endpoint+model 的 `compact_seeds_prompt_cache` 观测状态；
- 第一条和第二条 post-compact cached tokens；
- migration、repair、recompact 和 crash-recovery outcome；
- recap checkpoint-resolution outcome 和 auxiliary cache namespace；
- sidecar/segment bytes、reachable/orphan counts、GC outcome 和 quota fallback。

---

## 缓存验收标准

### 客户端硬门槛

在 memory/config 未变化时，第一条 post-compact 请求必须具备：

- 相同 canonical instructions；
- 相同 cache key；
- 相同 routing/auth；
- 相同 model/tools envelope；
- compact output 原样作为 input 前缀；
- 无 compact-only directive 漂移。

这是客户端必须保证的“可命中资格”。

### Provider 门槛

如果 provider capability 表示：

```text
compact_seeds_prompt_cache = true
```

第一条 post-compact 请求必须有 cached tokens。

如果 provider 不保证 compact output 预热：

- 第一条只观测；
- 第二条相同 envelope 请求必须命中；
- telemetry 必须区分 provider 不支持和客户端前缀漂移。

---

## 实施与发布硬顺序

### D0：恢复能力 shadow

- 部署 V1 recovery scanner；
- 不改变现有请求；
- 分别统计 sidecar 无损可恢复、live-wrapper-anchored staging 无损可恢复、pre-compact 有损 salvage 候选、journal typed-tail 无损可恢复和 unrecoverable 比例；
- 不把 legacy updates 文本重建计入无损可恢复率。

### D1a：POST 层与辅助请求纯加固

- 先把 recap 改为 checkpoint-aware lossless history expansion + typed normal request，并为 auxiliary request 分配独立 cache namespace；
- 审计其他读取 live conversation 的 side query；
- 将 gate 下沉到 `create_response`、`create_response_stream` 和 `compact_responses`；
- 私有化 `raw_body`；
- 引入 `SamplingDispatch::{Normal, ResolvedResponses, LegacyV1}`，贯通 actor/retry/completion；
- 引入 normal/resolved/temporary legacy permit 类型，但不改变现有 V1 cohort 行为；
- 验证 correlation、trace、revision metadata 和 frozen-body retry 无回归。

### D1b：V1 迁移灰度

- 全局停止创建新 V1；
- 按 session hash 迁移 1% → 10% → 50% → 100%；
- 非 cohort 存量 V1 暂时只能通过受校验的 `ValidatedLegacyReplayV1` 直通，不能使用任意 raw body；
- 观察 builtin migration、salvage-required 和 unrecoverable 比例；
- 达到 100% 后删除 V1 Replayable/legacy permit 路径。

### D1c：SystemSource

- 在 V1 migration 100% 后启用 SystemSource 写入和 MemoryContext 新表示；
- 读取代码可以提前部署，但必须保持 V1 canonical JSON/digest 稳定；
- 完成 model switch、session mode、spawn/resume、subagent inheritance 和 memory persistence 审计。

### D2：ReplayPlan 与 V2 reader-first

- 实现 ReplayPlan、ReplayV2、RecompactV2 和包含 correlation/trace/revision binding 的 opaque resolved 类型；
- 全量部署 V2 wrapper、V3 sidecar、schema-3 marker 的读取、验证、resume/replay/fork/rewind/chat-rebuild 能力；
- V2 写入 flag 保持关闭；
- 用 fixtures 确认所有节点均可读取 V2/V3。

### D3：Canonical envelope 与 cache shadow

同时构造但不发送 V2 请求，比较：

- prompt envelope hash；
- typed tail hash；
- compact input hash；
- normalized cache route fingerprint；
- normal/compact/post-compact 的 key 相等性。

### D4：V2 writer/remote canary

确认 reader-first 部署完成后，再开启 V2 写入及 remote compaction：1% → 10% → 50% → 100%。重点观察：

- prompt loss/duplication；
- migration reason；
- recompact 成功率；
- crash-recovery repair；
- sidecar/segment growth、GC 和 quota fallback；
- 第一条和第二条 cached tokens；
- memory revision 变化导致的预期 miss。

### 回滚硬约束

- 禁止回滚到 V1 不安全 replay；
- 禁止回滚到不能反序列化 V2 ConversationItem/marker 的旧二进制；
- 合法回滚目标是“具备 V2/V3 reader 和 migration 能力、关闭 V2 writer/remote flags”的版本；
- V2 关闭后只能回落 builtin compaction，同时保持 V2 sidecar/marker 可读和可迁移。

---

## 完成标准

只有同时满足以下条件，才能认为修复完成：

1. 三个真实 Responses POST 点均无法接收任意 raw checkpoint body，任何 unresolved checkpoint 都不能到达 HTTP；
2. recap 和其他读取 live conversation 的辅助请求不会隐式 flatten checkpoint；checkpoint recap 只使用经校验的 lossless portable history，失败时零 HTTP；
3. V1 无损 migration 可验证原 digest 且幂等；pre-compact 有损 salvage 不会被伪装成无损恢复，journal typed tail 尽可能保持无损，失败不会无限重试；
4. V1 fixture 在加入 SystemSource 后 deserialize→serialize 的 canonical digest 保持不变；
5. V2 支持首次 compact 和连续 recompact，resolved dispatch 在 actor/retry 中保持 frozen body，并在提交前验证 revision/identity/principal 新鲜度；
6. base instructions 和 memory 在 wire 中各出现一次，subagent 不会错误继承 MemoryContext；
7. provider output 始终保持不透明；
8. normal、compact、post-compact 使用同一 canonical envelope、归一化 deployment 路由和 logical cache namespace；辅助请求使用隔离 namespace；
9. marker、sidecar、resume、fork、rewind、chat rebuild 和 crash recovery 全部支持 V3，未知 Responses schema 不会静默降级或覆盖 history；
10. V2 reader 能力已先于任何 V2 writer 全量部署，回滚不会使新 history 不可读；
11. active/retained checkpoint 不会被 GC，quota 超限只会关闭新 remote checkpoint 并回落 builtin；
12. xAI live canary 中，第一条 post-compact 请求达到 provider capability 所承诺的 cache hit 行为。
