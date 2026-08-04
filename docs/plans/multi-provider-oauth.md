# 多 Provider OAuth 集成：任务计划

## 状态

计划已经 gpt-5.6-sol(max) review，并按当前单一 Responses 远程压缩实现重新修订，待实施。Phase 0（影响面清点）已完成。

核心原则：

1. **xAI 行为不变**。现有 xAI OAuth 认证（`xai-grok-shell/src/auth/`）、`xai-grok-auth` 接缝 crate、遥测、first-party 功能门控（`is_xai_auth` 等），以及 `responses-compact-grok` checkpoint/replay 行为均保持不动。
2. **降低与 `origin/main` 的耦合**。新增实现优先落在新文件和现有 Responses 隔离子模块；上游高频文件只保留最薄的注册/分派接线，避免把 provider 分支散入 xAI 主路径，降低后续同步冲突。

参照系：pi agent 的 OAuth 架构（`@earendil-works/pi-ai` 的 `dist/auth/` 与 provider api 模块，TypeScript 已编译可读）。所有 wire 细节以该源码为准逐文件移植，不凭印象；结构组织则以当前仓库的低耦合边界为准，不机械复制 pi 的文件布局。

## 目标

1. 从 pi 移植 xAI 以外的 6 家 OAuth provider：anthropic、openai-codex、github-copilot、openrouter、kimi-coding、radius。
2. `grok login`/`logout` CLI 与 TUI `/login`、`/logout` 加入 pi 式渠道选择步骤。
3. 模型路由集成：6 家凭证可真正驱动模型请求；复用现有 Messages/Responses/ChatCompletions 转换与流处理，并以新增 adapter 隔离各家 endpoint/body/header/error 方言（Radius 另有 pi-messages adapter）。

## 已确认决策

| 决策点 | 结论 |
|---|---|
| xAI OAuth 现有实现 | 保持不变；新增代码不得改变 xAI 路径可观察行为 |
| 移植范围 | pi 的全部 6 家非 xAI provider（含 Radius 新 api adapter、GitHub Enterprise 域名支持） |
| 旧 auth.json | 不动；6 家凭证存独立的 `~/.grok/providers.json` |
| 模型路由 | 纳入计划 |
| 6 家 API key 录入 | 不做交互录入，走 env 变量（codex 无 API key 路径） |
| ACP 登录协议 | **不复用**全局 authenticate 路径；新增 provider 作用域 RPC（见“关键设计约束”） |
| providers.json 并发 | **refresh/写删必须文件锁内 RMW**（见“关键设计约束”） |
| 上游同步策略 | 新文件承载实现；不为 provider 能力向 `SamplerConfig`、chat-state `SamplingConfig`、`ConversationRequest`、`ModelInfo` 等广泛传播的上游类型加字段，除非 W0 spike 证明没有更窄接缝 |
| Responses 远程压缩 | 全局开关保持全局；当前 route 能力独立判定。6 家第三方及未知自定义 route 默认均不支持 `responses-compact-grok` |
| NegativeCapabilityCache | 只保留为“声明支持但部署返回 404/405/501”后的被动兜底，不承担第三方 provider 的主动门控 |

## 关键设计约束（review 必改项）

### 1. providers.json 并发：原子写不够，refresh 必须锁内 RMW

refresh token 轮换（anthropic/codex）与 Copilot access-token 再签发并发下，双进程 read-modify-write 会造成整表互相覆盖；对会轮换 RT 的 provider 还会双耗旧 RT，导致服务端会话失效。因此：

- `get()` 预刷新路径：进程内锁即可。
- **写删/refresh：直接复用现有 `auth/manager/lock.rs::try_lock_auth_file_async`，以 `auth.json.lock` 作为所有凭证文件的共享锁；不泛化、不复制现有锁实现。**持锁后对 `providers.json` 执行完整 RMW：重读 → 5 分钟双检（可能已被兄弟进程刷新）→ 刷新 → 写回。
- 共享锁会串行化 xAI 与第三方凭证写入，但登录/refresh 是低频路径；只有实测锁竞争成为瓶颈时，才把现有锁抽成参数化通用模块。
- Windows 下 remove+rename 存在空窗（参考现有 `storage.rs` 写路径处理），`providers.json` 的原子写封装留在新增 `store.rs`：同目录 temp → flush/sync → rename。
- 读路径（非刷新）保持无锁；现有 xAI store/lock 代码和行为不改。

### 2. ACP：provider 登录走独立 RPC，不得触碰全局 `auth_method_id`

`set_auth_method`（`agent_ops.rs:158`）把 ACP auth method 发布到共享 live handle，**所有运行中会话的 per-turn auth 门控都会观察它**。若 6 家复用标准 authenticate 路径，全局身份会被改成 "anthropic" 等，xAI 会话门控被污染。

- 新增 provider 作用域 RPC：`x.ai/providerAuth/login` / `logout` / `info`（带 `provider` 参数），不触碰 `auth_method_id` 与全局 xAI 刷新逻辑。
- 授权 URL、device code、手动粘贴、取消仍复用现有单飞通道（`get_url`/`submit_code`/`cancel` 语义等价物挂在 provider RPC 下）。
- 现有 `x.ai/auth/*` 方法保持 xAI 专用，行为不变。

### 3. Provider 请求上下文 + 严禁凭证回落

第三方模型在新增 provider 模块内解析为 shell-local `ProviderRequestContext`，至少包含 **provider_id、token、scheme、base_url、headers、api_backend、wire dialect、capabilities**。copilot 的 base_url 是 per-credential 的，codex 的 account ID 和部分头部也是动态的；上下文必须每 turn / 401 后重建。

该上下文只在 provider 路由器、新增 wire adapter 和 Responses 能力接缝间传递，**不沿整个上游调用链给 `SamplerConfig`、chat-state `SamplingConfig`、`ConversationRequest` 或共享 `ModelInfo` 加 provider 字段**。模型使用 namespaced catalog ID 或 provider 模块的旁路索引关联 route，wire adapter 再还原真实上游 model ID；W0 spike 选择改动面最小且能保持 per-session 正确性的方案。

**第三方模型缺失对应凭证时，严禁沿 `resolve_credentials`（`agent/config.rs`）现有优先级链回落到 xAI session bearer**——那会把 xAI token 发到 anthropic 等第三方端点。第三方模型的解析链：BYOK（模型自有 key）→ 该 provider 的 `providers.json` 凭证 → 该 provider 的 env key → 失败。401 只刷新当前 provider，随后重建完整请求上下文。

### 4. 外发请求净化

第三方 wire adapter 采用最终 allowlist；所有 header 来源（静态、env、动态 provider header、trace injector）合并后再净化。非 xAI route 禁止携带：

- 所有 `x-grok-*` / xAI URL 派生认证头；
- `x-compaction-at`、`x-compactions-remaining`、doom 相关头；
- `x_search`、其他 xAI hosted capability 和 xAI-only body 字段；
- `POST /responses/compact` 及 `ResponsesCompactionCheckpoint`。

provider 特有头只由对应 adapter 写入。`provider = None` 的自定义端点必须做 URL 校验，并按第三方安全默认值处理：无 xAI 头、无 xAI capability、无远程压缩。

### 5. 上游冲突预算：新增模块承载实现，热点文件只接线

- 不重构现有 xAI auth/store/lock；共享现有 `auth.json.lock`，第三方存储逻辑全部放在新增 `auth/providers/store.rs`。
- 不把 6 家分支写进 `xai-grok-sampler/src/client.rs`、`request_task.rs` 或 `agent/config.rs` 主流程；各家 wire 放新增 adapter 文件，热点文件最多增加一次统一 dispatch。
- TUI provider 交互放新增 reducer/helper 模块，现有 pager action/router/slash-command 只增加注册与转发。
- Responses 改动限定在现有隔离边界 `session/compaction/responses.rs`，以及 `session/compaction.rs` 中最多一个候选/telemetry 接缝；不修改 checkpoint 类型、transport、storage、replay、GC 和 persistence 逻辑。
- `resolve_server_compaction_enabled` 继续只解析全局开关；主会话和 subagent 共用 SessionActor 的 route gate，不在 `agent/subagent/mod.rs` 再复制 provider 分支。
- 每个阶段记录“新增文件 / 修改上游文件”清单，并对最新 `origin/main` 做 merge-tree/rebase 演练；出现冲突时先收窄接缝，不把冲突留到最终集成。

## 总体架构

```
xai-grok-shell/src/auth/           ← 现有 xAI 认证（不动）
  └── providers/                   ← 新增
      ├── mod.rs                   ProviderId 枚举 + 注册表（显示名、流类型、能力位）
      ├── store.rs                 ~/.grok/providers.json 读写 + 复用 auth.json.lock 的锁内 RMW
      ├── route.rs                 shell-local ProviderRequestContext + capability/adapter 路由
      ├── flow.rs                  通用 PKCE loopback 驱动 + 通用 device code poll
      ├── anthropic.rs             PKCE loopback
      ├── openai_codex.rs          PKCE（固定 localhost:1455）+ device code 回退
      ├── github_copilot.rs        device code（+ GHE 域名输入）+ per-credential base_url
      ├── openrouter.rs            PKCE loopback（无 refresh token）
      ├── kimi_coding.rs           device code
      └── radius.rs                PKCE + device code

xai-grok-sampler/src/provider_wire/  ← 新增；复用协议转换，隔离 provider wire 方言
  ├── mod.rs                        单一 dispatch + 最终净化
  ├── anthropic.rs
  ├── openai_codex.rs
  ├── github_copilot.rs
  ├── openrouter.rs
  ├── kimi_coding.rs
  └── radius.rs
```

ACP provider RPC 与 pager 交互同样各放一个新增模块；现有 handler/router 只注册该模块，不承载 provider 状态机。

### 存储（store.rs）

- 路径 `~/.grok/providers.json`，权限 0600，独立于 auth.json（auth.json 是 `BTreeMap<String, GrokAuth>`，混入异构值会破坏反序列化；`hub_auth.rs` 直接读它，不能受干扰）。
- 值格式对齐 pi：`{"type":"oauth","access":"...","refresh":"...","expires":123}` + provider 特有字段（copilot 的 `base_url`）。**openrouter 无 refresh token**：该 provider 不进预刷新路径。
- 并发语义见“关键设计约束 1”；不新增 provider 专用锁实现。

### 各 provider wire 事实（已与 pi-ai 源码核对）

| provider | api_backend / dialect | base_url / 端点 | 鉴权 | 远程压缩 / grok checkpoint | 特有行为 |
|---|---|---|---|---|---|
| anthropic | Messages / anthropic | api.anthropic.com `/v1/messages` | Bearer + `anthropic-version` + `anthropic-beta` oauth 头 | 否 / 不接受 | system 字段单独处理 |
| openai-codex | Responses / codex | chatgpt.com/backend-api `/codex/responses` | Bearer + `chatgpt-account-id` | **否 / 不接受** | `store:false`；不用 `previous_response_id` 链；reasoning 密文 include；无 API key 路径 |
| github-copilot | 按模型三协议（CC/Responses/Messages） | per-credential 动态 proxy base（存于凭证） | Bearer + editor 头 | 否 / 不接受（含 Responses 模型） | 模型 ID 过滤；支持 GHE 域名 |
| openrouter | ChatCompletions / OpenAI | openrouter.ai/api/v1 | Bearer | 否 / 不接受 | **无 refresh token** |
| kimi-coding | Messages / anthropic-compatible | `https://api.kimi.com/coding` | Bearer + 特定 UA | 否 / 不接受 | — |
| radius | pi-messages（**需新 adapter**） | `gatewayConfig.baseUrl` | Bearer | 否 / 不接受 | 双流程；模型目录来自 `/v1/config` |

## 与 xAI OAuth 耦合功能的隔离边界与回归

以下功能硬依赖 first-party 判定，本计划不改变其 xAI 认证和门控：云沙箱、billing、share_session、managed MCP、远程会话同步、workspace/computer-use、feedback/session_registry。现有 xAI 路径须由 regression/golden 测试证明行为不变，避免用“理论上不受影响”代替验收。

web_search/image_gen/video_gen、voice STT、memory embedding、remote_settings 等独立 xAI 调用仍固定取 xAI 凭证，与主会话 provider 解耦；无 xAI 凭证时 Disabled 是正确行为。但这些能力不得以 hosted tool、header 或 bearer 的形式进入第三方模型主请求；尤其非 xAI route 禁止 `x_search`。

## 阶段任务（纵切：风险最高的 wire 与并发前置）

### Phase 0: 影响面清点（已完成）

### Phase W0: wire 核实 + 低耦合 spike + 最小测试基座

- 逐家精读 `pi-ai/dist/auth/oauth/<p>.js` 与 api 模块，产出每家的 wire 事实卡（endpoint、headers、请求体、refresh 语义、错误格式）。
- 产出 route capability、checkpoint compatibility、最终 header/body allowlist 三张矩阵；明确“Responses 协议族 ≠ 支持 `responses-compact-grok`”。
- 做最小 spike：证明 provider route 可在 shell-local sidecar/旁路索引中解析并到达请求 adapter 与 `session/compaction/responses.rs`，不向广泛使用的上游 config/request 类型加字段；同时覆盖 per-session model switch，不能依赖共享全局 current-model 状态。
- 复用现有 sampler/shell mock server 与断言 helper，只补 provider wire 所缺的薄封装；不先造通用录制框架。
- 建立改动面基线：列出不可避免的上游接缝，并对当前 `origin/main` 做一次 merge-tree 演练。

verify: 6 张 wire 事实卡和 3 张共享矩阵评审通过；route spike 覆盖主会话 + subagent；一个示例 provider 跑通；无广泛上游类型字段扩散。

### Phase W1: providers 基础设施（store + flow + route）

- 新增 `store.rs`：复用现有 `auth.json.lock`，按“关键设计约束 1”实现锁内 RMW、5min 双检和 Windows 安全写回；不新增/泛化 lock 模块。
- 新增 `flow.rs`：通用 PKCE loopback（参数化端口 + 手动粘贴兜底）+ 通用 device code poll。
- `oidc/protocol.rs` 仅做 `Pkce`/`generate_pkce` 的最小可见性放宽；不重构现有 xAI OIDC flow。
- 新增 `route.rs`：provider 注册默认值、shell-local 请求上下文、namespaced model 映射和 fail-closed capability；不接入 wire 前先完成纯函数测试。

verify: 双进程并发 refresh 下旋转 RT 只被消耗一次、无 key 丢失；xAI auth lock 测试原样通过；本阶段实现主要位于新增文件。

### Phase W2: anthropic + github-copilot 端到端（最复杂两家先行）

- 两家 OAuth（login/refresh）+ store 接入 + `ProviderRequestContext` + Messages/CC wire adapter + golden 测试。
- adapter 各自放新增文件；generic sampler 只增加一次统一 dispatch，不在 `client.rs` 各请求函数散布 provider 判断。
- 含 GHE 域名输入、copilot 动态 base_url/模型 ID 过滤和三协议 route；所有 Copilot route 均禁用 grok remote compaction/checkpoint。

verify: 两家 mock 端到端流式请求（含 tool_use、usage）+ 401 只刷新当前 provider 并重建 route + 最终 wire 净化；xAI sampler golden 不变。

### Phase W3: 其余 4 家 OAuth（codex / openrouter / kimi / radius）

- OAuth、store 和 route 都落在新增 provider 模块；ACP 只注册一个 provider-scoped handler，不扩展全局 xAI authenticate 状态机。

verify: 各家 mock endpoint 合同测试（授权 URL 参数、exchange body、refresh、错误路径；openrouter 无 refresh 分支）；登录/取消/完成均不改变全局 `auth_method_id`。

### Phase W4: 其余 wire（codex Responses dialect + radius adapter）

- codex 独立 adapter：`POST /codex/responses`、`store:false`、`includeSystemPrompt:false`、无 `previous_response_id`、`chatgpt-account-id`、reasoning 密文 include；正常连续性来自本地 transcript，不接受 grok checkpoint。
- radius：新增 pi-messages adapter + `/v1/config` 动态目录。
- `ProviderRequestContext` 覆盖全部 6 家；在真实第三方请求首次可达前完成关键设计约束 4 的最终 wire 净化，不能推迟到 UI 阶段。

verify: 各家 golden 流式请求；静态/env/injector 均无法重新注入禁用头；第三方 bearer/header 不串 provider；xAI 普通 Responses wire 不变。

### Phase W5: Responses route gate + picker/登录 UI

**远程压缩（codex 必做）**：当前远程压缩是独立的 grok 契约（`POST /responses/compact`，contract=`responses-compact-grok`），实现已隔离在 `xai-grok-sampler/src/client/responses_compact.rs`、`xai-grok-shell/src/session/compaction/responses.rs` 及专属 storage/replay 模块。多 provider 接入不得重新揉回通用 compaction 主文件。

- `resolve_server_compaction_enabled` 和 `CompactionPolicy.server_compaction` **保持纯全局开关**；不向 `agent/subagent/mod.rs` 加 provider 维度。
- provider 注册表只给默认值，最终 `supports_remote_compaction` / `accepts_responses_checkpoint` 由当前 shell-local route 决定，未知/custom 默认 false。当前只有受支持的 xAI route 为 true；6 家第三方均 false。
- 在 `session/compaction/responses.rs::prepare_server_request` 最前面做 route gate；不支持时不构造 compact request/client、不查询 NegativeCapabilityCache、不发 POST，直接走既有 builtin 路径。`session/compaction.rs` 最多增加一个薄 eligibility/telemetry 接缝用于记录 `unsupported_route`。
- `ensure_checkpoint_replayable_for_request` 独立检查当前 route 是否接受 wrapper 的 `responses-compact-grok` contract：不接受时从已强校验 sidecar 的 portable history 执行既有 builtin continuity migration，再 resubmit；codex 永远看不到 opaque xAI compact output。
- provider/model/endpoint/principal/prompt/cache route 漂移继续使用现有 `CheckpointIdentity` fail closed；remote route gate 之后 identity 的 provider 固定为真实 xAI，不再生成笼统的 `openai_compatible` checkpoint identity。
- 全局开关关闭只禁止创建/连续创建新的远程 checkpoint；同一兼容 xAI route 上已有 checkpoint 的普通 replay 仍可继续，下一次压缩再迁移到 builtin。
- NegativeCapabilityCache 不改：只测试 capability=true 的 endpoint 实际返回 404/405/501 后首次记录、TTL 内抑制；codex capability=false 的首个请求也必须是零 compact POST。
- 不修改 `responses_compact.rs` transport、checkpoint types、storage、replay、GC、persistence 与 durable commit 顺序；现有完整回归套件必须继续通过。

**登录/登出渠道选择**：
- 协议层：按“关键设计约束 2”新增一个 provider-scoped RPC 模块；现有 `x.ai/auth/*` 不改。provider 登录态由新 `x.ai/providerAuth/info` 返回，不扩写 xAI auth response 的核心类型。
- TUI：provider selector/交互状态放新增模块，复用现有 list/question 组件；slash command、action router、task-result 只做薄转发。`/login <name>` 可跳过选择；401 re-auth 直接针对发生 401 的 provider；非 xAI 完成态不进入 xAI 订阅/paywall/billing/ZDR/会话启动逻辑。
- CLI：`grok login` TTY 下编号选择（不引入新依赖）；非 TTY 缺省 xAI；`--provider/--all`；`--device-auth`/`--oauth` 对各 provider 强制传输方式。
- picker：provider 模型使用 namespaced catalog ID/旁路索引分组和判定可见性，尽量不修改共享 `ModelInfo`；现有 xAI `visible_for_auth` 保持不动，由 picker 投影层合并 provider 模型。

verify: 开关 × backend × route capability 真值表；xAI checkpoint → codex 先 builtin migration 再普通 `/codex/responses`，compact POST=0 且 opaque output 不出站；各家 e2e 登录/登出；picker 可见性回归；主会话和 subagent 走同一 gate。

### Phase W6: 逐家语义 + xAI 回归 + 上游同步演练

- 429/Retry-After/重试逐家校验；provider 差异留在新增 adapter，不在 `request_task.rs` 堆 provider `match`。usage 汇总、cache route 和 codex `prompt_cache_key` 语义分别校验。
- 遥测降级：非 xAI 请求不带 xAI 专属 OTel 属性；provider 遥测通过一个统一接缝写入。
- 主会话/subagent 对称回归：凭证、能力、header、401 refresh 均不串 route。
- 运行当前 Responses checkpoint/sidecar/replay/resume/fork/rewind/GC/prefire 全套回归，证明 W5 只增加 route gate。
- workspace `cargo test` 全绿 + xAI golden；定向测试先行，build/live e2e 只作 smoke（不进 CI）。
- 对最新 `origin/main` 做 merge-tree 或临时 rebase 演练；按文件报告冲突。若 provider 代码与上游热点发生大块冲突，先把实现再移入新增模块，而不是记录为“后续处理”。
- 文档：`docs/user-guide/02-authentication.md` 多 provider 章节（双文件分工、安全属性、env key）；`CONTEXT.md` 同步。

## 工作量估算

含测试 **9–14k 行**仅作为粗略上限，不是验收目标。主要代码应位于新增模块；允许删除被统一 adapter 取代的临时接线。比总行数更重要的验收指标是：共享上游类型无 provider 字段扩散、热点文件只有薄 dispatch、对最新 `origin/main` 的演练冲突可控。

## 风险与注意点

1. **pi 是 TypeScript**：所有移植以 `pi-ai/dist/` 可读 JS 为准，逐文件对照（Phase W0 的事实卡是后续阶段的输入）。
2. **RT 轮换并发**：见关键设计约束 1，双进程断言是 W1 验收红线。
3. **全局 ACP 身份污染**：见关键设计约束 2。
4. **凭证回落泄漏**：见关键设计约束 3，第三方模型严禁回落 xAI bearer。
5. **codex 固定端口 1455**：被占用时清晰报错并提示 `--device-auth` 回退。
6. **copilot base_url per-credential**：存进凭证、构造请求时读出，不能写死；GHE 域名需用户输入。
7. **openrouter 无 refresh token**：不进预刷新路径。
8. **远程压缩是 route capability，不是 Responses backend 固有能力**：见 Phase W5；codex/Copilot Responses 和 custom route 默认均 false，不能靠首次 404 探测。
9. **checkpoint 跨 route**：不兼容 route 必须先 builtin continuity migration；任何 opaque provider output 都不得未经既有强校验出站。
10. **外发请求净化顺序**：必须在静态/env/dynamic/injector 全部合并后执行，防止被后置 header source 绕过。
11. **共享 auth 锁的吞吐上限**：登录/refresh 被串行化是有意的低耦合取舍；只有出现实测竞争再泛化锁。
12. **上游同步冲突**：`client.rs`、`request_task.rs`、`agent/config.rs`、`subagent/mod.rs`、pager reducer 和通用 compaction/storage 文件均视为热点，新增逻辑不得在其中展开。
13. **auth.json 与 providers.json 双文件**：文档中讲清分工；共享锁不等于混合文件格式。
