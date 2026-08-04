# 多 Provider OAuth：wire 事实卡（Phase W0 产出）

来源：`@earendil-works/pi-ai/dist/auth/oauth/*.js` 与 `dist/api/*`（已编译未混淆 JS，逐行核实），以及当前仓库的单一 Responses 远程压缩实现：

- `xai-grok-sampler/src/client/responses_compact.rs`
- `xai-grok-shell/src/session/compaction/responses.rs`
- `xai-grok-shell/src/session/responses_server_compaction.rs`
- `xai-grok-sampling-types/src/conversation/responses_compaction.rs`

实现时以这些 wire/contract 事实为准，如需变更需同步更新本卡。

## 实现边界（低 `origin/main` 耦合）

- provider OAuth、store、route 和各家 wire adapter 以新增文件承载；现有 xAI auth 与请求路径保持默认分支。
- 不为 provider 信息向 `SamplerConfig`、chat-state `SamplingConfig`、`ConversationRequest`、`ModelInfo` 等广泛使用的上游类型加字段；shell-local route 通过 namespaced model/旁路索引在现有接缝按需解析。
- generic sampler、ACP 和 pager 热点文件最多保留一次统一注册/dispatch；不得散布六组 provider 条件分支。
- `providers.json` 的写删/refresh 直接共享现有 `auth.json.lock`，不新增或泛化锁实现。
- Responses 适配只在 `session/compaction/responses.rs` 增加 route gate/continuity migration 接缝；不改当前 checkpoint type、transport、storage、replay、GC 和 durable commit 实现。
- 每阶段对最新 `origin/main` 做 merge-tree/rebase 演练；若接线形成大块冲突，先把逻辑移回新增模块。

## 共享组件

### PKCE（`pkce.js`）
- verifier：32 随机字节 → base64url（无 padding）
- challenge：SHA-256(verifier) → base64url
- 与现有 `oidc/protocol.rs generate_pkce` 等价，直接复用

### Device code 轮询（`device-code.js`，RFC 8628）
- 最小间隔 1s；未提供 interval 时默认 5s
- `slow_down`：用服务端 `interval`（若给出）否则 +5s（RFC 8628 §3.5）；连续 slow_down 超时给专门提示（WSL/VM 时钟漂移）
- `waitBeforeFirstPoll` 选项（copilot/kimi 用）
- 截止时间 = now + expiresInSeconds；支持取消（signal）
- 超时消息区分：普通超时 vs slow_down 超时

### ProviderRequestContext / wire adapter

每次 turn 和 401 后在新增 provider 模块重建 shell-local 请求上下文：

```text
provider_id + credential/principal + auth scheme + base_url + provider headers
+ api_backend + wire dialect + route capabilities
```

- 注册表只提供默认值；动态 credential/base URL（尤其 Copilot、Radius）参与最终 route。
- 6 家模型缺少自己的 credential 时本地失败，绝不回落 xAI session bearer。
- provider-specific body/header/错误语义留在新增 adapter；generic transport 只做一次分派。
- `provider=None` / 未知自定义 endpoint 默认第三方安全策略：无 xAI 头、无 hosted xAI capability、无远程压缩、无 grok checkpoint replay。

### Responses 远程压缩 / checkpoint continuity

当前唯一 contract：

```text
responses-compact-grok
```

全局 `server_compaction` 开关只决定是否创建/连续创建远程 checkpoint；route 能力独立决定当前 endpoint 是否支持 contract。

| route | supports_remote_compaction | accepts_responses_checkpoint |
|---|---:|---:|
| 当前受支持的 xAI Responses route | true | true |
| openai-codex | false | false |
| github-copilot（含 Responses 模型） | false | false |
| anthropic / openrouter / kimi / radius | false | false |
| unknown/custom | false（默认） | false |

能力为 false 时：

1. `prepare_server_request` 在构造 client/request 和查询 NegativeCapabilityCache 前返回 builtin 路径，compact POST 必须为 0；
2. 若 live history 含 `ResponsesCompactionCheckpoint`，`ensure_checkpoint_replayable_for_request` 先从强校验 sidecar 的 portable history 做 builtin continuity migration，再 resubmit；
3. opaque xAI compact output 不得发送给不兼容 route。

NegativeCapabilityCache 只处理 capability=true 但部署实际返回 404/405/501 的失配；不能用它首次探测 codex。

## anthropic（PKCE loopback）

| 项 | 值 |
|---|---|
| client_id | `9d1c250a-e61b-44d9-88ed-5944d1962f5e`（base64 解码） |
| authorize | `https://claude.ai/oauth/authorize` |
| token | `https://platform.claude.com/v1/oauth/token` |
| redirect_uri | `http://localhost:53692/callback`（固定端口 53692，host 可 env 覆盖） |
| scope | `org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload` |

- authorize URL 参数：`code=true&client_id&response_type=code&redirect_uri&scope&code_challenge&code_challenge_method=S256&state`（**state = code_verifier**）
- exchange：POST JSON `{grant_type:authorization_code, client_id, code, state, redirect_uri, code_verifier}`；30s 超时
- refresh：POST JSON `{grant_type:refresh_token, client_id, refresh_token}`
- 响应字段：`access_token`/`refresh_token`/`expires_in`；`expires = now + expires_in*1000 − 5min`
- 手动粘贴解析：URL（取 code/state）→ `code#state` → `code=...` → 裸 code；state 必须匹配 verifier
- 回调校验：路径 `/callback`、code+state 齐全、state 匹配

**API wire（`api/anthropic-messages.js`）**：
- OAuth token 特征 `sk-ant-oat` 前缀 → **Bearer auth（非 x-api-key）**
- 头：`anthropic-version` + `anthropic-beta: claude-code-20250219,oauth-2025-04-20[,其他beta]`
- system 提示单独字段；`/v1/messages` 端点
- 非 OAuth（普通 sk-ant-api 或 x-api-key）路径不变

**route 能力/净化**：`provider_id=anthropic`、dialect=`anthropic_messages`、remote compaction=false、grok checkpoint=false；最终请求禁止所有 xAI 私有头、inline compact、doom 和 `x_search`。system 转换留在新增 Anthropic adapter，不改 xAI Messages 路径。

## openai-codex（PKCE 固定 1455 + device code 双流）

| 项 | 值 |
|---|---|
| client_id | `app_EMoamEEZ73f0CkXaXp7hrann` |
| authorize | `https://auth.openai.com/oauth/authorize` |
| token | `https://auth.openai.com/oauth/token` |
| redirect_uri | `http://localhost:1455/auth/callback`（**固定端口 1455**） |
| scope | `openid profile email offline_access` |

- authorize 附加参数：`id_token_add_organizations=true&codex_cli_simplified_flow=true&originator=<客户端名>`；state = 16 随机字节 hex（**非 verifier**）
- exchange：POST **form-urlencoded** `{grant_type:authorization_code, client_id, code, code_verifier, redirect_uri}`
- refresh：POST form `{grant_type:refresh_token, refresh_token, client_id}`；响应缺 `access_token/refresh_token/expires_in` 任一 → 报错
- **credential 附加字段 `accountId`**：从 access token JWT 的 claim `https://api.openai.com/auth` 取 `chatgpt_account_id`；取不到 → 登录失败
- device 流：`POST /api/accounts/deviceauth/usercode` JSON `{client_id}` → `{device_auth_id, user_code, interval}`；`POST /api/accounts/deviceauth/token` JSON `{device_auth_id, user_code}` → 200 时 `{authorization_code, code_verifier}`（**再用它走 exchange**，redirect_uri 用 `https://auth.openai.com/deviceauth/callback`）；403/404 → pending；`deviceauth_authorization_pending` → pending；`slow_down` → slow_down；超时 15min
- verification URI：`https://auth.openai.com/codex/device`
- 登录时用户选择 browser/device_code
- **无 API key 路径**

**API wire（`api/openai-codex-responses.js`）**：
- 端点 `/codex/responses`；请求体 `store:false`（**store:true 会被拒绝**）、`includeSystemPrompt:false`、`include:["reasoning.encrypted_content"]`
- 头：`Authorization: Bearer` + `chatgpt-account-id: <accountId>` + `originator: pi`
- 429 语义：`/usage_limit_reached|usage_not_included|rate_limit_exceeded/` 或 HTTP 429
- 不用 `previous_response_id` 链式；连续性来自客户端发送的本地 transcript

**route 能力/净化（验收红线）**：
- stable `provider_id=openai-codex`，dialect=`openai_codex_responses`；不得降级成笼统 `openai_compatible`。
- `api_backend=Responses` 只描述协议族；remote compaction=false、grok checkpoint=false，且无论全局 `server_compaction` 开关为何值都不得调用 `/responses/compact`。
- 从 xAI checkpoint 切换到 codex：先 builtin continuity migration，再向 `/codex/responses` 发送普通 history；opaque compact output 不出站。
- 最终 allowlist：Authorization、`chatgpt-account-id`、`originator` 和必要通用 HTTP 头。禁止所有 `x-grok-*`、xAI URL 派生认证头、inline compact、doom、`x_search` 及 xAI-only body 字段。
- 401 refresh 后重新解析 access token 与 account ID，并重建 route；provider principal/account 变化不得复用旧 route/cache identity。

## github-copilot（device code + 动态 base_url）

| 项 | 值 |
|---|---|
| client_id | `Iv1.b507a08c87ecfe98`（base64 解码） |
| device code | `https://{domain}/login/device/code`（POST form `client_id` + `scope=read:user`，UA 头） |
| access_token | `https://{domain}/login/oauth/access_token`（form `client_id,device_code,grant_type=urn:ietf:params:oauth:grant-type:device_code`） |
| copilot token | `https://api.{domain}/copilot_internal/v2/token`（**GET**，Bearer `<github access token>`） |

- domain：GHE 输入（登录时 prompt，可空）；`normalizeDomain` 取 hostname；默认 `github.com`
- verification_uri 必须 http(s)（防 open 任意协议）
- 轮询错误：`authorization_pending`→pending；`slow_down`→slow_down+interval
- copilot token 响应：`{token, expires_at(秒)}`；`expires = expires_at*1000 − 5min`
- **credential.refresh = github access token（不轮换）**；`credential.enterpriseUrl = GHE 域名`
- **base_url 每请求从 token 的 `proxy-ep=` 解析**：`proxy.xxx` → `api.xxx`，`https://api.xxx`；解析失败回退 enterpriseUrl → `https://copilot-api.{domain}` → 默认 `https://api.individual.githubcopilot.com`
- 登录后：enableAllGitHubCopilotModels（对每个已知模型 `POST {base}/models/{id}/policy` JSON `{state:"enabled"}`，头含 `openai-intent:chat-policy`、`x-interaction-type:chat-policy`）+ fetchAvailableModelIds（`GET {base}/models`，过滤 `model_picker_enabled===true && policy.state!=='disabled' && supports.tool_calls!==false`）

**固定头**：`User-Agent: GitHubCopilotChat/0.35.0`、`Editor-Version: vscode/1.107.0`、`Editor-Plugin-Version: copilot-chat/0.35.0`、`Copilot-Integration-Id: vscode-chat`；列表请求加 `X-GitHub-Api-Version: 2026-06-01`

**动态头（`api/github-copilot-headers.js`）**：`X-Initiator: user|agent`（末条消息 role != user → agent）、`Openai-Intent: conversation-edits`、含图片时 `Copilot-Vision-Request: true`

**API 协议**：按模型三协议（CC/Responses/Messages），模型表在 `providers/github-copilot.models.js`；W6 移植时对照。

**route 能力/净化**：
- 三种协议分别选择新增 Copilot adapter；包括 Responses 模型在内均 remote compaction=false、grok checkpoint=false。
- token 中 proxy base、GHE enterprise URL 或可用模型清单变化后重建 route；base URL 参与 route/cache identity。
- editor/intent/vision 头只允许进入 Copilot route；禁止 xAI 私有头、inline compact、doom 和 `x_search`。
- GHE、verification URI 和动态 proxy URL 均执行 http(s)/hostname 校验。

## openrouter（PKCE，无 refresh）

| 项 | 值 |
|---|---|
| authorize | `https://openrouter.ai/auth` |
| token | `https://openrouter.ai/api/v1/auth/keys` |
| 回调 | **ephemeral 端口** + 路径 `/oauth/callback/{uuid}`，无固定端口 |

- authorize 参数：`callback_url`（= 实际监听 URL）、`code_challenge`、`code_challenge_method=S256`；**无 client_id**
- exchange：POST JSON `{code, code_verifier, code_challenge_method:"S256"}`；30s 超时；响应 `{key}` → **永久 API key**
- credential：`access = key`，`refresh = ""`，`expires = 永不`；**refresh 是 no-op（返回自身）**
- 登录超时 5min；回调幂等（claimed 后 409）
- API：OpenAI ChatCompletions，`Authorization: Bearer <key>`，base `https://openrouter.ai/api/v1`
- route：`provider_id=openrouter`、dialect=`openai_chat_completions`、remote compaction=false、grok checkpoint=false；永久 key 仍是 OpenRouter 专属 credential，缺失时不得回落 xAI bearer，最终 wire 禁止 first-party 头/能力

## kimi-coding（device code）

| 项 | 值 |
|---|---|
| client_id | `17e5f671-d194-4dfb-9706-5516cb48c098` |
| oauth host | `https://auth.kimi.com`（env `KIMI_CODE_OAUTH_HOST` 可覆盖） |
| device_authorization | `POST {host}/api/oauth/device_authorization`（form `client_id`）→ `{device_code,user_code,verification_uri,verification_uri_complete,interval,expires_in}` |
| token | `POST {host}/api/oauth/token`（form `client_id,device_code,grant_type=device_code`） |

- verification_uri(_complete) 必须 http(s)
- 轮询错误：`authorization_pending`→pending、`slow_down`→slow_down+interval、`expired_token`→失败提示重登、`access_denied`→失败；5xx → 失败
- refresh：`POST {host}/api/oauth/token` form `{client_id, grant_type:refresh_token, refresh_token}`；401/403/`invalid_grant` → **凭证死亡**（应清理并提示重登）；429/5xx 退避重试 1s/2s/4s 最多 3 次
- 响应字段：`access_token`/`refresh_token`/`expires_in`（缺失报错）
- API：**Anthropic Messages 协议**，base `https://api.kimi.com/coding`，最终 messages 路径由 adapter 规范化一次，`Authorization: Bearer`（toAuth 返回 headers 形态）
- route：`provider_id=kimi-coding`、dialect=`kimi_anthropic_messages`、remote compaction=false、grok checkpoint=false；refresh 判定凭证死亡后清理并失败，不回落 xAI bearer

## radius（gateway OAuth，双流）

| 项 | 值 |
|---|---|
| client_id | `pi-gateway`（gateway 侧的注册 client） |
| 默认 gateway | `https://radius.pi.dev`（grok-build 需决定是否沿用或改为可配置） |
| OAuth discovery | `GET {gateway}/v1/oauth` → `{authorizationEndpoint}` |
| token | `POST {gateway}/v1/oauth/token`（form） |
| device | `POST {gateway}/v1/oauth/device`（form `client_id,scope`） |
| 回调 | **固定 127.0.0.1:1456**，路径 `/oauth/callback` |
| scope | `gateway offline_access` |

- 登录前用户选 browser/device-code；browser 流参数：`response_type=code&client_id&redirect_uri&scope&code_challenge&code_challenge_method=S256&handoff=url&state`
- exchange form：`{grant_type:authorization_code, client_id, redirect_uri, code, code_verifier}`
- refresh form：`{grant_type:refresh_token, client_id, refresh_token}`
- 响应：`access_token/refresh_token/expires_in`；`expires = now + expires_in*1000 − 60s`（**skew 60s，非 5min**）
- 错误体：JSON `{error, error_description}`（OAuthResponseError 语义：authorization_pending/slow_down/expired_token/access_denied）
- **credential 附加字段 `gatewayConfig`**：登录后从 `GET {gateway}/v1/config` 加载 `{baseUrl, models[]}`（模型带 `id/name/reasoning/input/cost/contextWindow/maxTokens`），存进凭证，模型清单由此而来
- API：**pi-messages 协议**（需新 adapter），`baseUrl` 来自 gatewayConfig
- route：`provider_id=radius`、dialect=`pi_messages`、remote compaction=false、grok checkpoint=false；`gatewayConfig.baseUrl`/模型目录更新后原子替换旁路 route 索引，不修改共享 `ModelInfo` schema
- 当前 gateway config 没有声明 `responses-compact-grok` 能力，因此默认 false；未来只有协议显式声明且实现同一 contract 时才可单独评审开启

## 通用交互契约（AuthInteraction，Rust trait 化）

- `notify`：`auth_url{url,instructions}` / `device_code{userCode,verificationUri,intervalSeconds,expiresInSeconds}` / `progress{message}`
- `prompt`：`text{message,placeholder}`（copilot GHE 域名）/ `select{message,options[]}`（codex browser/device、radius browser/device）/ `manual_code{message,placeholder}`（粘贴授权码/URL）
- 取消：signal 贯穿所有等待/轮询
- 所有交互都携带 provider ID，并经新增 provider-scoped RPC 模块完成；不得调用全局 xAI `set_auth_method` 或修改 `auth_method_id`
- pager 复用现有 list/question 组件；provider 状态机放新增模块，现有 action/router/task-result 仅做注册/转发
- subagent 不直接弹交互式登录；缺凭证时返回带 provider 的可诊断错误，由父级或用户显式发起登录

## 错误语义汇总

| 场景 | 行为 |
|---|---|
| 回调 state 不匹配 | 400 + 报错 |
| exchange/refresh HTTP 非 2xx | 报错含 status + body |
| kimi refresh 401/403/invalid_grant | 凭证死亡 → 清理 + 提示重登 |
| codex device 404 | “device code login 未启用，用 browser 登录” |
| route 不支持 remote compaction | 本地直接 builtin；compact POST=0，不写 NegativeCapabilityCache |
| capability=true 的 compact endpoint 返回 404/405/501 | 首次分类 Unsupported 并写 negative cache；TTL 内抑制后续请求 |
| live grok checkpoint 与当前 route 不兼容 | 强校验 sidecar → builtin continuity migration → resubmit；迁移前不发 provider 请求 |
| 非 xAI wire 发现 xAI credential/header/capability | 本地拒绝或最终净化，不允许发出后等待服务端报错 |
| provider refresh/credential 缺失 | 当前 provider 失败；不得回落 xAI session token 或其他 provider |
| 端口占用（1455/1456/53692） | 报错（codex 提示改用 device code 流） |
| device 轮询超时（含 slow_down 超时） | 专门超时文案 |
