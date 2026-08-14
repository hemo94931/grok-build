# 上游同步计划：origin/main (2026-08-13) → dev

按 upstream-sync skill 五阶段流程制定。目标：合并 3 个 "Synced from monorepo" drop，不丢 fork 功能，不保留被上游吸收的 fork 旧实现。

## Phase 0 — 基线（已完成）

| 项 | 值 |
|---|---|
| merge-base | `b13fa526` |
| fork head | `000d8dc`（2026-08-14，"test: close provider authentication regressions"） |
| upstream head | `eb267fe`（2026-08-13，"Synced from monorepo"） |
| 工作区 | 干净，无进行中 merge |
| 上游规模 | 673 文件，+155K / −112K（3 个 drop：be71313、e5fd481、eb267fe） |
| fork 规模 | 270 文件，+50K（33 个提交含 4 个 merge：provider OAuth/API key、DeepSeek/Z.AI/Radius provider、Responses server compaction、credentials 净化等） |

### 本次 drop 的性质：大规模重构

- **测试模块拆分**：约 15 处 `#[cfg(test)] mod ... { }` 内联测试被移入 `*_tests.rs`（compaction.rs、agent/config.rs、leader/server.rs、session_compact.rs、persistence.rs、goal_tracker.rs、slash_commands.rs、tool_index.rs 等），原文件用 `#[path = "..."] mod x;` 声明。
- **新 crate 抽取**：`xai-grok-session-events`、`xai-grok-session-search`、`xai-grok-bundle`、`xai-grok-foreign-sessions`、`xai-grok-active-sessions`、`xai-grok-diag-server`、`xai-grok-pager-diff`、`xai-fuzzy-file-search`、`xai-compaction-transcript`。
- **新功能**：`image_budget.rs`（body 精确测量 + 图像驱逐）、`acp_session_impl/image_strip.rs`（server-rejected 图像剥离持久化）、permission manager 重写（`manager/` + `bash_grants.rs` + `rules.rs`）、session search、workspace daemon、publish.rs、status_config 等。
- **API 变化**：`SamplingError::Http` 保持 newtype（fork 改成了 struct variant）；`try_parse_stream_error` 等函数名未变；新增 `StripReason`/`StripOutcome`/`ImagesStripped` 事件；session actor 新增 `pending_image_strip` 字段。

### 子sumption 检查（驱动所有 per-file 决策）

| fork 功能 | 上游现状 | 结论 |
|---|---|---|
| request_task.rs 的 strip_images 重试 | 上游已实现更完整版（`StripReason` 分类 + `emit_images_stripped` + shell 侧持久化 image_strip.rs） | **上游吸收 → 删 fork 版** |
| request_builder.rs 的 `conversation_body_bytes` 精确 body 测量驱逐 | 上游新 `image_budget.rs` 有同名同思路实现 | **上游吸收 → 删 fork 版** |
| sampler config 字段 stamping（client_version/deployment_id/bearer_resolver 等） | 上游抽成 `stamp_session_local_sampler_fields` helper | **部分吸收**：公共部分用上游 helper，fork 特有字段（supports_backend_search、compactions_remaining、doom_loop_recovery 等）保留 |
| `SamplingError::Http { kind, source }` + `HttpErrorKind` | 上游保持 `Http(reqwest::Error)` | **采纳上游形状**，fork 分类逻辑抽成独立 helper（见预重构 R1） |
| provider 生态（OAuth/API key/DeepSeek/Z.AI/Radius/OpenRouter） | 上游完全没有 | fork 独有，必须完整保留 |
| Responses compaction（cache_routing、responses_compaction、responses_server_compaction、compaction_gc） | 上游没有 | fork 独有，保留（fork-owned 文件本次 0 冲突） |
| credentials 净化（`*_with_credentials` 函数族） | 上游只有非 credentials 版本 | fork 独有，port 到上游新 error.rs |

## Phase 1 — 干跑冲突清单（已完成）

git 2.32 无 `merge-tree --write-tree`，改用临时 worktree（`git worktree add --detach` + `git merge --no-commit origin/main`）获取真实冲突清单，随后已清理。

**29 个内容冲突文件 / 51 个 hunk**，无 add/add、无 modify/delete 冲突（Cargo.lock 虽两侧都改但自动合并成功，不在冲突清单）。

**rename 安全检查（通过）**：4 个 "fork 修改过 + 上游移动" 的文件全部正确配对，fork 增量保留在新路径：
- `foreign_sessions/capability/mod.rs` → `xai-grok-foreign-sessions/`（fork 的去 `return` 改写保留）
- `search_bootstrap_tests.rs` → `xai-grok-session-search/src/bootstrap_tests.rs`（fork 的 `#[serial_test::serial]` ×3 保留）
- `compaction_transcript.rs` → `xai-compaction-transcript/src/lib.rs`
- `.gitignore`（fork 追加 4 行，合并后 == fork 版）

## Phase 2 — 冲突分类与策略

### A 类：机械字段 union（7 文件）— 双字段都保留

上游给 session actor struct 加了 `pending_image_strip: parking_lot::Mutex<Option<_>>`，fork 加了 `post_compact_usage_state: AtomicU8`。所有 struct literal 构造点冲突。解法：**逐 hunk 编辑冲突标记**——保留 theirs 的 `pending_image_strip` 行 + 补回 fork 的 `post_compact_usage_state` 行（每处 1 行，无语义判断）。⚠️ 不要整文件 `checkout --theirs`：这些文件里 fork 还有自动合并保留的其它改动（如 `cancel_running_task_tests.rs` fork 总 +19 行，只有 4 个 hunk 是字段冲突），整文件取 theirs 会丢。

- `acp_session_impl/spawn.rs`（hunk1）
- `acp_session_tests/support.rs`、`inline_auto_compact_flow_tests.rs`（×3）、`cancel_running_task_tests.rs`（×4）、`idle_resume_tests.rs`、`memory_config_tests.rs`、`replay_buffer_send_update_tests.rs`

### B 类：上游测试拆分（5 文件）— 取上游结构 + port fork 独有测试

上游把内联 test mod 移到 `*_tests.rs` 并改为 `#[path] mod` 声明；fork 版本仍内联且含 fork 独有测试。解法：保留上游的 `#[path] mod` 声明（即 ours 侧删除内联 mod 体），把 fork 独有测试搬进上游的 `*_tests.rs` 文件。

| 冲突文件 | 上游新测试文件 | fork 独有测试（需搬入） |
|---|---|---|
| `session/compaction.rs`（hunk1） | `compaction_two_pass_prefire_helper_tests.rs` | `checkpoint_item`、`prefire_layout_accepts_normal_and_unique_leading_checkpoint_only`、`prefire_discard_requires_seed_and_committed_total_to_shrink` |
| `session/compaction.rs`（hunk2） | `compaction_inline_auto_compact_flow_tests.rs` | `compact_errors_redact_credentials_before_acp_or_artifact_use`、`surface_compact_auth_failure_uses_request_time_provider_source`、`model_switch_arms_deferred_exact_request_compaction`、`clear_auth_suppress_rearms_deferred_model_switch_compaction`、`compaction_cancel_classification_accepts_server_and_builtin_errors`（其余与上游同名测试以 theirs 为准，注意 fork 的 `create_test_actor` 与上游差异） |
| `agent/config.rs`（hunk2） | `agent/config_tests.rs` | 已核：fork 在该 mod 内无独有测试（`main_cli_tools_override_preserves_profile_injection_policy` 上游 config_tests.rs 已有同内容测试），取 theirs 即可 |
| `leader/server.rs`（hunk1） | `leader/server_tests.rs` | merge 时核对 fork 的 `mod tests` 内是否有独有用例 |
| `session/helpers/session_compact.rs`（hunk2） | `session_compact_classify_tests.rs` | merge 时核对（fork 的 `is_det`/`sampling_api_4xx_is_deterministic_except_408_and_429` 等） |
| `session/persistence.rs`（hunk2） | `session/persistence_tests.rs` 及 15 个 `persistence_*_tests.rs` | merge 时核对（fork 的 `persistence_tests.rs` 有 +122 行） |

### C 类：上游吸收 fork 功能（3 文件）— 逐 hunk 取 theirs，删 fork 旧实现

⚠️ 这些文件 fork 有大量自动合并保留的改动（request_task.rs 总 +129/−39 只有 2 个冲突 hunk；client.rs +916/−350 只有 5 个 hunk；config.rs +505/−67 只有 2 个 hunk），**不得整文件 checkout --theirs**，只解冲突标记本身。

- **`sampler/actor/request_task.rs`**（hunk1、hunk2）：fork 的 `is_likely_body_rejected` 重写重试 + `strip_images` 分发，被上游的 `StripReason`/`strip_images` 版取代。两个 hunk 取 theirs；检查 fork 在 `SamplingDispatch` 上的其它增量（resolved 冻结语义上游是否已有，merge 时 diff 核对）。
- **`chat-state/actor/request_builder.rs`**（hunk1）：fork 的 body 测量驱逐被上游 `image_budget.rs` 吸收。该 hunk 取 theirs；确认 fork 是否还有其它增量（fork 版 1126 行 vs base 866，diff 核对剩余部分——注意上游该文件已缩到 293 行）。
- **`acp_session_impl/sampler_turn.rs`**（hunk1、hunk2）：hunk1 = fork 的内联 config stamping（client_version/deployment_id/user_id/bearer_resolver/supports_backend_search/compactions_remaining/doom_loop_recovery 等字段赋值——注意这些字段在 `SamplerConfig` 里 base 时代就有，上游也有）vs 上游已把公共 stamping 抽成 `agent/config.rs::stamp_session_local_sampler_fields` helper。解法：以 theirs 为 base（保留 helper 调用），merge 时 diff 两侧 stamping 逐一核对，只补 helper 未覆盖的字段（如 `compaction_at_tokens`）与取值来源差异（fork 用参数 `creds.client_version` vs 上游 `self.xxx`）。hunk2 = fork 的 `SamplingClient::new_with_route(cfg, route_hint)` vs 上游 `SamplingClient::new(cfg)`：fork 独有路由，port 到上游 client.rs 新构造路径（与 D 类 client.rs 条目联动）。

### D 类：fork 独有语义 port 到上游新结构（15 文件）

- **`sampling-types/error.rs`**（hunk1、hunk2）：取 theirs 为 base。fork 增量：`HttpErrorKind` + 分类逻辑 + `_with_credentials` 函数族 + `Http { .. }` 模式匹配的 Debug/retryable 实现。若先做预重构 R1，此处只剩 `_with_credentials` 函数的 port（fork 的 `try_parse_stream_error_with_credentials`/`user_facing_api_error_message_with_credentials`/`is_retryable_api_status` 及其调用点 client.rs hunk1 的 import 修正）。
- **`sampler/client.rs`**（5 hunk）：取 theirs 为 base，逐个 port：hunk1 import（credentials 版函数）、hunk2-3 `extra_tool_entries` 序列化 + xAI 工具注入（若上游已重构该路径则适配新结构）、hunk4-5 `CreateResponseWrapper::try_normal` 类型化 wrapper + `conversation_stream_responses`（fork 的 Responses compaction 入口）。注意上游新增 endpoint builder（`base_url`/query_params），fork 的 `new_with_route` 需要接在上游新构造路径上。
- **`chat-state/persistence.rs`**（4 hunk）：fork 的 `replace_history_and_ack` trait 方法 + `AcknowledgedReplaceHistory` 事件 + `HistoryReplaceError`；上游加了 `StripOutcome`。union——两边的 trait 方法都保留。
- **`chat-state/handle.rs`**（hunk1）：fork 的 `commit_compaction` 命令，port 到上游 handle.rs。
- **`chat-state/mutations.rs`**（hunk1、hunk2）：fork 的 checkpoint repair 逻辑 + `replace_history` 调用点（hunk2 显示上游删除了某处调用——以 theirs 为准，若 fork 的调用是该处独有的 durable 替换则补回）。fork 版 895 行 vs theirs 640，diff 核对所有增量。
- **`chat-state/lib.rs`**（hunk1）：export 列表 union（fork 的 `HistoryReplaceError` + 上游的 `StripOutcome`）。
- **`shell/session/persistence.rs`**（hunk1、hunk2）：fork 的 `ReplaceChatHistoryAndAck` 消息 + 处理分支，port 到上游 persistence.rs（注意上游已拆 15 个 `persistence_*_tests.rs`，fork 的 `persistence_tests.rs` +122 行增量按 B 类处理）。
- **`shell/session/chat_persistence.rs`**（2 hunk）：fork 的 `replace_history_and_ack` 实现（BrokenPipe/Indeterminate 语义），以 theirs 为 base 补回。
- **`shell/session/storage/mod.rs`**（hunk1）：fork 的 `replace_chat_history_durable` trait 方法，port 到上游 trait。
- **`shell/session/acp_session.rs`**（hunk1、hunk2）：hunk1 fork 的 `PromptCacheObservation.request_kind` 注释/逻辑；hunk2 = fork 侧删除了某个函数 vs 上游修改了 base 已有的 `persist_chat_history_jsonl_sync`（非上游新增）——取 theirs。
- **`shell/session/acp_session_impl/tasks_cancel.rs`**（hunk1）：fork 的 `suppress_task_wakes` stop-gesture 逻辑，port。
- **`pager/app/acp_handler/session_notification.rs`**（hunk1）：fork 的 `late_provider_reauth` vs 上游 `deferred_subagent_finish`——union，两个局部变量都保留，后续逻辑各自接线。
- **`shell/agent/config.rs`**（hunk1）：fork 的 `provider_context` headers 注入 + `inject_url_derived_headers`，port 到上游 config.rs（上游该文件缩至 5459 行，注意上下文变化）。
- **`sampling-types/conversation.rs`**（hunk1）：fork 的 `validate_for_backend`，port。
- **`sampling-types/lib.rs`**（hunk1）：export union。

### E 类：测试断言小修（其余 acp_session_tests 等）

- `sampler/tests/test_actor.rs`：import 列表 union（fork 的 `ProviderRouteHint` + 上游 `StripReason`）。
- `acp_session_tests/auth_error_no_retry_tests.rs`、`permission_auto_mode_tests.rs` 等未列出的：已在 A 类覆盖或合并干净。

### 其他注意

- `Cargo.lock`：干跑实测自动合并成功（无冲突），但 fork 侧 +9 / 上游侧 +196 改动量大，merge 后仍建议 `cargo metadata > /dev/null` 确认解析一致。
- `Cargo.toml`：fork 未改（依赖复用 base 已有）；上游 +18 行（新增 ~10 个 crate）。验证阶段确认 fork 用到的依赖在新 workspace 解析齐全。
- fork-owned 文件（`cache_routing.rs`、`responses_compaction.rs`、`compaction/responses.rs`、`compaction_gc.rs`、`responses_server_compaction.rs`、`provider_auth.rs` 等）本次 0 冲突，但其中 **22 处 `xai_chat_state::compaction_transcript::` 旧路径引用**必须修复：上游已把该模块移到独立 crate `xai-compaction-transcript`（`xai_chat_state` 无 re-export，已核实）。受影响文件：`acp_session_impl/tool_dispatch.rs`、`compaction_gc.rs`、`compaction_segments.rs`、`storage/jsonl/copy.rs`、`storage/jsonl/copy_tests.rs`、`storage/jsonl/mod.rs`、`storage/jsonl/tests.rs`、`storage/responses_compaction.rs` 等——全部改为 `xai_compaction_transcript::`（merge 后 `xai-grok-shell` 已依赖新 crate）。

## Phase 2+ — 预 merge 解耦重构（建议 2 个提交，行为保持）

### R1（推荐，经 review 核实）：采纳上游 `Http(reqwest::Error)` 变体形状（提交 1）

fork 把 `Http` 从 newtype 改成了 `Http { kind: HttpErrorKind, source }`。上游保持 newtype 且 error.rs 每轮都在增长（本次 +359 行）。保留 fork 变体形状 = 每次上游动 error.rs 都冲突。

重构内容（在 dev 上先做，不改变任何行为）：
1. `error.rs`：变体改回 `Http(reqwest::Error)`；`From<reqwest::Error>` 的 if/else 分类链抽成 `pub fn http_error_kind(err: &reqwest::Error) -> HttpErrorKind`（free function，放 fork 侧新位置或 error.rs 底部）；Debug/retryable 实现改用 `http_error_kind(&err)`。
2. 调用点（共 6 文件，已核实）：`error.rs` 自身（Debug impl 用 kind、retryable 判断）改用 `http_error_kind(&err)`；消费 kind 的 3 处用 `Http(source)` + helper：`retry.rs:298`（显示文案 `({kind}{status})`）、`retry.rs:404`（sanitized clone）、`shell/sampling/error.rs:119`；仅模式匹配不消费 kind 的用 `Http(_)`：`request_task.rs:454`（经 `source` 取 status）、`events.rs:233`、`session_compact.rs:83`。
3. 验证：`cargo check -p xai-grok-sampling-types -p xai-grok-sampler -p xai-grok-shell` + 相关测试。
4. 收益：本次 merge 的 `session_compact.rs` hunk1 自动消失、`error.rs` hunk1 缩小；未来上游 error.rs 变更不再因变体形状冲突。

### R2（可选）：compaction.rs 测试拆分采纳（提交 2）

把 fork 版 compaction.rs 的两个内联 test mod 按上游方式拆出（`#[path]` 声明 + fork 独有测试放入 `compaction_two_pass_prefire_helper_tests.rs` / `compaction_inline_auto_compact_flow_tests.rs`）。收益：本次 merge 的 compaction.rs 两个 hunk 消失——compaction.rs 是 fork 每轮 sync 都冲突的高耦合文件（responses compaction 内联其中），拆分测试只是第一步，真正的解耦是后续把 responses compaction 逻辑整体移入 fork-owned 模块（本计划不执行，标注为后续项）。

不做的理由（其余文件）：config.rs/server.rs 等测试拆分收益只在本次 merge，且工作量与 merge 内解决相同；skill 原则是重构只在能转化为干净自动合并时做。

## Phase 3 — 执行步骤

1. 确认工作区干净、无 MERGE_HEAD；`git fetch origin`。
2. 提交 R1（推荐）；如需 R2 则提交 R2。
3. `git merge origin/main`（保留默认 merge，不 squash）。
4. 按 Phase 2 分类表逐文件解决 29 个冲突（**全部逐 hunk 处理，禁止整文件 `checkout --theirs`**——除冲突 hunk 外，fork 还有大量自动合并保留的改动）：
   - A 类：编辑冲突标记，保留 theirs 的 `pending_image_strip` + 补回 fork 的 `post_compact_usage_state`。
   - B 类：ours 删除内联 test mod 体（保留上游 `#[path]` 声明），fork 独有测试写入上游 `*_tests.rs` 文件。
   - C 类：冲突 hunk 取 theirs（上游已吸收），文件其余部分保持自动合并结果。
   - D 类：以 theirs 侧为基底，逐 hunk port fork 逻辑。
5. 修复 fork-owned 文件的旧路径引用：`xai_chat_state::compaction_transcript` → `xai_compaction_transcript`（22 处）。
6. `cargo metadata > /dev/null` 重新生成 Cargo.lock，不手解。
7. `grep -rn '<<<<<<<' crates/ docs/ scripts/ .agents/` 确认无残留冲突标记（51 个 hunk 全清）；`git add -A`。
8. 提交前用 `git status` 确认 "All conflicts fixed but you are still merging"（若被 stash/重启打断：恢复 `.git/MERGE_HEAD` = `git rev-parse origin/main`、`.git/MERGE_MODE`、`.git/MERGE_MSG`）。
9. `git commit` 写审计消息（见 Phase 5）。

## Phase 4 — 验证计划（targeted）

用真实退出码（`PIPESTATUS`/tail 原始输出，不要 `| grep` 后看 `$?`）。环境注意：`RUST_MIN_STACK=67108864`；磁盘空间（debug 构建易爆盘，先 `df -h`）。

按冲突面排序的定向验证（不是全量 6000 测试）：

1. `cargo check --workspace --all-targets` 或至少受影响 crate：`xai-grok-sampling-types`、`xai-grok-sampler`、`xai-chat-state`、`xai-grok-shell`、`xai-grok-pager`、`xai-grok-workspace`、`xai-acp-lib`、`xai-grok-foreign-sessions`、`xai-grok-session-search`、`xai-compaction-transcript`。→ 先过编译（**必须 `--all-targets`**：测试/集成目标会暴露 fork-owned 文件的旧路径引用），错误即按 D 类"残留引用"清单修（grep fork 独有符号：`ProviderRouteHint`、`new_with_route`、`replace_history_and_ack`、`replace_chat_history_durable`、`commit_compaction`、`ResponsesCompactResponse`、`HttpErrorKind`、`_with_credentials`、`late_provider_reauth`、`post_compact_usage_state`、`cache_routing`、`compaction_transcript` 等）。
2. 定向测试（按本次冲突面）：
   - `xai-grok-sampling-types`：error.rs 相关（R1 后）
   - `xai-grok-sampler`：request_task/retry/client（C/D 类核心）
   - `xai-chat-state`：actor/mutations + persistence + server_compaction（fork 的 `tests/server_compaction.rs`）、image_budget/conversation 相关（上游新功能与 fork 增量交界）
   - `xai-grok-shell`：session/compaction（two_pass_prefire、inline_auto_compact_flow、classify）、session/persistence、acp_session_tests（spawn/support/cancel/idle/memory_config/replay_buffer）、storage/jsonl、responses_compaction 相关、agent/config 与 leader/server 拆分套件
   - `xai-grok-pager`：acp_handler/session_notification（含 `deferred_subagent_finish` 上游新套件）+ interactions
   - **fork 新增 integration 必须显式跑**：`tests/responses_compaction_recovery.rs`、`tests/responses_server_compaction.rs`、`xai-grok-sampler/tests/provider_http_matrix.rs`、`xai-chat-state/tests/server_compaction.rs`（它们依赖的 fixture/API 可能被上游改动破坏）
3. **每个失败必须归因**：`git log -S <symbol>` 定位引入提交；`git show <pre-merge>:<path>` 对比失败路径在 merge 前是否相同；`git merge-base --is-ancestor` 判定先后。merge 前 fork 提交上就失败的 → 标注 pre-existing + 引入提交，写进 merge commit 消息。便宜且归因清晰的顺手修（如挂起测试缺 drain），需设计的留给后续并记录。

## Phase 5 — merge commit 审计消息模板

```
Merge origin/main into dev (upstream 2026-08-13 drop)

Upstream: 3 monorepo syncs (be71313..eb267fe), 673 files, test-module split
to *_tests.rs, new crates (session-events/search/bundle/foreign-sessions/...),
image_budget + image_strip features, permission manager rewrite.

Resolutions (29 conflicts / 51 hunks) — 逐文件基底/port/drop/原因见下：
- A 机械字段: spawn.rs 等 7 文件 — 双字段保留（pending_image_strip + post_compact_usage_state）
- B 测试拆分: compaction.rs 等 5 文件 — 采用上游 #[path] mod，fork 独有测试迁入 *_tests.rs
- C 上游吸收: request_task.rs(strip 重试) / request_builder.rs(body 驱逐) — 冲突 hunk 取 theirs
  (upstream image_strip/image_budget 已实现同功能)
- D fork 独有: sampler_turn.rs(config stamping 采纳上游 helper + new_with_route 路由) /
  error.rs(credentials 函数族) / client.rs(try_normal+extra_tool_entries) /
  chat-state persistence(ReplaceHistoryAndAck) / config.rs(provider headers) / ... — port 到上游新结构
- fork API 删除: SamplingError::Http{kind,source} → Http(reqwest::Error)（分类抽为 http_error_kind helper，提交 <oid>）
- 采纳上游: StripReason/StripOutcome/ImagesStripped/pending_image_strip/stamp_session_local_sampler_fields
- 残留引用修复: xai_chat_state::compaction_transcript → xai_compaction_transcript（22 处）

【逐文件决策表】29 文件 × (基底: ours/theirs/union, port 了什么, drop 了什么, 原因) — merge 时按 Phase 2 分类表填全。
【fork API → 替代 API 完整清单】被上游吸收而删除的 fork API 及其上游替代品（image-strip 重试→StripReason/emit_images_stripped；body 驱逐→image_budget.rs；config stamping→stamp_session_local_sampler_fields；Http{kind,source}→Http+http_error_kind）。

Pre-existing failures (none / 列出引入提交与证据; 标注 fixed 或 deferred):
Verification: cargo check <crates>; 定向测试 <列表>; 结果
```

## Review 记录

**第一轮（k3-256k）**：hunk 数 46→51（cancel_running_task ×4、sampler_turn 补 hunk2）；fork 提交数 30→33（含 4 merge）；sampler_turn hunk1 字段归属描述纠错（字段 base 时代即存在，差异在取值来源与 helper 抽取）；acp_session hunk2 实为 fork 删除 vs 上游改 `persist_chat_history_jsonl_sync`；Cargo.lock 无冲突（自动合并）；fork 未改 Cargo.toml；config_tests 无 fork 独有测试。R1 行为保持已核实（`Http{kind,source}` 唯一构造点为 From impl，kind 可确定性重推导）。

**第二轮（gpt-5.6-sol，xhigh）**：分类计数修正（A 8→7、B 6→5、D 12→15，模板 C 行误含 sampler_turn 已移回 D）；**禁止整文件 `checkout --theirs`**（request_task +129/−39 仅 2 hunk 冲突、client +916/−350 仅 5、config +505/−67 仅 2——fork 大量改动自动合并保留，整文件取 theirs 会丢）；**新增残留引用修复项**：22 处 `xai_chat_state::compaction_transcript`（上游已移入 `xai-compaction-transcript` crate 且无 re-export，已核实）→ 改 `xai_compaction_transcript`；R1 无附加风险（HttpErrorKind 无 serde、跨边界只存 `SamplingErrorKind::Http`）；验证补 `--all-targets`、config/server 拆分套件、image_budget/conversation、pager deferred_subagent_finish 套件、显式跑 fork 新增 integration（provider_http_matrix、responses_compaction_recovery、responses_server_compaction）；审计模板补逐文件决策表与 fork API→替代 API 清单。

## 风险与后续

- **静默丢失风险已排除**：4 个 moved 文件 rename 配对成功，fork 增量（去 return 改写、serial 标注、gitignore 行）均在干跑中验证保留。
- **后续解耦项**：compaction.rs 仍是高耦合点（fork 的 responses compaction 内联在 upstream 的 compaction.rs 中），本次 merge 后可把逻辑整体搬入 `session/compaction/responses.rs` 等 fork-owned 模块，让下次 sync 的冲突面继续下降。
- **R1 若不做**：error.rs 的 hunk1 需在 merge 内做同样的事（采纳上游变体 + helper），只是工作混进 merge commit、未来冲突面不降。
- **遗留核对项**：B 类表格中"需核对"的 fork 独有测试（server.rs/session_compact.rs/persistence.rs 的内联 test mod），merge 时逐文件 diff fork 版 test mod 与上游 `*_tests.rs`，保证无测试静默丢失。
