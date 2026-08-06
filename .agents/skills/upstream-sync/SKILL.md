---
name: upstream-sync
description: Merge upstream changes (origin/main monorepo syncs) into a long-lived fork branch (dev) without losing fork features. Use whenever the user asks to sync/merge/rebase upstream into dev, pull new changes from the monorepo, resolve merge conflicts against upstream, or says things like "合并上游", "sync from monorepo", "merge origin/main". Covers dry-run conflict inventory via git merge-tree, pre-merge refactors on the fork branch that shrink conflict surface — adopting upstream names/structure for mechanical divergence, and extracting inline fork logic into fork-owned modules for durable decoupling from upstream — per-file resolution strategy (upstream wins where it subsumes fork features, fork deltas ported onto upstream's new structure otherwise), pre-existing-vs-regression failure attribution through git archaeology, targeted verification instead of full test suites, and audit-ready merge commits.
---

# Upstream Sync

A repeatable workflow for merging upstream monorepo drops into a fork's long-lived feature branch. Two failure modes define this work: silently dropping fork features by blindly taking "theirs", and silently keeping stale fork duplicates of what upstream has since implemented better. The process below exists to make both impossible by construction: analyze before touching the worktree, decouple before merging, attribute every test failure, and leave an audit trail in the merge commit.

The core principle: **upstream is the direction of travel**. Where upstream has subsumed fork functionality, adopt upstream's implementation and delete the fork's. Where the fork has unique functionality, keep it — but re-express it on top of upstream's new structure rather than preserving the fork's old shape.

## Phase 0 — Ground truth before anything else

Never start resolving from a dirty or ambiguous state.

1. Commit or stash any in-progress work. If a previous merge is mid-flight (`.git/MERGE_HEAD` exists), finish it first — never stack two merges.
2. `git fetch origin` and identify three commits: merge base (`git merge-base HEAD origin/main`), fork head, upstream head. Read upstream's commit messages and diff stat (`git diff --stat <base>..origin/main`) to understand what kind of drop this is.
3. **Check subsumption**: for each major fork feature, grep upstream's tree for its surface (file names, key symbols, protocol strings). If upstream now contains the feature, plan to drop the fork's version. If upstream lacks it entirely, the fork's must survive the merge intact. Write the answer down — it drives every per-file decision later.

## Phase 1 — Dry-run conflict inventory

Analyze conflicts without dirtying the worktree:

```bash
git merge-tree --write-tree HEAD origin/main   # prints a tree OID + conflict list
```

The printed tree OID is a real, inspectable object: `git show <tree>:<path>` shows the auto-merged content of any file (including conflict markers), and files *not* in the conflict list are already final. This lets you fully analyze every conflict — and even draft resolutions — before deciding how to merge. For each conflicted file, extract both sides and the base (`git show <base>:<path>` etc.) and answer: what did the fork change here, what did upstream change, which changes are the same idea in different shapes.

## Phase 2 — Strategy: classify, then decouple before merging

For every conflicted file, classify the divergence:

- **Upstream subsumed it** → take upstream's side wholesale; delete the fork's version, including its call sites and tests that assert the old API. Do not keep "both" — that creates dead code and double behavior.
- **Fork-only feature** → keep the fork's *semantics*, but expect to re-express them on upstream's new structure (see "port, don't preserve" below).
- **Both changed, orthogonal** → union: take upstream's refactor and append the fork's additions at the right layer.
- **Rename/extraction** → upstream's names win everywhere, including fork call sites. The fork's old names must not survive.

### Pre-merge decoupling refactor (do this when conflicts are mechanical)

When a large share of the conflict surface comes from *mechanical* divergence — upstream renamed a symbol, extracted a module, moved a file, restructured a function the fork also touches — do **not** resolve those hunks by hand in the merge. Instead, apply the same change on the fork branch first, as a standalone refactor commit:

1. Port upstream's new names/module structure into the fork's code (copy the extracted file verbatim from upstream if it exists there), updating all fork call sites, **without changing any behavior**.
2. Verify: full build + the fork's relevant tests green.
3. Commit on the fork branch *before* merging.

After this, the real merge's conflicts shrink to the genuinely semantic ones — often most conflicts simply vanish from the merge-tree output. A behavior-preserving refactor is also far easier to review and to bisect than hunks resolved inside a merge commit, where they become invisible. Concrete example from this repo: upstream extracted `token_suffix` into a shared `xai_grok_auth::bearer_suffix` module; applying the identical rename on dev first (commit "adopt shared bearer_suffix ahead of upstream sync") eliminated most of the conflict surface in four files before the merge even started.

Skip this step when conflicts are few and deeply semantic — the refactor only pays off when it converts large mechanical conflicts into clean auto-merges.

### Structural decoupling: give fork code its own modules

The rename-adoption refactor above shrinks *this* merge's conflicts. A complementary move shrinks *every future* sync's conflicts: when the dry run shows the same files conflicting sync after sync because fork logic is interleaved with upstream code (long functions the fork extended inline, match arms the fork added variants to, structs the fork added fields to), refactor the fork side so its logic lives in fork-owned modules behind thin call-site glue.

Coupling is what makes merges expensive: a conflict hunk that mixes both sides' changes forces a semantic resolution every time upstream touches that region. The same logic in a fork-owned file with one-line call sites in upstream files conflicts rarely, and when it does, the resolution is mechanical (keep upstream's file, re-add the one-liner).

How to apply it on the fork branch before merging:

1. Identify the high-coupling spots from the dry run: files that conflicted in previous syncs *and* conflict now, where fork edits are inline rather than isolated.
2. Extract the fork's inline logic into fork-owned helpers/modules (e.g. `fork_feature.rs` with a clear API), leaving the smallest possible call in the upstream-owned file — one function call, one match delegation, one field with a fork-typed value.
3. Keep the refactor behavior-preserving; verify build + relevant tests; commit separately from the merge.
4. In the merge itself, upstream's version of the interleaved file is then safe to take as the base: re-applying the thin glue is trivial, and the fork's real logic never enters the conflict at all.

This repo shows the payoff: dev's Responses-compaction code lives almost entirely in fork-owned files (`client/responses_compact.rs`, `storage/responses_compaction.rs`, `session/cache_routing.rs`) and those files never conflict — the conflicts concentrate exactly where dev logic is inline in upstream files (recap.rs, copy.rs, rewind.rs). Each sync is an opportunity to move one more inline block behind a fork-owned boundary, so conflict surface trends downward instead of staying constant.

Judge the investment: decouple the spots with a history of repeated conflicts first; a file that conflicted once for an obvious one-off reason does not justify restructuring.

### Port, don't preserve

The most common trap: upstream refactored the exact function the fork's feature lives in (extracted helpers, new modules, changed signatures). Taking "ours" reverts upstream's refactor; taking "theirs" deletes the fork feature. Neither is acceptable. The resolution is a **port**:

1. Take upstream's version of the file as the base.
2. Re-apply the fork's deltas as small edits onto the new structure — which may mean adding a field to upstream's new struct, threading a parameter through upstream's new helper, or moving fork logic into upstream's new extension point.
3. When upstream added *new* features in the same area, wire the fork's concerns into them too (e.g. the fork's isolated cache-namespace scheme extended to upstream's new side-call kinds).
4. After resolving, grep the file and its call sites for fork-only symbols that upstream renamed or removed — leftover references are the classic post-merge compile error.

## Phase 3 — Execute the merge

1. `git merge origin/main`. Resolve each conflicted file per the Phase-2 strategy (`git checkout --theirs <path>` is a fine starting point for port-style resolutions).
2. Regenerate lockfiles (`cargo metadata > /dev/null`, etc.) rather than hand-resolving them.
3. Grep the tree for leftover conflict markers (`<<<<<<<`) and stage everything.
4. Guard the merge state: if anything (accidental `git stash`, shell restart) disturbs the worktree mid-merge, recovery is `git stash pop` plus manually re-writing `.git/MERGE_HEAD` (to `git rev-parse origin/main`), `.git/MERGE_MODE`, and `.git/MERGE_MSG`. Verify with `git status` showing "All conflicts fixed but you are still merging" before committing.

## Phase 4 — Verify, and attribute every failure

Build and test with **real exit codes**. A pipeline like `cargo check 2>&1 | grep -E '^error'; echo $?` reports grep's exit code, not cargo's — a silently failing build looks green. Use `PIPESTATUS`, `tail` the raw output, or drop the grep.

- Prefer targeted suites over full test runs: the modules you touched plus their direct consumers. A 6000-test debug-mode suite costing an hour verifies nothing extra about a 6-file conflict surface.
- **Every failure or hang must be attributed**: pre-existing fork breakage, or merge regression? The tools are git archaeology, not guesswork: `git log -S <symbol>` to find when the breaking change landed, `git show <pre-merge-commit>:<path>` to check whether the failing code path was identical before the merge, `git merge-base --is-ancestor` for chronology. If the failing chain (code + test + trigger) was identical at the pre-merge fork commit, it is pre-existing — say so explicitly, with the introducing commit.
- Pre-existing failures that are cheap and clearly attributable (e.g. a test that hangs because it never drains a channel a feature now waits on) are worth fixing in the merge when the fix is obvious — it leaves the branch greener than both parents. Ones that need design decisions get documented in the merge commit message and left alone.
- Environment gotchas in this repo: tests need `RUST_MIN_STACK=67108864` (deep debug futures stack-overflow otherwise); `/tmp` is tmpfs with non-insertion readdir order (sort before truncating directory listings in tests); never `pkill -f <crate-name>` — the pattern matches the invoking shell's own cmdline and kills it; use a character-class trick like `pkill -f "name[-]"`. Rust debug builds can fill the disk — check `df -h` when the compiler dies with `No space left on device`.

## Phase 5 — Commit with an audit trail

The merge commit message is the only place where the resolution *reasoning* is recorded. Write it so a future sync can reconstruct why each decision was made:

- For each conflicted file: which side was taken as the base, what was ported/dropped/added, and why.
- Any fork APIs deleted because upstream subsumed them (name the upstream replacement).
- Pre-existing failures found, each with its introducing commit and evidence it is not a regression.
- Verification performed: build status, which suites ran, results.

## Final report

Close by telling the user: the merge commit OID, a table of conflicts and how each was resolved, what upstream features were adopted, what fork features were preserved and where they were ported, test results, and the list of pre-existing issues discovered (each marked clearly as pre-existing, not caused by this merge).
