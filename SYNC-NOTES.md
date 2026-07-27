# Sync notes: `integration/static-embeddings-and-trigram` onto `upstream/main`

Date: 2026-07-25
Performed locally only. **Nothing was pushed to `origin`.**

## Summary

The fork's 9-commit integration branch (7 non-merge commits) was rebased onto
`upstream/main`, which had moved 145 commits ahead (rmcp 0.6→1.7 migration,
`CK_INDEX_DIR`, lexical chunk line-spans, lenient tantivy query parsing,
`--hidden` flag, mixedbread ONNX embedding provider, C/C++/Markdown/Dart/Elixir
chunking, tree-sitter bumps, Windows CI fixes, etc.).

Result: **clean rebase, all conflicts resolved, `cargo build --release`
clean, 284/284 tests passing (9 pre-existing ignored), full functional smoke
test passed** (semantic search, trigram-accelerated regex, glob-filtered
search, hybrid search, multi-model ensemble indexing/search, MCP server
`initialize` handshake).

**Post-sync addition (commit `4eb9721`, on top of the 7 rebased commits):**
a real bug surfaced by vault-p1 using this rebuilt binary on their P1 vault —
`build_trigram_index()` computed its target directory as a raw
`repo_root.join(".ck")`, bypassing `ck_core::index_dir()`'s `CK_INDEX_DIR`
relocation that the manifest/sidecars/lock already honor. Under
`CK_INDEX_DIR`, nothing ever creates that in-tree `.ck` path, so every
trigram write failed with ENOENT, silently — the warning also only logged
anyhow's outermost context string, never the underlying `io::Error`. Fixed
both (resolve+create the CK_INDEX_DIR-aware directory; log the full error
chain), confirmed against vault-p1's exact repro description and a local
minimal repro (fresh corpus, `CK_INDEX_DIR` set, no in-tree `.ck` ever
created — trigram write now succeeds, search over it returns correct
results). Full test suite re-verified after this fix: still 284/284.

Before touching anything, every branch in scope was snapshotted:
`backup/<name>-pre-sync-20260725` (plus the pre-existing
`backup/static-embeddings-and-trigram`).

## Divergence map (fork commits vs. upstream overlap risk)

The fork's 7 commits touch: `ck-embed` (static embeddings, model2vec-style,
new crate-internal deps), `ck-trigram` (new crate, roaring-bitmap regex
acceleration), `ck-index`/`ck-engine`/`ck-cli::main` (glob include-filter,
symlink following, PDF stack-overflow guard, multi-model manifest + RRF
ensemble search). Upstream's 145 commits rewrote almost every one of those
same files structurally:

| File | Fork touches | Upstream churn (lines) |
|---|---|---|
| `ck-engine/src/lib.rs` | model resolution, index-update dispatch | +1646/-  |
| `ck-index/src/lib.rs` | manifest, walker, embeddings | +1044/- |
| `ck-cli/src/main.rs` | CLI flags, status display | +620/- |
| `ck-cli/src/mcp_server.rs` | `SearchOptions` construction | +567/- |
| `ck-core/src/lib.rs` | `FileCollectionOptions`/`SearchOptions` | +466/- |
| `ck-embed/src/lib.rs` | embedder dispatch | +149/- |

This was correctly a high-risk zone — not because the *features* collided
(they didn't; every fork feature is additive and orthogonal to what upstream
built), but because upstream's structural rewrite of the model-resolution and
model-config plumbing meant almost every fork hunk landed in code that no
longer existed verbatim, requiring hand re-integration rather than a
mechanical patch apply.

## Conflicts hit and how resolved

Rebase applied cleanly through Cargo.lock/toml plumbing for the first 5
commits (mostly additive: new `ck-trigram` crate, new `ck-embed` static
embedder module, CLI flags, symlink/PDF fixes) — those needed field-name
merges (`show_hidden` + `glob_patterns` both landing in
`FileCollectionOptions`, `.hidden(true)` merging with upstream's
`.hidden(!options.show_hidden)`, etc.) but no architectural rework.

The real hairiness was the 6th commit, **"Add multi-model embedding support
with ensemble search"** (665 lines), and its follow-up bugfix. Specific
issues, in order of how they were found:

1. **Two different `ResolvedModel` shapes.** Upstream had already refactored
   `ResolvedModel` to `{ alias, config: ModelConfig }` with `.canonical_name()`
   / `.dimensions()` accessor methods. The fork's diff assumed an older
   `{ canonical_name, alias, dimensions }` field-literal shape. Git's
   line-based 3-way merge auto-merged large chunks of fork code (the `"all"`
   ensemble placeholder, the `resolve_model` closure, `find_model_entry`
   *calls*) into the file **without flagging a conflict**, because upstream's
   struct definition sat in a part of the file the fork commit didn't touch
   directly — but the auto-merged fork code still referenced the old field
   names. This would have been an invisible compile break if I'd trusted git's
   "no conflict" as clean. Fixed by converting every fork struct literal to
   the `{alias, config}` shape and re-adding the (upstream-superseded but
   still-needed) `find_model_entry` helper, which upstream's own equivalent
   logic (`ModelRegistry::resolve`) doesn't fully substitute for at every call
   site.
2. **`legacy_model_config()` (both the `ck-engine` and `ck-index` copies)
   under-populated `ModelConfig`.** Upstream's copy predates the fork's
   `mrl_dims`/`default_threshold` fields (added by the static-embeddings
   commit for MRL truncation + model-specific thresholds). Both files'
   helper still only set the original 5 fields, which auto-merged silently
   and would not compile once the ensemble commit's code paths executed it.
   Also found and fixed the same gap in the `mxbai-xsmall` `ModelConfig`
   literal in `ck-models/src/lib.rs` (upstream's mixedbread model entry,
   also missing the fork's two newer fields). Caught by a project-wide
   struct-literal scan, not by the compiler alone — worth flagging because a
   narrower "fix what `cargo check` complains about" pass would have hit
   these one at a time across three files.
3. **Real semantic conflict: model-mismatch error vs. multi-model
   coexistence.** Upstream's `resolve_model_from_root` /
   `index_directory_inner` / `smart_update_index_with_detailed_progress`
   *error* when the requested model doesn't match what's indexed. The fork's
   entire point in this commit is to drop that error and instead register
   additional models via `manifest.add_model(...)`, tracking `is_new_model`
   so only the new model's embeddings get computed. Resolved by keeping
   upstream's *better* model-resolution logic (it already correctly prefers
   the manifest's existing model over the hardcoded default when `--model`
   is omitted — using `ModelRegistry::resolve()` with a `legacy_model_config`
   fallback for old/unregistered model names) and layering the fork's
   multi-model semantics on top (no error; `add_model()`; `is_new_model`
   computed via `manifest.has_model()`).
4. **The 7th commit ("fix: search without --model flag...") was a bugfix for
   a bug that no longer existed** in my merged code — upstream's model
   resolution already didn't have the "defaults to bge-small before checking
   the manifest" bug the fork patch describes, because I'd already adopted
   upstream's smarter resolution order while resolving commit 6. Verified
   this is genuinely redundant (not a different bug) by tracing
   `is_new_model`'s definition and its one call site
   (`ck-index/src/lib.rs:1053`); kept my (upstream-derived) code and dropped
   the fork's reimplementation for both hunks it touched.

**Superseded fork work:** none of the fork's *features* were superseded —
upstream never independently built glob-filtering, trigram indexing, static
embeddings, symlink-following, or multi-model ensemble search. What *was*
superseded, in the sense of "redundant reimplementation, prefer upstream's":
the fork's manual `find_model_entry`/`selected_model` model-resolution
plumbing in `ck-index::index_directory_inner`,
`smart_update_index_with_detailed_progress`, and
`ck-engine::resolve_model_from_root`, all replaced with upstream's
`ModelRegistry::resolve()` + `legacy_model_config()` idiom while preserving
the fork's multi-model/no-error/`is_new_model` behavior on top.

**Hidden-files feature:** the fork's `-g/--glob` flag is a distinct, additive
feature from upstream's independently-added `--hidden` flag (#176) and
`expand_glob_patterns` (shell-style glob expansion of file args on the
command line, not an include-filter). No overlap; both coexist in
`FileCollectionOptions` (`show_hidden` + `glob_patterns` fields).

**Conflict count:** 4 commits needed manual resolution (the trigram commit
had trivial Cargo.toml/test-file conflicts; the symlink/PDF commit had a
2-hunk conflict; the glob-flag commit had ~10 mechanical field-addition
conflicts; the ensemble commit was the hard one, ~8 conflict regions across
4 files, described above). None were unresolvable; all were reconciled by
understanding both sides' intent rather than blindly picking one.

**One hygiene fix:** an early `git add -A` accidentally swept two stray
untracked scratch docs (`BRANCH_SUMMARY.md`, `STATIC_EMBEDDINGS_ASSESSMENT.md`
— pre-existing local notes, not part of any commit's intended diff) into the
trigram commit. Caught and fixed via `git rebase -i` + `git rm --cached` +
amend before finishing; verified the rest of the rebase replayed cleanly on
top of the corrected commit. Both files are back to untracked/local-only.

## Feature branches

All 5 feature branches (`feature/static-embeddings`, `feature/trigram-indexing`,
`feature/glob-patterns`, `fix/glob-patterns-and-ctrl-c`,
`fix/symlinks-and-pdf-stack`) have tip commits that are direct ancestors of
the *old* integration branch tip (`95e53ef`) — 0 unique commits each. Their
entire content is fully contained in (and now rebased via) the integration
branch. Left as-is, per instructions, with backups already taken above; no
separate refresh needed since there's nothing to rebase on top of that isn't
already in `integration/static-embeddings-and-trigram`.

## Verification

- `cargo build --release --workspace --all-features`: clean, no errors, no
  warnings in the final build.
- `cargo test --workspace --all-features --release`: **284 passed, 0 failed,
  9 ignored** (ignored tests are pre-existing, network-dependent mixedbread
  model-download tests, unrelated to this sync).
- Functional smoke test (indexed `ck-core/src/lib.rs`, `ck-index/src/lib.rs`,
  `ck-engine/src/lib.rs`, `README.md` into a scratch dir):
  - `ck --index .` — indexed 4 files, built both the semantic embeddings and
    `trigrams.bin`.
  - `ck --sem "resolve embedding model configuration" .` — returned relevant
    matches (`ResolvedModel`, `resolve_model`, `legacy_model_config`).
  - `ck "fn resolve_model" .` and `ck "ModelRegistry" .` (regex/lexical,
    trigram-accelerated) — correct matches only in files containing the
    pattern.
  - `ck -g '*.rs' "fn main" .` and `ck -g '*.md' "ck" .` — glob include-filter
    correctly scoped results to the matching extension only.
  - `ck --hybrid "index manifest model" .` — sensible combined results.
  - Multi-model ensemble: indexed with `--model bge-small` (default), then
    `--index --model static-retrieval-en .` (adds a second model without
    erroring, manifest correctly lists both models under
    `embedding_models`); `--sem ... --model bge-small` and
    `--sem ... --model static-retrieval-en` both independently return
    results (confirming both models' embeddings coexist per-chunk, i.e. the
    second indexing pass didn't clobber the first model's embeddings);
    `--sem ... --model all` (RRF ensemble) also returned fused results.
  - `ck --serve` (MCP server): sent a raw JSON-RPC `initialize` request over
    stdio, got back a correct `2024-11-05` protocol response listing all six
    tools (`semantic_search`, `regex_search`, `hybrid_search`,
    `index_status`, `reindex`, `health_check`) — confirms the rmcp 1.7 API
    migration upstream did is compatible with the fork's MCP server changes
    (glob patterns wired into `mcp/context.rs`, `mcp/session.rs`,
    `mcp_server.rs`).

Note on the toolchain: this repo requires Rust 1.88+ (`ort`, `darling`,
`time` deps). The environment's Homebrew `cargo` is 1.86.0 and shadows
`rustup`'s 1.92.0 toolchain in `PATH`. All builds/tests above were run with
`PATH="$HOME/.rustup/toolchains/stable-aarch64-apple-darwin/bin:$PATH"`
prepended. Whoever picks this up next should either fix their `PATH` or keep
using that prefix — plain `cargo build` in a fresh shell will fail with a
misleading "rustc not supported" error otherwise.

## Nothing broke

No upstream feature was left non-functional and no fork feature regressed.
Every search mode (regex, semantic, lexical/trigram, hybrid, glob-filtered,
ensemble) and the MCP server work end-to-end post-merge.

## Push commands (NOT executed — operator's call)

The rewritten history means `origin`'s tracking branches need a
force-push. Since the 5 feature branches are fully superseded by (contained
in) `integration/static-embeddings-and-trigram` and were **not** locally
rewritten, they don't need pushing at all — only `integration/...` changed.

```bash
# From /Users/ochafik/github/ck, after reviewing `git log --oneline
# upstream/main..integration/static-embeddings-and-trigram` and this file:

git push --force-with-lease origin integration/static-embeddings-and-trigram

# The feature/* and fix/* branches were NOT rewritten (left untouched, with
# backups) since their content is fully contained in the rebased integration
# branch. No push needed for them. If you want origin's copies to visibly
# reflect that they're superseded (e.g. to prompt closing associated PRs
# without deleting the branches), that's a separate decision — not covered
# by a mechanical push command here.
```

Backups created this session (all local, not pushed):
- `backup/integration-static-embeddings-and-trigram-pre-sync-20260725` (old tip: `95e53ef`)
- `backup/feature-static-embeddings-pre-sync-20260725` (`8877274`)
- `backup/feature-trigram-indexing-pre-sync-20260725` (`23b7e84`)
- `backup/feature-glob-patterns-pre-sync-20260725` (`a9c5a31`)
- `backup/fix-glob-patterns-and-ctrl-c-pre-sync-20260725` (`ca64e76`)
- `backup/fix-symlinks-and-pdf-stack-pre-sync-20260725` (`d672940`)

## Determinism findings (librarian prototype, 26 Jul 2026) — two upstream-worthy bugs
Discovered while prototyping content-addressed index replication. A ck index is a
pure function of (content, model, ck_version, ABSOLUTE PATH), not content alone:
1. **trigrams.bin is non-deterministic** — 3 builds of identical content produce 3
   different hashes (unstable serialization order). Blocks content-addressing the
   index. Fix: stable/sorted trigram serialization.
2. **.ck payload is path-dependent** — the absolute index path leaks into the bytes,
   so an index built at /a can't be relocated to /b. Fix: a relocatable/relative-path
   index mode.
Both are prerequisites for sound content-addressed index-blob replication (iroh/mesh).
Neither blocks local-reindex use. Full analysis: homelab/prototypes/librarian/RESULTS.md.
