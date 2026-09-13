# Contributing to oxmera

Thank you for your interest. This project has unusual rules; please read
them before opening a PR — they are enforced by CI, not by goodwill.

## Every commit

- **DCO sign-off and a cryptographic signature.** `git commit -sS` is the
  only spelling. The `Signed-off-by:` trailer must match the author.
- **Conventional Commits**: `feat:`, `fix:`, `docs:`, `test:`, `ci:`,
  `chore:`, `refactor:`, `perf:`, plus `exercise:` for changes to the
  exercise ladder.
- **Zero AI attribution — anywhere, ever.** No `Co-Authored-By` naming any
  AI, bot, or agent; no "Generated with …"; no 🤖; no `*[bot]` authors — in
  commits, PRs, comments, docs, or exercise solutions. CI enforces this over
  full history.
- **AI-tooling policy**: use whatever tools you like while working. What
  lands carries no AI attribution, and your sign-off asserts it as your own
  work under the DCO.
- Green `just ci` before pushing — gate on the **exit code**
  (`just ci && git commit -sS …`), never on a pipeline's tail.

## Scope of contributions

The original learning boundary was repealed on 2026-08-22 (ADR-0006): PRs
implementing functionality are welcome at production quality. Ground rules:
new ops need tests against the CPU reference and, where differentiable, a
gradcheck entry; Metal changes need a parity test; terminal changes need a
golden (blessed from a frame you verified by eye) and must keep the
determinism contract — no clocks, durations, or absolute paths in
assertable regions.

## The dependency firewall

No crate in the stable workspace may depend, directly or transitively, on
`cuda-oxide`, `reconverge`, or `launchbound` — the research CUDA toolchain
under `research/oxmera-cuda-oxide`. `deny.toml` bans them and CI fails on
violations. The shipped CUDA backend reaches the driver through `cudarc`,
which is not in those families. Adding any dependency to a firewalled crate needs a
maintainer's explicit sign-off. New dependencies anywhere must satisfy the
`deny.toml` license allowlist.

## The pin policy

Four pins move together or not at all, each bump in its own commit with no
behaviour change riding along:

| component | current pin |
|---|---|
| nightly | `nightly-2026-04-03` |
| reconverge | 0.5.0 |
| cuda-oxide | rev `a766fc26` |
| launchbound | 2.1.0 / action `@v2` (SHA-pinned) |

**Regenerating the CUDA PTX is `just ptx`**, and it is the only recipe that
needs `nvcc` on PATH. It writes `crates/oxmera-cuda/kernels.ptx` and
`kernels.ptx.source` **together** — commit both. Editing `kernels.cu` and
running anything less fails
`the_shipped_ptx_was_built_from_the_shipped_source`, which exists because
the crate ships the kernels twice and the driver picks between them at run
time.

A bump must touch, together: `research/rust-toolchain.toml` (nightly), the
`cuda-oxide` rev in the research workspace manifests, the
`reconverge-version` and `toolchain` inputs of the launchbound action in
`.github/workflows/`, the pin table in `ARCHITECTURE.md`, this table, and
the enumerated family ban lists in `deny.toml` — cargo-deny bans are exact
names, so any crate the new version adds to a family must be added there.
If a bump changes a measured baseline, `docs/research-baseline.md` must be
re-measured, not edited.

The cuda-oxide bump target is **not upstream HEAD**. reconverge is a rustc
driver and must be built by the same nightly cuda-oxide needs, so the only
cuda-oxide rev this project can take is the one the pinned reconverge
release records as verified (`conformance/PIN` and `rust-toolchain.toml`
in its tag). The pins watch (`.github/workflows/pins.yml`) compares against
that and reports upstream HEAD for information only; when HEAD needs a
newer nightly than reconverge is built on, the pin waits for a reconverge
release, not the other way round.

`termlens` (currently 0.11) pairs with nothing and moves alone, in its own
commit, gated by the PTY suites and both 100-iteration stresses. A bump must
touch, together: the dev-dependency in `crates/oxmera-cli/Cargo.toml` (the
only manifest that names it), `Cargo.lock` (three jobs run `--locked`),
`PIN_TERMLENS` in `.github/workflows/pins.yml`, every `cli-version:` input of
the termlens report action under `.github/workflows/`, this line, and the
vendored agent skill `.claude/skills/termlens/SKILL.md` — copied verbatim from
`termlens/skills/termlens/SKILL.md`, since the published crate does not ship
it. `.github/scripts/check-skill-version.sh` fails CI when any of them
disagrees with the dependency, because every one of these is silent when it
goes stale: a stale skill hands every agent working here the idioms of a
release that is gone, and a stale `PIN_TERMLENS` makes the Monday pins watch
file a drift issue about a bump that already happened. The 0.11 bump moved
three of them and left the other two behind, which is why the check now covers
the whole list rather than the skill alone.

## Testing policy

- Unit tests on everything with a shape; property tests (`proptest`) where a
  law exists; golden tests (`insta`) for serialized output; terminal tests
  (`termlens`) through a real PTY on hermetic fixtures.
- Hardware tests live behind a `hardware` feature and `#[ignore]`
  (`cargo test -p oxmera-cuda --features hardware` on a CUDA machine; the
  Metal parity suite runs wherever a Metal device exists). CI is green with
  no GPU present — that *is* the no-GPU path's regression test.
- MSRV is measured, not declared: `rust-version` reflects what the
  dependency graph requires, verified in CI against the committed lockfile,
  raised in its own commit when a dependency forces it.

## Release checklist (maintainer)

1. `docs/LIMITATIONS.md` and the README limitations section are current.
2. Clean tree; `just ci` green by exit code.
3. Versions bumped in their own commit; `CHANGELOG.md` updated.
4. Tag `vX.Y.Z` matching the crate version exactly — the release workflow
   refuses a mismatch.
5. Publishing uses crates.io Trusted Publishing (OIDC). There is no
   registry token to leak, so do not create one. A brand-new crate joining
   the workspace needs its trusted publisher configured on crates.io after
   its first publish — ADR-0005 records the one-time procedure used for
   the initial ten.

## License

By contributing, you agree that your contributions are dual-licensed under
MIT or Apache-2.0, at the user's option, and you certify the
[Developer Certificate of Origin](https://developercertificate.org/) via
your sign-off.
