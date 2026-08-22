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

## The learning boundary

The computational parts of oxmera — tensor ops, kernels, autograd, layers,
optimizers, and **solutions to any exercise** — are the maintainer's to
write; that is the point of the project. PRs implementing them will be
declined regardless of quality. PRs improving the room — seams, types,
tests-as-specs, tooling, CI, docs — are welcome. An exercise test may assert
properties, invariants, and hand-computed constant cases; it may never
contain a working implementation of the thing under test.

## The dependency firewall

No crate in the stable workspace may depend, directly or transitively, on
`cuda-oxide`, `reconverge`, or `launchbound`. `deny.toml` bans them and CI
fails on violations. Adding any dependency to a firewalled crate needs a
maintainer's explicit sign-off. New dependencies anywhere must satisfy the
`deny.toml` license allowlist.

## The pin policy

Four pins move together or not at all, each bump in its own commit with no
behaviour change riding along:

| component | current pin |
|---|---|
| nightly | `nightly-2026-04-03` |
| reconverge | 0.3.0 |
| cuda-oxide | rev `a766fc26` |
| launchbound | 1.0.2 / action `@v1` |

A bump must touch, together: `research/rust-toolchain.toml` (nightly), the
`cuda-oxide` rev in the research workspace manifests, the
`reconverge-version` and `toolchain` inputs of the launchbound action in
`.github/workflows/`, the pin table in `ARCHITECTURE.md`, this table, and
the enumerated family ban lists in `deny.toml` — cargo-deny bans are exact
names, so any crate the new version adds to a family must be added there.
If a bump changes a measured baseline, `docs/research-baseline.md` must be
re-measured, not edited.

## Testing policy

- Unit tests on everything with a shape; property tests (`proptest`) where a
  law exists; golden tests (`insta`) for serialized output; terminal tests
  (`termlens`) through a real PTY on hermetic fixtures.
- Hardware tests live behind a `hardware` feature and `#[ignore]`. CI is
  green with no GPU present — that *is* the no-GPU path's regression test.
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
