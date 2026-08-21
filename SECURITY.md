# Security Policy

## Supported versions

Nothing is released yet. When releases exist, only the latest published
version of each crate is supported.

## Reporting a vulnerability

Please use GitHub's **private vulnerability reporting** on this repository
(Security → Report a vulnerability). Do not open a public issue for a
security problem.

You can expect an acknowledgement within a week. This is a part-time,
single-maintainer learning project — severity will be triaged honestly, and
fixes for anything real will be prioritized over roadmap work.

## Scope notes

- This project ships no network services and no unsafe code in the stable
  workspace (`#![forbid(unsafe_code)]`).
- Supply-chain policy: `cargo deny` runs in CI (advisories, bans, licenses,
  sources); Dependabot alerts and security updates are enabled; workflows
  default to `permissions: contents: read`.
- Registry token policy: releases use crates.io Trusted Publishing (a
  short-lived token minted from GitHub OIDC per run). **No long-lived
  registry token exists anywhere.** The one-time first-publish exception
  ADR-0005 documented was retired immediately after `v0.0.1`: the secret
  was deleted and the token revoked.
