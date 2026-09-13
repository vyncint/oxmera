#!/usr/bin/env bash
# Everything that names a termlens version must name the one we depend on.
#
# termlens pairs with nothing and moves alone, so its version is written down
# in five places (CONTRIBUTING.md lists them). The 0.11 bump moved three and
# left two — `PIN_TERMLENS` and CONTRIBUTING's own line — at 0.10.1, where
# nothing would have noticed until the Monday pins watch filed a drift issue
# about a bump that had already happened (#68). Each of these is silent when
# it goes stale, which is why the check covers the list rather than one entry.
#
# `.claude/skills/termlens/SKILL.md` is a copy of the file termlens ships for
# coding agents. It is refreshed by hand, and the failure mode is silent: the
# dev-dependency gets bumped, the copy does not, and every agent working in
# this repository is then handed guidance for a version that is no longer here
# — wrong signatures, absent APIs, advice that was true one release ago. The
# 0.9 copy taught `assert_screen_snapshot!(screen)` and `mouse_mode()`, both
# of which 0.10 changed, and nothing in CI noticed.
#
# Nothing can diff it against upstream: the published crate does not ship the
# skill, so there is no registry copy to compare with. What *is* checkable is
# that the two versions agree, which is exactly the drift that happens.
#
# Compares major.minor only. A termlens patch release does not rewrite the
# skill, and demanding a re-copy for every one of them would make this noise.
#
# Usage: check-skill-version.sh [SKILL.md] [Cargo.toml] [pins.yml] [CONTRIBUTING.md] [Cargo.lock]
set -euo pipefail

skill="${1:-.claude/skills/termlens/SKILL.md}"
# termlens is a dev-dependency of exactly one crate, not a workspace
# dependency (see CONTRIBUTING.md: it pairs with nothing and moves alone).
manifest="${2:-crates/oxmera-cli/Cargo.toml}"
pins="${3:-.github/workflows/pins.yml}"
contributing="${4:-CONTRIBUTING.md}"
lock="${5:-Cargo.lock}"

for f in "$skill" "$manifest" "$pins" "$contributing" "$lock"; do
  [ -f "$f" ] || { echo "::error::$f does not exist"; exit 1; }
done

# "Written against **termlens 0.10.1**." -> 0.10
skill_version="$(sed -n 's/.*Written against \*\*termlens \([0-9][0-9.]*\)\*\*.*/\1/p' "$skill" | head -1)"
[ -n "$skill_version" ] || {
  echo "::error::$skill has no 'Written against **termlens X.Y.Z**' line to check"
  exit 1
}
skill_minor="$(echo "$skill_version" | cut -d. -f1,2)"

# Both spellings, because either is a legitimate way to write the dependency:
#   termlens = { version = "0.10", features = ["serde"] }   ->  0.10
#   termlens = "0.10"                                       ->  0.10
dep_version="$(sed -n \
  -e 's/^termlens = .*version = "\([0-9][0-9.]*\)".*/\1/p' \
  -e 's/^termlens = "\([0-9][0-9.]*\)".*/\1/p' \
  "$manifest" | head -1)"
[ -n "$dep_version" ] || {
  echo "::error::no termlens dependency with a version found in $manifest"
  exit 1
}
dep_minor="$(echo "$dep_version" | cut -d. -f1,2)"

if [ "$skill_minor" != "$dep_minor" ]; then
  echo "::error::the vendored termlens skill is written against ${skill_version} but this repository depends on ${dep_version}."
  echo "::error::Refresh it: cp ../termlens/skills/termlens/SKILL.md ${skill}"
  exit 1
fi

echo "the vendored termlens skill (${skill_version}) matches the dependency (${dep_version})"

# The same drift, one layer out. The termlens `report` action takes the
# termlens-cli version as a literal in the workflow files, and a literal beside
# a dependency is a pin that goes stale silently: the suite would then be
# rendered by a tool from a different release than the library that produced
# the screens. Nothing else compares the two, so this does.
cli_pins="$(grep -rhoE 'cli-version: *"[0-9][0-9.]*"' .github/workflows/ 2>/dev/null | grep -oE '[0-9][0-9.]+' | sort -u)"
if [ -n "$cli_pins" ]; then
  for pin in $cli_pins; do
    pin_minor="$(echo "$pin" | cut -d. -f1,2)"
    if [ "$pin_minor" != "$dep_minor" ]; then
      echo "::error::a workflow pins termlens-cli ${pin} but this repository depends on termlens ${dep_version}."
      echo "::error::Bump every 'cli-version:' under .github/workflows/ to match."
      exit 1
    fi
  done
  echo "the termlens-cli pins in .github/workflows ($(echo "$cli_pins" | tr '\n' ' ')) match the dependency (${dep_version})"
fi

# `PIN_TERMLENS` is what the weekly pins watch compares with crates.io's newest
# release, so it has to name the version this repository actually resolves —
# not the requirement, which is a range that several releases satisfy. When the
# two part company the watch reports drift toward a version we already have and
# opens an issue for a bump that is already done. Exact, not major.minor,
# because exact is what the watch compares.
locked="$(sed -n '/^name = "termlens"$/,/^$/ s/^version = "\([0-9][0-9.]*\)"/\1/p' "$lock" | head -1)"
[ -n "$locked" ] || {
  echo "::error::no resolved termlens version found in $lock"
  exit 1
}

pin_version="$(sed -n 's/^ *PIN_TERMLENS: *"\{0,1\}\([0-9][0-9.]*\)"\{0,1\} *$/\1/p' "$pins" | head -1)"
[ -n "$pin_version" ] || {
  echo "::error::no PIN_TERMLENS: X.Y.Z line found in $pins"
  exit 1
}

if [ "$pin_version" != "$locked" ]; then
  echo "::error::$pins pins termlens ${pin_version} but $lock resolves ${locked}."
  echo "::error::Set PIN_TERMLENS to ${locked}, or the Monday pins watch files drift for a bump that already happened."
  exit 1
fi

echo "PIN_TERMLENS (${pin_version}) matches the resolved dependency (${locked})"

# CONTRIBUTING's termlens paragraph opens by naming the current version, and it
# is the list every bump is supposed to follow. A list that misstates its own
# subject is the least likely thing to be reread.
doc_version="$(sed -n 's/.*`termlens` (currently \([0-9][0-9.]*\)).*/\1/p' "$contributing" | head -1)"
[ -n "$doc_version" ] || {
  echo "::error::$contributing has no '\`termlens\` (currently X.Y)' line to check"
  exit 1
}
doc_minor="$(echo "$doc_version" | cut -d. -f1,2)"

if [ "$doc_minor" != "$dep_minor" ]; then
  echo "::error::$contributing says termlens is currently ${doc_version} but this repository depends on ${dep_version}."
  echo "::error::Update that line; it is the checklist every termlens bump follows."
  exit 1
fi

echo "$contributing (${doc_version}) matches the dependency (${dep_version})"
