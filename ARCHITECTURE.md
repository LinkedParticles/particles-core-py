# Architecture — linkedparticles-core

This is the **Client layer**: everything you can do with particles *without* a
store or a graph. It is deliberately store-free, and that boundary is a checked
invariant, not an aspiration — an import-linter contract fails CI on any import
from this layer into the state-holding engine.

## What lives here

| Area | Responsibility |
|---|---|
| `core/` | Schema models; the status machine and its transition validator; confidence math (the value is calibrated once at creation and never mutated) |
| `config.py`, `secrets.py` | Configuration model and secret access |
| `embeddings.py`, `http.py`, `url_safety.py`, `url_canonical.py` | Shared client-side utilities |
| `llm/` | The completion-provider port and its wire-protocol adapters; every model call routes through here |
| `extraction/` | Turning source bytes into candidate particles with unresolved subject names — no graph, no store |
| `conformance/` | Schema + SHACL validation against the normative artifacts |
| `interchange.codec`, `interchange.jsonl` | The pure, store-free serialization codec |
| `render.markdown` | The store-free Markdown renderer |

## The one rule to know

**Client never reaches into the engine.** Producing a candidate particle,
validating it, and serializing it must work with no accumulated state. The
engine (`linkedparticles`) depends on this package and adds everything stateful:
reconciliation, subject resolution, the store, query, and lint.

## One import package, two distributions

This distribution and `linkedparticles` both ship modules under the `particles`
import package, so a consumer writes `from particles.core.schema import
Particle` without caring which wheel it came from. Exactly one distribution
ships each file, and this one owns `particles/__init__.py`,
`particles/py.typed`, and the `__init__.py` of the two packages the layers
split (`particles/render/`, `particles/interchange/`) — it is the dependency,
so it is always present when the engine is. Each of those calls
`pkgutil.extend_path`, which is what lets the engine's modules resolve from a
different directory.

Two things follow. `particles.interchange` re-exports the store-free half only
(the codec and the JSONL / YAML-LD containers); the store-aware
export/import functions live in `particles.interchange.store` and arrive with
the engine. And the two distributions are version-locked — `linkedparticles`
pins `linkedparticles-core` to an exact version — so they always advance
together.

## Why it is split out

The standard is not one implementation. Keeping the produce/validate/serialize
substrate as its own distribution lets a second-language implementation reuse
the same conceptual boundary, and lets tools depend on the schema layer without
pulling in a database. What makes any implementation *conforming* is the
conformance suite in the standard repository — not this repository's shape.

For the normative definitions behind the invariants named here (the confidence
math, the status machine, the interchange format), see the technical
specification in
[`particles-standard`](https://github.com/LinkedParticles/particles-standard).
