# linkedparticles-core

> **Particles is shared memory for humans and AI agents.** Each particle is one
> claim, plus what you need to judge it: who said it, where, when, and how
> confident they were. Facts, opinions, and memories are all claims, recorded
> the same way as particles. Particles are not edited or deleted. Particles are
> superseded, retracted, or disputed in the open. How much to trust it is a
> perspective applied at query time, never baked into the record.

`linkedparticles-core` is the store-free **Client layer** of the Particles
reference implementation: the schema and its invariants, the validators, the
wire format, and the extraction pipeline. Everything you need to *make* and
*check* particles — with no database, no graph, and no accumulated state.

> ### Most people want the other package
>
> If you want to **run** a knowledge store — deposit sources, extract beliefs,
> query with provenance, lint for contradictions, plus an HTTP API, a CLI, and
> an MCP server — install
> **[`linkedparticles`](https://pypi.org/project/linkedparticles/)** instead.
> It depends on this package and pulls it in automatically. This one is the
> substrate underneath it.

## Install

```bash
pip install linkedparticles-core
```

Python 3.11+.

## What it gives you

```python
from particles.conformance.jsonschema import validate_particle_dict
from particles.core.schema import Confidence, Particle, UncertaintyNature
from particles.interchange import from_unit, to_unit

p = Particle(
    content="Pluto is a dwarf planet.",
    confidence=Confidence(value=0.95),
    uncertainty_nature=UncertaintyNature.EPISTEMIC,
    asserted_by="iau-2006",
)

validate_particle_dict(p.model_dump(mode="json"))   # [] — valid against the normative JSON Schema
unit = to_unit(p, [])                               # JSON-LD interchange unit
from_unit(unit)                                     # …and back again
```

```json
{
  "@context": "https://linkedparticles.org/schemas/context.jsonld",
  "@type": "Particle",
  "formatVersion": "1.0",
  "schemaVersion": "1.0.0",
  "sourceParticleId": "632994da-f26d-4b4a-b288-9d9ad8f22856",
  "particleType": "CLAIM",
  "content": "Pluto is a dwarf planet.",
  "confidenceValue": 0.95,
  "calibrationSource": "EXTRACTOR_DIRECT",
  "uncertaintyNature": "EPISTEMIC",
  "assertedBy": "iau-2006",
  "assertedAt": "2026-08-23T22:42:34.936574+00:00",
  "status": "ACTIVE",
  "provenance": [],
  "subjects": []
}
```

That `@context` is a live identifier: it resolves, byte for byte, to the
normative artifact published at
[linkedparticles.org](https://linkedparticles.org).

## What is in here

- **The schema models and their invariants** — a stored `confidence.value` is
  immutable, status moves only through the validated transition table, and the
  Core/Extension split is enforced rather than documented.
- **The extraction layer** — source bytes to candidate particles with
  confidence, uncertainty, provenance, and unresolved subject names. Extractor
  plugins register here.
- **Conformance validation** — the normative JSON Schema and the five SHACL
  shapes, shipped inside the wheel, so validation works from an installed
  package and not only from a checkout.
- **The interchange codec** — pure JSON-LD / JSON Lines / YAML-LD, no I/O.
- **The completion-provider port**, embeddings, confidence math, and the
  validating HTTP transport.

Anything that *holds or reasons over accumulated state* — reconciliation,
subject resolution, querying, linting, review — is deliberately absent. That
lives in [`linkedparticles`](https://pypi.org/project/linkedparticles/).

## Reach for this package alone when

- you are **validating or exchanging** particles produced elsewhere, and do not
  want a store, a driver, or a migration path;
- you are **producing** particles for someone else's engine — an extractor, an
  importer, a service that emits interchange units;
- you are **implementing the standard** in your own system and want the
  reference schema, validators, and conformance artifacts to check against;
- you already depend on `linkedparticles` and want to import from the Client
  half explicitly.

Both distributions ship under the same import package, `particles`, split along
the Client/Engine line, and always at the same exact version. Installing the
engine installs this; the reverse is not true, and importing a store-layer
module from a core-only install is a clean `ImportError` rather than a surprise.

## Security posture

The two controls that matter most in an agent-memory stack ship **in this
package**, because this is where untrusted bytes meet the model:

- **Prompt-injection fencing.** Every LLM call site that touches
  attacker-controllable text keeps trusted instructions in the system turn and
  wraps the untrusted material in a per-call, 128-bit-nonce data fence, with a
  JSON contract enforced at the parser behind it. The nonce is unguessable, so
  injected text cannot forge a closing delimiter. This raises the bar
  materially. It is hardening, not immunity.
- **A validating fetch transport.** Outbound requests resolve the host, check
  the address against a blocklist, and connect to *that vetted address* —
  re-resolved and re-validated on every redirect hop, closing DNS rebinding and
  redirect SSRF rather than only the first lookup.

Behind both: the model is never given tools, and no model output is ever
executed. Nothing it emits becomes a shell command, a SQL fragment, or a fetch,
so an injection can at worst distort *claims* — never trigger *actions*.

The whole package was audited adversarially before it was opened; the verdict,
verbatim, was **GO-WITH-FIXES** with **33 findings — 2 High, 7 Medium, 20 Low,
4 Info** — the ranked must-fix set merged the following day, and the last open
finding closed in `v1.128.0`. Trust model, known limitations, and how to report
a vulnerability privately:
[SECURITY.md](https://github.com/LinkedParticles/particles-core-py/blob/main/SECURITY.md).

## Documentation

The SDK documentation at
**[docs.linkedparticles.org](https://docs.linkedparticles.org)** covers both
distributions — its [API reference](https://docs.linkedparticles.org/api/schema/)
spans the Client and Engine halves together. The standard itself — whitepaper,
[technical specification](https://linkedparticles.org/spec/technical-specification/),
and the normative schema, context, and
[vocabulary](https://linkedparticles.org/vocab/) artifacts — is published at
**[linkedparticles.org](https://linkedparticles.org)**.

## The three repositories

| Repo | What it is |
|---|---|
| [`particles-standard`](https://github.com/LinkedParticles/particles-standard) | The standard: whitepaper, technical specification, normative schema + SHACL artifacts, conformance fixtures |
| [`particles-core-py`](https://github.com/LinkedParticles/particles-core-py) | **This repo** — the Python Client layer (`linkedparticles-core`) |
| [`particles-engine-py`](https://github.com/LinkedParticles/particles-engine-py) | The Python Engine layer + surfaces ([`linkedparticles`](https://pypi.org/project/linkedparticles/)) |

## Contributing

See
[CONTRIBUTING.md](https://github.com/LinkedParticles/particles-core-py/blob/main/CONTRIBUTING.md)
and
[ARCHITECTURE.md](https://github.com/LinkedParticles/particles-core-py/blob/main/ARCHITECTURE.md).
Contributions are accepted under a Developer Certificate of Origin sign-off —
there is no CLA.

## License

Apache-2.0. See
[LICENSE](https://github.com/LinkedParticles/particles-core-py/blob/main/LICENSE)
and
[NOTICE](https://github.com/LinkedParticles/particles-core-py/blob/main/NOTICE).
