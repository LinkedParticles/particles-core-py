# linkedparticles-core

> **Particles is shared memory for humans and AI agents.** Each particle is one
> claim, plus what you need to judge it: who said it, where, when, and how
> confident they were. Facts, opinions, and memories are all claims, recorded
> the same way as particles. Particles are not edited or deleted. Particles are
> superseded, retracted, or disputed in the open. How much to trust it is a
> perspective applied at query time, never baked into the record.

The store-free **Client layer** of the Particles reference implementation — the
substrate for defining, validating, serializing, and producing candidate
particles without a graph or a store. This package holds the pieces that never
touch persistent state:

- the schema models and their invariants (immutable confidence, the status
  machine, the Core/Extension split);
- the extraction layer that turns source bytes into candidate particles with
  unresolved subject names;
- confidence math, the completion-provider port, embeddings, conformance
  validation, and the pure interchange codec.

Holding or reasoning over accumulated state — reconciliation, subject
resolution, querying, linting — lives in the **engine** package
(`linkedparticles`), which depends on this one.

## Install

```bash
pip install linkedparticles-core
```

## The three repositories

| Repo | What it is |
|---|---|
| [`particles-standard`](https://github.com/LinkedParticles/particles-standard) | The standard: whitepaper, technical specification, normative schema + SHACL artifacts, conformance fixtures |
| [`particles-core-py`](https://github.com/LinkedParticles/particles-core-py) | **This repo** — the Python Client layer (`linkedparticles-core`) |
| [`particles-engine-py`](https://github.com/LinkedParticles/particles-engine-py) | The Python Engine layer + surfaces (`linkedparticles`) |

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and [ARCHITECTURE.md](ARCHITECTURE.md).
Contributions are accepted under a Developer Certificate of Origin sign-off —
there is no CLA.
