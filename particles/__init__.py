# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Particles — Python SDK for the Particles epistemic knowledge standard."""

from importlib.metadata import PackageNotFoundError, version
from pkgutil import extend_path

# Split-package build (D1). The `particles` import package is shipped
# by two distributions — `linkedparticles-core` (the Client layer, which owns
# this file) and `linkedparticles` (the Engine layer + surfaces) — so the
# package must resolve submodules that live in a *different* installed tree.
# `extend_path` buys namespace-package path semantics while keeping a real
# `__init__.py` to hold `__version__`, the re-exports, and the `py.typed`
# marker. Without it `particles.__path__` has exactly one entry and the Engine
# half is invisible whenever the two dists land in different `sys.path` roots
# (`pip install --target`, `--user` site, Lambda layers, editable installs).
# On a single-root install or a source checkout this is a no-op. Every package
# directory whose contents straddle the two dists needs the same two lines —
# today `particles/`, `particles/render/`, and `particles/interchange/`.
__path__ = extend_path(__path__, __name__)

# Keyed on the distribution that owns this file (D3), falling back to
# the unified distribution the private monorepo installs, then to a sentinel.
# The two published dists are pinned to one exact version, so in an engine
# install the core version *is* the engine version. The final fallback covers a
# source checkout, which carries no distribution metadata at all — previously
# an unconditional `PackageNotFoundError` out of `import particles`.
try:
    __version__ = version("linkedparticles-core")
except PackageNotFoundError:
    try:
        __version__ = version("linkedparticles")
    except PackageNotFoundError:  # source checkout / not installed
        __version__ = "0+unknown"
