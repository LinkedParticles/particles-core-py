# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Back-compat re-export — :func:`schemas_dir` now lives in ``particles.core``.

It moved there so ``extraction`` could reach it too: ``extraction``
may not import ``conformance`` (``conformance.validator`` already imports the
extractor registry, so that edge would close a new subpackage cycle against the
``acyclic_siblings`` contract), and both packages need to locate
``context.jsonld``. Import from :mod:`particles.core._resources` in new code.
"""

from __future__ import annotations

from particles.core._resources import schemas_dir

__all__ = ["schemas_dir"]
