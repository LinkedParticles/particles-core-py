"""Client-layer rendering utilities.

Pure, store-free rendering of particles / lint reports / digests into Markdown,
plus the filesystem-safe write and subject-slug helpers shared across the SDK.
Both the Engine *reasoning* layer (``operations``) and the Engine *output*
layer (``exporters``) depend on this package **downward**, so it must never
import Engine code (enforced by the import-linter contract).

Import the concrete helpers from :mod:`particles.render.markdown`.
"""

from pkgutil import extend_path

# Straddling package: `markdown` ships in `linkedparticles-core`, which owns
# this file, while `article_synthesis` ships in `linkedparticles`. See the note
# in `particles/__init__.py` (D1) — a top-level `extend_path` alone
# leaves this package regular, so `particles.render.article_synthesis` stays
# invisible across separate `sys.path` roots without these two lines.
__path__ = extend_path(__path__, __name__)
