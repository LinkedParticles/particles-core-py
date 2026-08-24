# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Numista-specific extractors and importer.

This package handles three Numista source surfaces:

- ``NUMISTA_API_COIN``    — ``NumistaCoinExtractor`` reads a single
  ``GET /types/{id}`` response (rich infobox, no LLM call).
- ``NUMISTA_API_ISSUER``  — ``NumistaIssuerExtractor`` reads a combined
  ``GET /types?issuer=…`` response stored by the importer.
- ``NUMISTA_LISTING_HTML`` — ``NumistaListingExtractor`` parses public
  catalogue listing HTML (no API key needed).

The importer (``NumistaImporter``, which dispatches between the coin and
listing paths) moved to the Engine layer at
``particles.ingest.importers.numista``.

The constants and classes re-exported here are the stable public surface —
``SOURCE_TYPE_*`` strings in particular are stored in the database and
must not change. See the `particles.extraction.AGENTS.md` package-layout
note for why this directory has the same shape as
`particles/extraction/github/`.
"""

from __future__ import annotations

from particles.extraction.numista._shared import (
    EXTRACTOR_ID_COIN,
    EXTRACTOR_ID_ISSUER,
    EXTRACTOR_ID_LISTING,
    EXTRACTOR_VERSION_COIN,
    EXTRACTOR_VERSION_ISSUER,
    EXTRACTOR_VERSION_LISTING,
    SOURCE_TYPE_COIN,
    SOURCE_TYPE_ISSUER,
    SOURCE_TYPE_LISTING,
)
from particles.extraction.numista.coin import NumistaCoinExtractor
from particles.extraction.numista.issuer import NumistaIssuerExtractor
from particles.extraction.numista.listing import NumistaListingExtractor

__all__ = [
    "EXTRACTOR_ID_COIN",
    "EXTRACTOR_ID_ISSUER",
    "EXTRACTOR_ID_LISTING",
    "EXTRACTOR_VERSION_COIN",
    "EXTRACTOR_VERSION_ISSUER",
    "EXTRACTOR_VERSION_LISTING",
    "NumistaCoinExtractor",
    "NumistaIssuerExtractor",
    "NumistaListingExtractor",
    "SOURCE_TYPE_COIN",
    "SOURCE_TYPE_ISSUER",
    "SOURCE_TYPE_LISTING",
]
