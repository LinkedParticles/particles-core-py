# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Syntactic URL canonicalization + harvesting.

Pure, **no-network** normalization of URLs found in source text, so that "the
same link, written ten ways" collapses to one canonical key for the
``url_mentions`` citation-signal store. Redirect-resolution — following a URL to
its real target — *is* crawling (explicit non-goal) and is out of
scope: opaque shorteners (``t.co``, ``bit.ly``) canonicalize to *themselves*,
never to their destination. Only *embedded-target* wrappers, where the
destination is already present in the URL string (Google ``/url?q=``, AMP),
are unwrapped, because that needs no network.

The rule sets are deliberately a **starter** (§ Deferred) and grow
additively under the "codify after the second instance"
discipline — add a tracking-param name or a wrapper pattern when a second real
instance shows up; never reach for the network.

This module is Client substrate: pure functions over strings, no
store and no graph. Both the Engine ingest pipeline (mention capture) and the
deposit reconciliation path import it; it imports nothing from either.
"""

from __future__ import annotations

import re
from urllib.parse import parse_qs, parse_qsl, unquote, urlencode, urlsplit, urlunsplit

# Query-parameter names that are pure tracking / analytics noise: dropping them
# never changes which resource the URL addresses. Matched case-insensitively.
_TRACKING_PARAMS: frozenset[str] = frozenset(
    {
        "fbclid",
        "gclid",
        "dclid",
        "gclsrc",
        "msclkid",
        "yclid",
        "mc_eid",
        "mc_cid",
        "igshid",
        "igsh",
        "_hsenc",
        "_hsmi",
        "vero_id",
        "oly_enc_id",
        "oly_anon_id",
        "ref",
        "ref_src",
        "ref_url",
        "referrer",
        "spm",
        "scm",
        "s_cid",
        "cmpid",
        "ncid",
        "soc_src",
        "soc_trk",
    }
)

# Prefix families: utm_source / utm_medium / …, plus the common analytics
# vendors. A param whose lowercased name starts with one of these is dropped.
_TRACKING_PREFIXES: tuple[str, ...] = (
    "utm_",
    "ga_",
    "_ga",
    "pk_",
    "mtm_",
    "hsa_",
    "matomo_",
)

# Default ports stripped when they match the scheme.
_DEFAULT_PORTS: dict[str, int] = {"http": 80, "https": 443}

# A pragmatic URL matcher for plain text / HTML: an http(s) URL up to the first
# whitespace, quote, angle bracket, backtick, or closing bracket — characters
# that almost always end a URL in prose or markup. Trailing sentence
# punctuation that the class does admit (``.`` ``,`` …) is peeled in
# :func:`canonicalize_url`.
_URL_RE = re.compile(r"""https?://[^\s"'<>)\]}`]+""", re.IGNORECASE)

# Trailing characters peeled off a harvested URL before parsing — sentence and
# markup punctuation that clings to the end of a match.
_TRAILING_TRIM = ".,;:!?’\"')]}>"

# Bound on wrapper-unwrap recursion (a wrapped wrapped URL is implausible but
# the guard keeps a crafted input from looping).
_MAX_UNWRAP_DEPTH = 3


def _is_tracking(key: str) -> bool:
    """True if a query-parameter name is tracking noise (exact or prefix)."""
    k = key.lower()
    if k in _TRACKING_PARAMS:
        return True
    return any(k.startswith(p) for p in _TRACKING_PREFIXES)


def _strip_query(query: str) -> str:
    """Drop tracking params, preserving the order and repetition of the rest."""
    if not query:
        return ""
    kept = [(k, v) for k, v in parse_qsl(query, keep_blank_values=True) if not _is_tracking(k)]
    return urlencode(kept)


def _unwrap(url: str) -> str | None:
    """Return the embedded target of a known wrapper URL, else ``None``.

    Only patterns that carry the destination *inside the URL string* are
    handled — no network. Recognized: Google redirect (``/url?q=`` /
    ``?url=``), Google AMP viewer (``/amp/s/``), and AMP CDN
    (``cdn.ampproject.org/c/s/``).
    """
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()

    is_google = host == "google.com" or host.endswith(".google.com")

    # Google redirect: https://www.google.com/url?q=<target>
    if is_google and parts.path == "/url":
        qs = parse_qs(parts.query)
        for key in ("q", "url"):
            values = qs.get(key)
            if values:
                cand = unquote(values[0])
                if cand.startswith(("http://", "https://")):
                    return cand

    # Google AMP viewer: https://www.google.com/amp/s/<target>
    if is_google and parts.path.startswith("/amp/"):
        m = re.match(r"^/amp/(s/)?(.+)$", parts.path)
        if m:
            scheme = "https" if m.group(1) else "http"
            target = f"{scheme}://{m.group(2)}"
            return f"{target}?{parts.query}" if parts.query else target

    # AMP CDN: https://<pub>.cdn.ampproject.org/c/s/<target>
    if host == "cdn.ampproject.org" or host.endswith(".cdn.ampproject.org"):
        m = re.match(r"^/c/(s/)?(.+)$", parts.path)
        if m:
            scheme = "https" if m.group(1) else "http"
            target = f"{scheme}://{m.group(2)}"
            return f"{target}?{parts.query}" if parts.query else target

    return None


def canonicalize_url(url: str, *, _depth: int = 0) -> str | None:
    """Canonicalize one URL syntactically; ``None`` if not a usable http(s) URL.

    Lowercases scheme + host, drops a default port and any ``#fragment``,
    strips tracking query params, normalizes a single trailing slash, and
    unwraps embedded-target wrappers (bounded recursion). Returns ``None`` for
    non-http(s) schemes, hostless URLs, and unparseable input — these are not
    citation signals.
    """
    if not url:
        return None
    url = url.strip().strip(_TRAILING_TRIM)
    if not url:
        return None
    try:
        parts = urlsplit(url)
    except ValueError:
        return None

    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        return None
    if not parts.netloc:
        return None

    # Unwrap embedded-target wrappers first (a wrapper's own host/params are
    # irrelevant once the real target is recovered).
    if _depth < _MAX_UNWRAP_DEPTH:
        target = _unwrap(url)
        if target is not None and target != url:
            return canonicalize_url(target, _depth=_depth + 1)

    host = (parts.hostname or "").rstrip(".")
    if not host:
        return None
    # Reconstruct netloc from host + non-default port only — credentials in the
    # authority are dropped (not part of resource identity).
    netloc = host
    if parts.port is not None and parts.port != _DEFAULT_PORTS.get(scheme):
        netloc = f"{host}:{parts.port}"

    # Path: keep "/" for root; strip a single trailing slash elsewhere so
    # "/a/" and "/a" collapse. Case is preserved (paths are case-sensitive).
    path = parts.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/") or "/"

    query = _strip_query(parts.query)
    # Fragment always dropped (client-side anchor; HTTP never sees it).
    return urlunsplit((scheme, netloc, path, query, ""))


def harvest_urls(text: str) -> list[str]:
    """Extract and canonicalize every external http(s) URL mentioned in ``text``.

    Returns canonical URLs de-duplicated in first-seen order. Pure and
    best-effort: matches that fail to canonicalize (non-http, malformed) are
    dropped silently. Works over plain text, HTML, and Markdown alike — the
    matcher stops at the quotes/brackets that bound URLs in attributes and link
    syntax.
    """
    if not text:
        return []
    seen: dict[str, None] = {}
    for raw in _URL_RE.findall(text):
        canon = canonicalize_url(raw)
        if canon is not None and canon not in seen:
            seen[canon] = None
    return list(seen)
