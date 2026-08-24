# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for syntactic URL canonicalization + harvesting.

Pure functions, no network — :func:`canonicalize_url` and
:func:`harvest_urls` in ``particles/url_canonical.py``.
"""

from __future__ import annotations

import pytest

from particles.url_canonical import canonicalize_url, harvest_urls


class TestCanonicalizeBasics:
    def test_scheme_and_host_lowercased(self) -> None:
        assert canonicalize_url("HTTPS://Example.COM/Path") == "https://example.com/Path"

    def test_path_case_preserved(self) -> None:
        # Paths are case-sensitive; only scheme + host fold.
        assert canonicalize_url("https://example.com/AbC") == "https://example.com/AbC"

    def test_fragment_dropped(self) -> None:
        assert canonicalize_url("https://example.com/a#section") == "https://example.com/a"

    def test_default_port_dropped(self) -> None:
        assert canonicalize_url("https://example.com:443/a") == "https://example.com/a"
        assert canonicalize_url("http://example.com:80/a") == "http://example.com/a"

    def test_nondefault_port_kept(self) -> None:
        assert canonicalize_url("https://example.com:8443/a") == "https://example.com:8443/a"

    def test_trailing_slash_normalized(self) -> None:
        assert canonicalize_url("https://example.com/a/") == "https://example.com/a"

    def test_root_slash_preserved(self) -> None:
        assert canonicalize_url("https://example.com") == "https://example.com/"
        assert canonicalize_url("https://example.com/") == "https://example.com/"

    def test_credentials_dropped(self) -> None:
        assert canonicalize_url("https://user:pw@example.com/a") == "https://example.com/a"

    def test_trailing_punctuation_peeled(self) -> None:
        assert canonicalize_url("https://example.com/a).") == "https://example.com/a"
        assert canonicalize_url("https://example.com/a,") == "https://example.com/a"


class TestCanonicalizeTrackingParams:
    def test_utm_family_stripped(self) -> None:
        assert (
            canonicalize_url("https://example.com/a?utm_source=x&utm_medium=y")
            == "https://example.com/a"
        )

    def test_known_click_ids_stripped(self) -> None:
        assert (
            canonicalize_url("https://example.com/a?fbclid=123&gclid=456")
            == "https://example.com/a"
        )

    def test_real_params_preserved(self) -> None:
        assert (
            canonicalize_url("https://example.com/a?id=7&utm_source=x")
            == "https://example.com/a?id=7"
        )

    def test_mixed_case_param_name_stripped(self) -> None:
        assert (
            canonicalize_url("https://example.com/a?UTM_Source=x&Q=1")
            == "https://example.com/a?Q=1"
        )


class TestCanonicalizeRejections:
    @pytest.mark.parametrize(
        "url",
        [
            "",
            "ftp://example.com/a",
            "mailto:bob@example.com",
            "javascript:alert(1)",
            "not a url",
            "https://",
            "/relative/path",
        ],
    )
    def test_non_http_or_malformed_rejected(self, url: str) -> None:
        assert canonicalize_url(url) is None


class TestCanonicalizeWrappers:
    def test_google_redirect_unwrapped(self) -> None:
        assert (
            canonicalize_url("https://www.google.com/url?q=https://target.example/story&sa=D")
            == "https://target.example/story"
        )

    def test_google_redirect_url_param(self) -> None:
        assert (
            canonicalize_url("https://google.com/url?url=https%3A%2F%2Ftarget.example%2Fp")
            == "https://target.example/p"
        )

    def test_amp_cdn_unwrapped(self) -> None:
        assert (
            canonicalize_url("https://example-com.cdn.ampproject.org/c/s/example.com/story")
            == "https://example.com/story"
        )

    def test_google_amp_viewer_unwrapped(self) -> None:
        assert (
            canonicalize_url("https://www.google.com/amp/s/example.com/story")
            == "https://example.com/story"
        )

    def test_opaque_shortener_not_resolved(self) -> None:
        # Resolving t.co would require a network round trip (non-goal):
        # the shortener canonicalizes to itself, never to its destination.
        assert canonicalize_url("https://t.co/AbCdEf") == "https://t.co/AbCdEf"


class TestHarvest:
    def test_harvest_from_prose(self) -> None:
        text = "See https://a.example/x and http://b.example/y for details."
        assert harvest_urls(text) == ["https://a.example/x", "http://b.example/y"]

    def test_harvest_from_html_attr(self) -> None:
        html = '<a href="https://a.example/x">link</a> <img src="https://b.example/i.png">'
        assert harvest_urls(html) == ["https://a.example/x", "https://b.example/i.png"]

    def test_harvest_from_markdown(self) -> None:
        md = "[label](https://a.example/x) and <https://b.example/y>"
        assert harvest_urls(md) == ["https://a.example/x", "https://b.example/y"]

    def test_harvest_dedupes_after_canonicalization(self) -> None:
        text = (
            "https://Example.com/a?utm_source=z https://example.com/a https://example.com/a/#frag"
        )
        assert harvest_urls(text) == ["https://example.com/a"]

    def test_harvest_skips_non_http(self) -> None:
        text = "mailto:bob@example.com ftp://x.example/a https://ok.example/p"
        assert harvest_urls(text) == ["https://ok.example/p"]

    def test_harvest_empty(self) -> None:
        assert harvest_urls("") == []
        assert harvest_urls("no links here") == []
