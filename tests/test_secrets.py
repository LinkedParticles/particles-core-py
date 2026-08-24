# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``particles.secrets`` — the single seam for secret env-var reads.

The contract is small but easy to break in a future refactor: each helper
either raises ``ValueError`` on missing values, returns ``None``, or returns
a default. Pinning the per-helper behaviour here means a regression to the
shape (e.g. silently returning ``""``) is caught immediately.
"""

from __future__ import annotations

import pytest

from particles.secrets import (
    get_anthropic_api_key,
    get_anthropic_api_key_optional,
    get_engine_token_optional,
    get_github_api_key_optional,
    get_local_llm_api_key_optional,
    get_numista_api_key,
    get_otel_exporter_headers_optional,
    get_particles_api_key,
    get_particles_config_path,
)


class TestAnthropicApiKey:
    def test_returns_value_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        assert get_anthropic_api_key() == "sk-ant-test"

    def test_raises_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            get_anthropic_api_key()

    def test_raises_when_empty_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Empty string is treated as unset — matches the prior call-site
        # behaviour in particles/llm.py (`if not api_key`).
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            get_anthropic_api_key()


class TestAnthropicApiKeyOptional:
    def test_returns_value_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        assert get_anthropic_api_key_optional() == "sk-ant-test"

    def test_returns_none_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert get_anthropic_api_key_optional() is None

    def test_returns_none_when_empty_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")
        assert get_anthropic_api_key_optional() is None


class TestGithubApiKeyOptional:
    def test_returns_value_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GITHUB_API_KEY", "ghp_secret")
        assert get_github_api_key_optional() == "ghp_secret"

    def test_returns_none_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GITHUB_API_KEY", raising=False)
        assert get_github_api_key_optional() is None

    def test_returns_none_when_empty_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GITHUB_API_KEY", "")
        assert get_github_api_key_optional() is None


class TestLocalLlmApiKeyOptional:
    def test_returns_value_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PARTICLES_LOCAL_LLM_API_KEY", "secret-token")
        assert get_local_llm_api_key_optional() == "secret-token"

    def test_returns_none_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PARTICLES_LOCAL_LLM_API_KEY", raising=False)
        assert get_local_llm_api_key_optional() is None

    def test_returns_none_when_empty_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PARTICLES_LOCAL_LLM_API_KEY", "")
        assert get_local_llm_api_key_optional() is None


class TestNumistaApiKey:
    def test_returns_value_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NUMISTA_API_KEY", "numista-key")
        assert get_numista_api_key() == "numista-key"

    def test_raises_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NUMISTA_API_KEY", raising=False)
        with pytest.raises(ValueError, match="NUMISTA_API_KEY"):
            get_numista_api_key()

    def test_raises_when_empty_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NUMISTA_API_KEY", "")
        with pytest.raises(ValueError, match="NUMISTA_API_KEY"):
            get_numista_api_key()


class TestParticlesApiKey:
    def test_returns_value_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PARTICLES_API_KEY", "prod-secret")
        assert get_particles_api_key() == "prod-secret"

    def test_returns_dev_key_default_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PARTICLES_API_KEY", raising=False)
        # Default is "dev-key" — the FastAPI auth dependency treats it as
        # a signal to skip authentication entirely.
        assert get_particles_api_key() == "dev-key"

    def test_returns_empty_string_when_explicitly_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Matches the prior call-site behaviour: os.environ.get("…", _DEV_KEY)
        # only falls back when the var is *missing*. An explicit empty value
        # passes through and is *not* equal to "dev-key", which means auth
        # is enforced with an empty bearer — i.e. all requests are denied.
        monkeypatch.setenv("PARTICLES_API_KEY", "")
        assert get_particles_api_key() == ""


class TestEngineToken:
    """the thin-client bearer token, distinct from PARTICLES_API_KEY."""

    def test_returns_value_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PARTICLES_ENGINE_TOKEN", "engine-secret")
        assert get_engine_token_optional() == "engine-secret"

    def test_returns_none_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PARTICLES_ENGINE_TOKEN", raising=False)
        assert get_engine_token_optional() is None

    def test_returns_none_when_empty_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PARTICLES_ENGINE_TOKEN", "")
        assert get_engine_token_optional() is None

    def test_independent_of_particles_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The client token and the server's expected key are separate roles.
        monkeypatch.setenv("PARTICLES_API_KEY", "server-key")
        monkeypatch.delenv("PARTICLES_ENGINE_TOKEN", raising=False)
        assert get_engine_token_optional() is None
        assert get_particles_api_key() == "server-key"


class TestOtelExporterHeaders:
    """the OTLP exporter credential (a secret, never in config.yaml)."""

    def test_returns_value_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PARTICLES_OTEL_EXPORTER_HEADERS", "authorization=Bearer tok")
        assert get_otel_exporter_headers_optional() == "authorization=Bearer tok"

    def test_returns_none_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PARTICLES_OTEL_EXPORTER_HEADERS", raising=False)
        assert get_otel_exporter_headers_optional() is None

    def test_returns_none_when_empty_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PARTICLES_OTEL_EXPORTER_HEADERS", "")
        assert get_otel_exporter_headers_optional() is None


class TestParticlesConfigPath:
    def test_returns_value_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PARTICLES_CONFIG", "/etc/particles/config.yaml")
        assert get_particles_config_path() == "/etc/particles/config.yaml"

    def test_returns_none_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PARTICLES_CONFIG", raising=False)
        assert get_particles_config_path() is None

    def test_returns_none_when_empty_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PARTICLES_CONFIG", "")
        assert get_particles_config_path() is None


class TestLLMApiKeyOptional:
    """the PARTICLES_LLM_API_KEY_<NAME> naming convention."""

    def test_reads_the_named_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from particles.secrets import get_llm_api_key_optional

        monkeypatch.setenv("PARTICLES_LLM_API_KEY_OPENAI", "sk-luna")
        assert get_llm_api_key_optional("openai") == "sk-luna"

    def test_non_alphanumerics_map_to_underscore(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from particles.secrets import get_llm_api_key_optional

        monkeypatch.setenv("PARTICLES_LLM_API_KEY_KIMI_K2", "sk-kimi")
        assert get_llm_api_key_optional("kimi-k2") == "sk-kimi"

    def test_local_falls_back_to_legacy_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from particles.secrets import get_llm_api_key_optional

        monkeypatch.delenv("PARTICLES_LLM_API_KEY_LOCAL", raising=False)
        monkeypatch.setenv("PARTICLES_LOCAL_LLM_API_KEY", "legacy-token")
        assert get_llm_api_key_optional("local") == "legacy-token"
        # The new-style var wins over the legacy one when both are set.
        monkeypatch.setenv("PARTICLES_LLM_API_KEY_LOCAL", "new-token")
        assert get_llm_api_key_optional("local") == "new-token"

    def test_returns_none_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from particles.secrets import get_llm_api_key_optional

        monkeypatch.delenv("PARTICLES_LLM_API_KEY_DEEPSEEK", raising=False)
        assert get_llm_api_key_optional("deepseek") is None
