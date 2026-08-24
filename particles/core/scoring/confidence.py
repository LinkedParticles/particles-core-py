# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Two-quantity confidence separation (§6.3).

The separation is **normative** — §6.3 of the technical specification says an
implementation that conflates the two quantities produces inconsistent
behaviour across extraction, query, and trust evaluation. The exact formulas,
the constants they consume, and worked test vectors an implementation
reproduces bit-for-bit are pinned in the Conformance Profile §4.

confidence.value   — stored immutably at creation, as calibrated at creation
                     time (raw EXTRACTOR_DIRECT value, or temperature-scaled
                     with calibration_source = CALIBRATED_BENCHMARK when the
                     extractor carried an active calibration; or a
                     raw agent self-report, calibration_source = AGENT_ASSERTED,
                     via the MCP write surface)
effective_conf     — derived at query time; never stored
"""

from __future__ import annotations

from enum import StrEnum

from particles.config import get_config


class CalibrationSource(StrEnum):
    EXTRACTOR_DIRECT = "EXTRACTOR_DIRECT"  # raw model output; lowest trust
    AGENT_ASSERTED = "AGENT_ASSERTED"  # agent self-report (MCP write); uncalibrated
    CALIBRATED_BENCHMARK = "CALIBRATED_BENCHMARK"  # post-hoc calibrated; medium trust
    HUMAN_REVIEW = "HUMAN_REVIEW"  # assigned by human reviewer; highest trust
    DERIVED = "DERIVED"  # machine-derived from premise particles; min-of-premises
    # Migrated in from another memory store (§6.3, the IMPORTED row).
    # The number is *ours* — a flat configured import floor — never the
    # incumbent's own score, which is preserved as a tag and structurally kept
    # out of the ranking arithmetic. §6.3 states that half normatively: an
    # implementation MUST NOT map a source system's own score onto
    # ``confidence.value``, because that field is stored immutably and
    # multiplies into effective_confidence. Distinct from AGENT_ASSERTED on
    # purpose: no agent asserted a migrated record, and merging the two
    # populations would destroy the "what I brought with me" vs "what I have
    # learned since" distinction, and would deny operators a separate
    # read-side cap on the imported population.
    IMPORTED = "IMPORTED"


def compute_effective_confidence(
    value: float,
    extractor_trust_weight: float = 1.0,
    source_trust_rank: float = 1.0,
    recency_factor: float = 1.0,
    calibration_source: CalibrationSource | None = None,
) -> float:
    """Compute effective_confidence at query time.  Never written back.  §6.3.

    effective_confidence =
        capped_value × extractor_trust_weight × source_trust_rank × recency_factor

    ``value`` is the stored ``confidence.value`` — already calibrated at
    creation time when the extractor carried an active calibration.

    When the read-side **uncalibrated cap** is enabled in
    ``config.confidence.uncalibrated_cap`` and ``calibration_source`` is listed
    in the configured ``sources``, ``value`` is first clamped down to
    ``cap_value`` via ``min`` before the multiply (``capped_value =
    min(value, cap_value)``). The cap is purely **read-side**: it never mutates
    the stored, immutable ``confidence.value`` (§6.3). It is default-OFF,
    and ``calibration_source=None`` (the default) never triggers it, so existing
    callers are unaffected. The cap composes with the trust-weight cap:
    a particle may have both its value clamped here and its extractor trust
    weight clamped upstream.
    """
    capped_value = value
    if calibration_source is not None:
        cap = get_config().confidence.uncalibrated_cap
        if cap.enabled and calibration_source in cap.sources:
            capped_value = min(capped_value, cap.cap_value)
    return min(
        1.0,
        max(
            0.0,
            capped_value * extractor_trust_weight * source_trust_rank * recency_factor,
        ),
    )


def merge_co_evidential_confidence(
    entries: list[tuple[float, str]],
) -> float:
    """Trust-weighted noisy-OR merge over a co-evidential group (§6.9).

    Adding corroborating evidence should *increase* certainty, even if the
    individual evidence is weak. A simple max or average would either ignore
    weak corroboration or let weak particles drag down strong ones. Noisy-OR
    treats each particle as an independent positive signal and combines them
    multiplicatively:

        merged = 1 - product over p in G of (1 - eff_conf(p) × source_independence(p))

    To prevent a single chatty source from saturating the merge — a spammer
    spinning up 1000 subdomains parroting the same claim should not produce
    99.99% confidence — particles from the *same source* are discounted by
    1/k, where k is the particle's rank within its source in the group
    (first from a source = 1.0; second = 1/2; third = 1/3; etc.). The first
    particle from each distinct source contributes its full effective
    confidence.

    Within each source, particles are ranked by **descending effective
    confidence**: the strongest claim from a source carries full weight and
    weaker repeats absorb the 1/k discount. This is the operator-favorable
    reading of the throttle (the discount exists to stop repetition from
    *saturating* the merge, not to penalize the source's best evidence), and
    it makes the result independent of input order — callers may pass entries
    in any order (e.g. iterating an unordered group) and get the same value.

    Args:
        entries: List of (effective_confidence, source_key) tuples. ``source_key``
                 is whatever identifier groups particles by source — typically the
                 corpus_entry_id of the first SOURCE provenance ref, but may be
                 the source domain or author for finer-grained throttling.
                 Order does not matter.

    Returns:
        Merged confidence in [0, 1]. An empty group returns 0.0. A single-entry
        group returns its effective confidence unchanged.
    """
    if not entries:
        return 0.0
    if len(entries) == 1:
        return min(1.0, max(0.0, entries[0][0]))

    # Deterministic merge order: descending confidence, source_key tiebreak.
    # Only the within-source order affects the result (each source has its own
    # k counter), but a total order makes the merge reproducible bit-for-bit.
    ordered = sorted(entries, key=lambda e: (-e[0], e[1]))

    # Per-source occurrence count → 1/k discount
    seen: dict[str, int] = {}
    product = 1.0
    for eff_conf, source_key in ordered:
        seen[source_key] = seen.get(source_key, 0) + 1
        k = seen[source_key]
        weight = 1.0 / k
        bounded = min(1.0, max(0.0, eff_conf)) * weight
        product *= 1.0 - bounded

    return min(1.0, max(0.0, 1.0 - product))


def derive_abstraction_confidence(premise_values: list[float]) -> float:
    """Stored ``confidence.value`` for a derived particle (§6.3).

    An abstraction is entailed by the *conjunction* of its premises, so its
    credence must not exceed its weakest premise: the rule is **min over the
    premises' stored values**, clamped to [0, 1]. Anything cleverer
    (corroboration-aware combining across premises) is deferred work and is
    deliberately not implemented here. The result is stored once and
    immutably, like every other ``confidence.value`` (§6.3); everything reactive
    — the stale-support discount — lives in effective confidence
    at read time.

    Args:
        premise_values: The premises' stored ``confidence.value``s. Must be
            non-empty — a derived particle without premises is a contradiction
            in terms.

    Returns:
        The minimum premise value, clamped to [0, 1].

    Raises:
        ValueError: If ``premise_values`` is empty.
    """
    if not premise_values:
        raise ValueError("derived particle requires at least one premise")
    return min(1.0, max(0.0, min(premise_values)))
