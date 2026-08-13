"""
fusion/_shared.py

Small shared helpers for the higher-layer simulation (Option B: HMAC,
Option C: TOTP freshness). Kept in one place so 02_higher_layer_sim.py and
03_fusion_eval.py use *identical* key derivation and validation logic --
important because 03 needs to construct attack variants (a "stolen key" that
legitimately validates, a replayed nonce) using the same primitives that 02
uses to build the genuine baseline.

Nothing here touches real RF data. Every value in this module is part of
the SIMULATED higher layer -- Iridium has no message authentication, so
none of this exists in the real signal. That boundary should be stated
plainly in the write-up.
"""
from __future__ import annotations

import hashlib
import hmac
import base64
import pyotp

# --------------------------------------------------------------------------
# Config -- adjust to match your real fusion/evidence.csv column names.
# --------------------------------------------------------------------------
COLUMN_MAP = {
    "id": "message_id",
    "true_sat": "true_satellite",
    "claimed_sat": "claimed_satellite",
    "predicted_sat": "rf_predicted_sat",
    "confidence": "rf_confidence",
}

MASTER_SEED = "iridium-fusion-demo-v1"   # fixed -> reproducible across runs
SATELLITES = [92, 85, 87, 51, 109]       # the five-satellite subset
BASE_TIME = 1_700_000_000                # arbitrary but realistic unix time;
                                          # keeps TOTP timecodes well away from
                                          # zero (pyotp needs counter > 0)

# --------------------------------------------------------------------------
# Per-satellite key material (deterministic, so re-running reproduces
# identical results -- important for a dissertation artefact).
# --------------------------------------------------------------------------

def derive_hmac_key(sat_id) -> bytes:
    """Deterministic 32-byte HMAC key for a given satellite's ground identity."""
    return hashlib.sha256(f"{MASTER_SEED}:hmac:{sat_id}".encode()).digest()


def derive_totp_secret(sat_id) -> str:
    """Deterministic base32 TOTP secret for a given satellite's ground identity."""
    raw = hashlib.sha256(f"{MASTER_SEED}:totp:{sat_id}".encode()).digest()[:10]
    return base64.b32encode(raw).decode()


def get_totp(sat_id) -> pyotp.TOTP:
    return pyotp.TOTP(derive_totp_secret(sat_id), interval=30)


# --------------------------------------------------------------------------
# HMAC (Option B) -- RFC 2104 / FIPS 198-1, via stdlib hmac. Not hand-rolled.
# --------------------------------------------------------------------------

def canonical_message(global_index, claimed_sat, nonce) -> bytes:
    return f"{global_index}|{claimed_sat}|{nonce}".encode()


def compute_mac(key: bytes, global_index, claimed_sat, nonce) -> str:
    msg = canonical_message(global_index, claimed_sat, nonce)
    return hmac.new(key, msg, hashlib.sha256).hexdigest()


def verify_mac(key: bytes, global_index, claimed_sat, nonce, mac: str) -> bool:
    expected = compute_mac(key, global_index, claimed_sat, nonce)
    return hmac.compare_digest(expected, mac)


# --------------------------------------------------------------------------
# Freshness (Option C) -- RFC 6238 TOTP, via pyotp. A code is only fresh if
# it (a) validates for the claimed satellite's TOTP secret at the given time
# AND (b) has not been consumed before -- this second check is what catches
# replay; TOTP validity alone does not, since a captured code is still
# time-valid within its window.
# --------------------------------------------------------------------------

def generate_nonce(sat_id, at_time: int) -> str:
    return get_totp(sat_id).at(at_time)


def verify_freshness(sat_id, nonce: str, at_time: int, seen_nonces: set) -> bool:
    totp_valid = get_totp(sat_id).verify(nonce, for_time=at_time, valid_window=1)
    key = (sat_id, nonce)
    replayed = key in seen_nonces
    if totp_valid and not replayed:
        seen_nonces.add(key)
        return True
    return False


# --------------------------------------------------------------------------
# Confidence bands -- FINALISED.
#
# The obvious approach (derive edges from the authentication-metrics script's
# FRR/FAR crossover) turned out not to apply: that curve thresholds
# P(claimed satellite), a materially different quantity from rf_confidence
# (P(top predicted class)). For the ~75% of messages the classifier gets
# wrong, P(claimed) sits well below rf_confidence, dragging that curve's
# crossover down to tau=0.19 -- below evidence.csv's entire observed
# confidence range (min 0.21). Reusing it directly would compare two
# different scores.
#
# Built the matched version instead, directly from evidence.csv: does
# rf_confidence actually distinguish correct RF predictions from incorrect
# ones? AUC = 0.506 (p = 0.77, Mann-Whitney) -- statistically indistinguishable
# from chance. The classifier's self-reported confidence carries no real
# information about whether it's right. There is no data-derived threshold
# to find here; that null result is itself worth reporting in the write-up.
#
# Given that, these edges are an explicit DESIGN CHOICE, not an optimised
# threshold: tertiles of the observed confidence distribution, chosen for a
# roughly even split (455 / 360 / 418 messages) rather than for any
# discriminative power the data doesn't have.
# --------------------------------------------------------------------------
CONF_BAND_LOW_MAX = 0.2667
CONF_BAND_HIGH_MIN = 0.3067


def confidence_band(confidence: float, low_max: float, high_min: float) -> str:
    if confidence >= high_min:
        return "high"
    if confidence <= low_max:
        return "low"
    return "middle"
