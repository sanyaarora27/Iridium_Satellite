"""
fusion/_shared.py

Shared helpers for the simulated higher layer used by the satellite
authentication prototype.

Implemented proof-of-concept mechanisms:
    Option B: HMAC-SHA256 message authentication
    Option C: TOTP-based freshness + explicit replay-state tracking

IMPORTANT
---------
This module does NOT implement the real Iridium authentication protocol.
All keys, payloads, nonces and higher-layer verdicts are simulated so that
real RF-fingerprinting evidence can be evaluated inside a layered security
prototype.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
from pathlib import Path
from typing import MutableSet

import pyotp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FUSION_DIR = PROJECT_ROOT / "fusion"

FUSION_OUTPUT_DIR = FUSION_DIR / "outputs"
FUSION_TABLES_DIR = FUSION_OUTPUT_DIR / "tables"
FUSION_REPORTS_DIR = FUSION_OUTPUT_DIR / "reports"
FUSION_FIGURES_DIR = FUSION_OUTPUT_DIR / "figures"

FUSION_TABLES_DIR.mkdir(parents=True, exist_ok=True)
FUSION_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FUSION_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

COLUMN_MAP = {
    "id": "message_id",
    "true_sat": "true_satellite",
    "claimed_sat": "claimed_satellite",
    "predicted_sat": "rf_predicted_sat",
    "confidence": "rf_confidence",
}

# SIMULATION ONLY.
# Deterministic derivation is used solely to make dissertation experiments
# reproducible. It is not a production key-management design.
SIM_MASTER_SEED = "iridium-fusion-demo-v2"

SATELLITES = [92, 85, 87, 51, 109]

# Deterministic simulated clock. A real deployment would use an authenticated
# protocol/capture time source.
SIM_BASE_TIME = 1_700_000_000
TOTP_INTERVAL_SECONDS = 30
TOTP_VALID_WINDOW = 1

def derive_hmac_key(sat_id: int) -> bytes:
    material = f"{SIM_MASTER_SEED}:hmac:{int(sat_id)}".encode("utf-8")
    return hashlib.sha256(material).digest()

def derive_totp_secret(sat_id: int) -> str:
    material = f"{SIM_MASTER_SEED}:totp:{int(sat_id)}".encode("utf-8")
    raw = hashlib.sha256(material).digest()[:20]
    return base64.b32encode(raw).decode("ascii")

def get_totp(sat_id: int) -> pyotp.TOTP:
    return pyotp.TOTP(
        derive_totp_secret(sat_id),
        interval=TOTP_INTERVAL_SECONDS,
    )

def build_simulated_payload(message_id: int, true_satellite: int) -> str:
    """
    Deterministic synthetic payload for integrity testing.

    This is not an Iridium protocol payload.
    """
    return f"telemetry:{int(message_id)}:source:{int(true_satellite)}"

def canonical_message(
    message_id: int,
    claimed_satellite: int,
    nonce: str,
    payload: str,
) -> bytes:
    return (
        f"{int(message_id)}|{int(claimed_satellite)}|{nonce}|{payload}"
    ).encode("utf-8")

def compute_mac(
    key: bytes,
    message_id: int,
    claimed_satellite: int,
    nonce: str,
    payload: str,
) -> str:
    msg = canonical_message(
        message_id=message_id,
        claimed_satellite=claimed_satellite,
        nonce=nonce,
        payload=payload,
    )
    return hmac.new(key, msg, hashlib.sha256).hexdigest()

def verify_mac(
    key: bytes,
    message_id: int,
    claimed_satellite: int,
    nonce: str,
    payload: str,
    mac: str,
) -> bool:
    expected = compute_mac(
        key=key,
        message_id=message_id,
        claimed_satellite=claimed_satellite,
        nonce=nonce,
        payload=payload,
    )
    return hmac.compare_digest(expected, str(mac))

def corrupt_mac(mac: str) -> str:
    mac = str(mac)
    if not mac:
        return "0"
    replacement = "0" if mac[-1].lower() != "0" else "1"
    return mac[:-1] + replacement

def generate_nonce(sat_id: int, at_time: int) -> str:
    return get_totp(sat_id).at(int(at_time))

def verify_freshness(
    sat_id: int,
    nonce: str,
    at_time: int,
    seen_nonces: MutableSet[tuple[int, str]],
) -> bool:
    """
    Freshness passes only if the TOTP is valid and the same
    (satellite, nonce) pair has not already been consumed.
    """
    sat_id = int(sat_id)
    nonce = str(nonce)

    valid = get_totp(sat_id).verify(
        nonce,
        for_time=int(at_time),
        valid_window=TOTP_VALID_WINDOW,
    )

    replay_key = (sat_id, nonce)
    replayed = replay_key in seen_nonces

    if valid and not replayed:
        seen_nonces.add(replay_key)
        return True

    return False

def higher_layer_decision(hmac_pass: bool, freshness_pass: bool) -> str:
    return "pass" if bool(hmac_pass) and bool(freshness_pass) else "reject"

def alternate_satellites(true_satellite: int) -> list[int]:
    """Return every allowed false claimed identity for a true satellite."""
    true_satellite = int(true_satellite)
    alternatives = [sat for sat in SATELLITES if sat != true_satellite]
    if not alternatives:
        raise ValueError(f"No alternate satellite available for {true_satellite}")
    return alternatives

def choose_alternate_satellite(true_satellite: int) -> int:
    """Return one deterministic alternate identity for single-case controls."""
    return alternate_satellites(true_satellite)[0]
