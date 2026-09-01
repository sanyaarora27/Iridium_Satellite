"""
Basic unit tests for the fusion layer.

Run from the repository root with:
    pytest -q
"""
from pathlib import Path
import importlib.util
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FUSION_DIR = PROJECT_ROOT / "fusion"

sys.path.insert(0, str(FUSION_DIR))

from _shared import (  # noqa: E402
    build_simulated_payload,
    compute_mac,
    derive_hmac_key,
    generate_nonce,
    verify_freshness,
    verify_mac,
)


def load_numeric_module(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(
        module_name,
        FUSION_DIR / filename,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


fusion_eval = load_numeric_module("04_fusion_eval.py", "fusion_eval")


def test_valid_hmac_verifies():
    key = derive_hmac_key(92)
    payload = build_simulated_payload(1, 92)
    nonce = "123456"

    mac = compute_mac(key, 1, 92, nonce, payload)

    assert verify_mac(key, 1, 92, nonce, payload, mac)


def test_wrong_key_fails_hmac():
    key_92 = derive_hmac_key(92)
    key_85 = derive_hmac_key(85)

    payload = build_simulated_payload(1, 92)
    nonce = "123456"

    mac = compute_mac(key_85, 1, 92, nonce, payload)

    assert not verify_mac(key_92, 1, 92, nonce, payload, mac)


def test_tampered_payload_fails_hmac():
    key = derive_hmac_key(92)
    payload = build_simulated_payload(1, 92)
    nonce = "123456"

    mac = compute_mac(key, 1, 92, nonce, payload)

    assert not verify_mac(
        key,
        1,
        92,
        nonce,
        payload + "|tampered",
        mac,
    )


def test_replay_fails_on_second_use():
    sat = 92
    t = 1_700_000_000
    nonce = generate_nonce(sat, t)

    seen = set()

    assert verify_freshness(sat, nonce, t, seen)
    assert not verify_freshness(sat, nonce, t, seen)


def test_expired_nonce_fails():
    sat = 92
    t = 1_700_000_000
    nonce = generate_nonce(sat, t)

    seen = set()

    assert not verify_freshness(
        sat,
        nonce,
        t + 120,
        seen,
    )


def test_stolen_target_key_can_verify():
    target_sat = 92
    attacker_true_sat = 51
    message_id = 7
    t = 1_700_000_000

    target_key = derive_hmac_key(target_sat)
    payload = build_simulated_payload(message_id, attacker_true_sat)
    nonce = generate_nonce(target_sat, t)

    mac = compute_mac(
        target_key,
        message_id,
        target_sat,
        nonce,
        payload,
    )

    assert verify_mac(
        target_key,
        message_id,
        target_sat,
        nonce,
        payload,
        mac,
    )


def test_fusion_hmac_failure_rejects():
    assert fusion_eval.decide(
        rf_match=True,
        hmac_pass=False,
        freshness_pass=True,
    ) == "reject"


def test_fusion_freshness_failure_rejects():
    assert fusion_eval.decide(
        rf_match=True,
        hmac_pass=True,
        freshness_pass=False,
    ) == "reject"


def test_fusion_rf_match_accepts():
    assert fusion_eval.decide(
        rf_match=True,
        hmac_pass=True,
        freshness_pass=True,
    ) == "accept"


def test_fusion_rf_mismatch_flags():
    assert fusion_eval.decide(
        rf_match=False,
        hmac_pass=True,
        freshness_pass=True,
    ) == "flag"
