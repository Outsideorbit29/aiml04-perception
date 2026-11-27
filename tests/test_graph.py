import numpy as np
import pandas as pd

from fraud.graph import best_blend, blend_logits, encode_entities, find_rings


def test_encode_entities_skips_nan():
    df = pd.DataFrame({"client_key": ["a", "b", "a"], "device_key": ["d1", None, "d1"]})
    enc = encode_entities(df, ["client_key", "device_key"])
    tx, ent, n = enc["client_key"]
    assert n == 2 and tx.tolist() == [0, 1, 2] and ent.tolist() == [0, 1, 0]
    tx, ent, n = enc["device_key"]
    assert n == 1 and tx.tolist() == [0, 2] and ent.tolist() == [0, 0]


def test_find_rings_links_through_specific_devices_only():
    clients = pd.Series(["c1", "c2", "c2", "c3", "c4", "c5", "x1", "x2", "x3", "x4"])
    devices = pd.Series(["dA", "dA", "dB", "dB", "dB", None, "gen", "gen", "gen", "gen"])
    # c1-c2 via dA, c2-c3-c4 via dB -> ring of 4; "gen" has 4 clients > max 3 -> ignored
    rings = find_rings(clients, devices, max_clients_per_device=3, min_size=3)
    assert set(rings["client_key"]) == {"c1", "c2", "c3", "c4"}
    assert rings["ring_id"].nunique() == 1
    assert (rings["ring_size"] == 4).all() and (rings["ring_devices"] == 2).all()


def test_find_rings_min_size_and_empty():
    clients = pd.Series(["a", "b"])
    devices = pd.Series(["d", "d"])
    assert find_rings(clients, devices, 5, 3).empty
    assert len(find_rings(clients, devices, 5, 2)) == 2


def test_blend_endpoints_and_choice():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.1).astype(int)
    good = np.clip(0.2 + 0.6 * y + rng.normal(0, 0.15, 2000), 0.01, 0.99)
    noise = rng.random(2000)
    assert np.allclose(blend_logits(good, noise, 1.0), good, atol=1e-6)
    assert np.allclose(blend_logits(good, noise, 0.0), noise, atol=1e-6)
    w, scores = best_blend(y, good, noise, [0.0, 0.5, 1.0])
    assert w >= 0.5 and scores[1.0] > scores[0.0]
