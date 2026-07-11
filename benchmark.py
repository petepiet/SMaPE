#!/usr/bin/env python3
"""Hand-assignment benchmark harness for SMaPE.

Measures how accurately SMaPE assigns notes to the left/right hand (and,
secondarily, fingers) against a human-corrected ground truth.

Workflow (see benchmark/README.md for the full authoring guide):

    1. Run SMaPE on a video -> fingering.json
    2. python benchmark.py init fingering.json benchmark/<song>/truth.json
    3. Hand-correct the wrong "hand" values in truth.json
    4. Copy fingering.json to benchmark/<song>/predicted.json
    5. python benchmark.py            # scans ./benchmark, prints a table

Pure scoring logic (`join_notes`, `score`) has no I/O or third-party
dependencies and is unit-tested in selftest.py with plain dicts/lists. I/O
(`load_truth`, `load_pred`, `init_truth`, `run_dir`) and the CLI live in
this same module but are exercised manually / via the CLI, not selftest.py.
"""

from __future__ import annotations

import json
import os
import sys


# ---------------------------------------------------------------------------
# Pure logic — unit-tested in selftest.py
# ---------------------------------------------------------------------------

def join_notes(truth, pred, sec_tol=0.1):
    """Match each truth note to the nearest unused same-pitch pred note.

    truth and pred are lists of note dicts with at least "pitch" and
    "onsetSec". Greedy, truth-note-order matching: for each truth note (in
    the order given), pick the closest-onset pred note of the same pitch
    that is within sec_tol seconds and not already used. Returns a list of
    (truth_note, pred_note_or_None) tuples, one per truth note.
    """
    used = [False] * len(pred)
    # Group pred indices by pitch for faster lookup.
    by_pitch = {}
    for i, p in enumerate(pred):
        by_pitch.setdefault(p["pitch"], []).append(i)

    pairs = []
    for t in truth:
        candidates = by_pitch.get(t["pitch"], [])
        best_idx = None
        best_dist = None
        for i in candidates:
            if used[i]:
                continue
            dist = abs(pred[i]["onsetSec"] - t["onsetSec"])
            if dist > sec_tol:
                continue
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx is None:
            pairs.append((t, None))
        else:
            used[best_idx] = True
            pairs.append((t, pred[best_idx]))
    return pairs


def score(pairs):
    """Compute coverage / hand-accuracy / finger-accuracy metrics from pairs."""
    labeled = len(pairs)
    matched_pairs = [(t, p) for (t, p) in pairs if p is not None]
    matched = len(matched_pairs)
    coverage = (matched / labeled) if labeled else 0.0

    hand_correct = sum(1 for (t, p) in matched_pairs if t["hand"] == p["hand"])
    hand_accuracy = (hand_correct / matched) if matched else 0.0

    finger_pairs = [(t, p) for (t, p) in matched_pairs if t.get("finger") is not None]
    finger_labeled = len(finger_pairs)
    finger_correct = sum(1 for (t, p) in finger_pairs if t["finger"] == p.get("finger"))
    finger_accuracy = (finger_correct / finger_labeled) if finger_labeled else 0.0

    l_as_r = sum(1 for (t, p) in matched_pairs if t["hand"] == "L" and p["hand"] == "R")
    r_as_l = sum(1 for (t, p) in matched_pairs if t["hand"] == "R" and p["hand"] == "L")

    return {
        "labeled": labeled,
        "matched": matched,
        "coverage": coverage,
        "hand_correct": hand_correct,
        "hand_accuracy": hand_accuracy,
        "finger_labeled": finger_labeled,
        "finger_correct": finger_correct,
        "finger_accuracy": finger_accuracy,
        "confusion": {"L_as_R": l_as_r, "R_as_L": r_as_l},
    }


def trim_notes(notes, max_sec):
    """Keep only notes whose onset is within the first ``max_sec`` seconds.

    For labeling a contiguous window (e.g. correct the first ~90s in
    Symplethesia, export, then trim the truth to that window so the metric
    scores only what was actually reviewed). Pure. Notes lacking "onsetSec"
    are dropped (nothing to place them in time)."""
    return [n for n in notes if n.get("onsetSec") is not None and n["onsetSec"] <= max_sec]


# ---------------------------------------------------------------------------
# I/O + orchestration — not unit-tested (needs real files)
# ---------------------------------------------------------------------------

def load_truth(path):
    """Read truth.json, return its "notes" list."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["notes"]


def load_pred(path):
    """Read a SMaPE fingering.json, return normalized {"pitch", "onsetSec",
    "hand", "finger"} dicts. Notes lacking startSec or hand are skipped.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for n in data.get("notes", []):
        if "startSec" not in n or "hand" not in n:
            continue
        out.append({
            "pitch": n["pitch"],
            "onsetSec": n["startSec"],
            "hand": n["hand"],
            "finger": n.get("finger"),
        })
    return out


def init_truth(fingering_path, out_path):
    """Read a fingering.json and write a truth.json TEMPLATE pre-filled with
    SMaPE's current guesses, sorted by onsetSec, for the human to correct.
    """
    with open(fingering_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    notes = []
    for n in data.get("notes", []):
        if "startSec" not in n or "hand" not in n:
            continue
        notes.append({
            "pitch": n["pitch"],
            "onsetSec": n["startSec"],
            "hand": n["hand"],
            "finger": n.get("finger"),
        })
    notes.sort(key=lambda n: n["onsetSec"])

    template = {"video": data.get("source", ""), "notes": notes}

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2)

    print(f"Wrote {out_path} ({len(notes)} notes)")


def _fmt_pct(x):
    return f"{x * 100:.1f}%"


def run_dir(bench_dir):
    """Scan immediate subdirectories of bench_dir; score every one that has
    both truth.json and predicted.json. Prints a per-song table plus a
    micro-averaged aggregate row. Returns 0 if at least one song scored,
    1 otherwise.
    """
    if not os.path.isdir(bench_dir):
        print(f"No such directory: {bench_dir}")
        print(f"Use `python benchmark.py init <fingering.json>` to scaffold a case.")
        return 1

    rows = []
    for name in sorted(os.listdir(bench_dir)):
        song_dir = os.path.join(bench_dir, name)
        if not os.path.isdir(song_dir):
            continue
        truth_path = os.path.join(song_dir, "truth.json")
        pred_path = os.path.join(song_dir, "predicted.json")
        if not (os.path.isfile(truth_path) and os.path.isfile(pred_path)):
            continue

        truth = load_truth(truth_path)
        pred = load_pred(pred_path)
        pairs = join_notes(truth, pred)
        result = score(pairs)
        rows.append((name, result))

    if not rows:
        print(f"No scoreable songs found under {bench_dir!r}.")
        print("Each song needs a truth.json AND predicted.json subdirectory entry.")
        print("Use `python benchmark.py init <fingering.json>` to scaffold one.")
        return 1

    header = f"{'song':<30}{'labeled':>9}{'coverage':>11}{'hand acc':>11}{'finger acc':>12}"
    print(header)
    print("-" * len(header))

    total_labeled = 0
    total_matched = 0
    total_hand_correct = 0
    total_finger_labeled = 0
    total_finger_correct = 0

    for name, r in rows:
        print(
            f"{name:<30}{r['labeled']:>9}{_fmt_pct(r['coverage']):>11}"
            f"{_fmt_pct(r['hand_accuracy']):>11}{_fmt_pct(r['finger_accuracy']):>12}"
        )
        total_labeled += r["labeled"]
        total_matched += r["matched"]
        total_hand_correct += r["hand_correct"]
        total_finger_labeled += r["finger_labeled"]
        total_finger_correct += r["finger_correct"]

    agg_coverage = (total_matched / total_labeled) if total_labeled else 0.0
    agg_hand_acc = (total_hand_correct / total_matched) if total_matched else 0.0
    agg_finger_acc = (total_finger_correct / total_finger_labeled) if total_finger_labeled else 0.0

    print("-" * len(header))
    print(
        f"{'AGGREGATE':<30}{total_labeled:>9}{_fmt_pct(agg_coverage):>11}"
        f"{_fmt_pct(agg_hand_acc):>11}{_fmt_pct(agg_finger_acc):>12}"
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _default_init_out_path(fingering_path):
    base = os.path.basename(fingering_path)
    for suffix in (".fingering.json", ".json"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    out_name = f"{base}.truth.json"
    return os.path.join(os.path.dirname(fingering_path), out_name)


def main(argv):
    if argv and argv[0] == "init":
        rest = argv[1:]
        if not rest:
            print("usage: python benchmark.py init <fingering.json> [out.json]")
            return 1
        fingering_path = rest[0]
        out_path = rest[1] if len(rest) > 1 else _default_init_out_path(fingering_path)
        init_truth(fingering_path, out_path)
        return 0

    if argv and argv[0] == "trim":
        rest = argv[1:]
        if len(rest) < 2:
            print("usage: python benchmark.py trim <truth.json> <max_sec> [out.json]")
            return 1
        truth_path, max_sec = rest[0], float(rest[1])
        out_path = rest[2] if len(rest) > 2 else truth_path
        with open(truth_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        before = len(data.get("notes", []))
        data["notes"] = trim_notes(data.get("notes", []), max_sec)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"trimmed {before} -> {len(data['notes'])} notes (<= {max_sec}s), wrote {out_path}")
        return 0

    bench_dir = argv[0] if argv else "benchmark"
    return run_dir(bench_dir)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
