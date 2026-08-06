"""Provisional voice profiles: pending pool that promotes after 3 confirming matches."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

COSINE_THRESHOLD = 0.75
PROMOTION_DISTINCT_SESSIONS = 3
PROVISIONAL_OBSERVATION_SECONDS = 10.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cosine(a: list[float], b: list[float]) -> float:
    av = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(av)
    nb = np.linalg.norm(bv)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(av, bv) / (na * nb))


class ProvisionalProfiles:
    """Lives at voice_profiles/pending.json + voice_profiles/binding_log.jsonl"""

    def __init__(self, voice_profiles_dir: Path):
        self.dir = Path(voice_profiles_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.pending_path = self.dir / "pending.json"
        self.log_path = self.dir / "binding_log.jsonl"
        self.profiles_path = self.dir / "profiles.json"

    # ---------- internal storage ----------
    def _load_pending(self) -> dict:
        if not self.pending_path.exists():
            return {}
        try:
            return json.loads(self.pending_path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_pending(self, data: dict) -> None:
        tmp = self.pending_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self.pending_path)

    def _load_profiles(self) -> dict:
        if not self.profiles_path.exists():
            return {}
        try:
            return json.loads(self.profiles_path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_profiles(self, data: dict) -> None:
        tmp = self.profiles_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self.profiles_path)

    def _log(self, action: str, profile_name: Optional[str], session_id: Optional[str], pending_id: Optional[str]) -> None:
        row = {
            "ts_utc": _now_iso(),
            "action": action,
            "profile_name": profile_name,
            "session_id": session_id,
            "pending_id": pending_id,
        }
        with self.log_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    @staticmethod
    def _mean_embedding(observations: list[dict]) -> list[float]:
        arr = np.asarray([o["embedding"] for o in observations], dtype=np.float64)
        return arr.mean(axis=0).tolist()

    # ---------- public API ----------
    def add_unknown(
        self,
        embedding: list[float],
        suggested_name: str,
        session_id: str,
    ) -> str:
        """Add a candidate observation for an unknown speaker.

        If an existing pending record's mean embedding matches (cosine > 0.75),
        merge this observation into that record — caller doesn't need to know
        the pending_id, the system finds it. If the same session_id is already
        observed for that record, it's a no-op. If embedding doesn't match
        anything pending, create a new pending entry.
        """
        pending = self._load_pending()

        for pid, rec in pending.items():
            mean_emb = self._mean_embedding(rec["observations"])
            if _cosine(mean_emb, embedding) > COSINE_THRESHOLD:
                seen_sessions = {o["session_id"] for o in rec["observations"]}
                if session_id in seen_sessions:
                    return pid
                rec["observations"].append({
                    "embedding": list(embedding),
                    "session_id": session_id,
                    "ts_utc": _now_iso(),
                })
                distinct = {o["session_id"] for o in rec["observations"]}
                if len(distinct) >= PROMOTION_DISTINCT_SESSIONS:
                    self._save_pending(pending)
                    self._promote_and_clear(pid, session_id)
                else:
                    self._save_pending(pending)
                self._log("added", rec["suggested_name"], session_id, pid)
                return pid

        pending_id = uuid.uuid4().hex
        pending[pending_id] = {
            "pending_id": pending_id,
            "suggested_name": suggested_name,
            "created_at": _now_iso(),
            "observations": [
                {
                    "embedding": list(embedding),
                    "session_id": session_id,
                    "ts_utc": _now_iso(),
                }
            ],
        }
        self._save_pending(pending)
        self._log("added", suggested_name, session_id, pending_id)
        return pending_id

    def observe(
        self,
        pending_id: str,
        embedding: list[float],
        session_id: str,
    ) -> dict:
        """Record another match; promote on 3rd DISTINCT session_id."""
        pending = self._load_pending()
        if pending_id not in pending:
            raise KeyError(f"unknown pending_id: {pending_id}")

        rec = pending[pending_id]
        mean_emb = self._mean_embedding(rec["observations"])
        sim = _cosine(mean_emb, embedding)
        if sim <= COSINE_THRESHOLD:
            return {"promoted": False, "matches": len({o["session_id"] for o in rec["observations"]}), "similarity": sim, "reason": "below_threshold"}

        # Append observation (idempotent: if same session_id already present, don't double-count)
        existing_sessions = {o["session_id"] for o in rec["observations"]}
        if session_id not in existing_sessions:
            rec["observations"].append({
                "embedding": list(embedding),
                "session_id": session_id,
                "ts_utc": _now_iso(),
            })

        distinct_sessions = {o["session_id"] for o in rec["observations"]}
        match_count = len(distinct_sessions)

        if match_count >= PROMOTION_DISTINCT_SESSIONS:
            pending[pending_id] = rec
            self._save_pending(pending)
            result = self._promote_and_clear(pending_id, session_id)
            return result

        pending[pending_id] = rec
        self._save_pending(pending)
        return {"promoted": False, "matches": match_count}

    def _promote_and_clear(self, pending_id: str, trigger_session_id: str) -> dict:
        """Promote a pending entry into profiles.json, then delete the pending row.
        Reusable from both add_unknown and observe."""
        pending = self._load_pending()
        if pending_id not in pending:
            return {"promoted": False, "matches": 0, "reason": "vanished"}
        rec = pending[pending_id]
        profile_name = rec["suggested_name"]
        new_mean = self._mean_embedding(rec["observations"])
        obs_count = len(rec["observations"])
        new_total_seconds = obs_count * PROVISIONAL_OBSERVATION_SECONDS

        profiles = self._load_profiles()
        new_sources = [f"provisional:{o['session_id']}" for o in rec["observations"]]
        if profile_name in profiles:
            existing = profiles[profile_name]
            ex_emb = np.asarray(existing["embedding"], dtype=np.float64)
            ex_secs = float(existing.get("total_seconds", 0.0))
            nv_emb = np.asarray(new_mean, dtype=np.float64)
            total_secs = ex_secs + new_total_seconds
            weighted = (ex_emb * ex_secs + nv_emb * new_total_seconds) / total_secs if total_secs > 0 else nv_emb
            profiles[profile_name] = {
                "embedding": weighted.tolist(),
                "total_seconds": total_secs,
                "source_files": existing.get("source_files", []) + new_sources,
            }
        else:
            profiles[profile_name] = {
                "embedding": new_mean,
                "total_seconds": new_total_seconds,
                "source_files": new_sources,
            }
        self._save_profiles(profiles)

        del pending[pending_id]
        self._save_pending(pending)
        self._log("promoted", profile_name, trigger_session_id, pending_id)
        return {"promoted": True, "profile_name": profile_name, "matches": len({o["session_id"] for o in rec["observations"]})}

    def rollback(self, profile_name: str) -> int:
        """Remove profile + any pending bound to its name. Returns # entries removed."""
        removed = 0
        profiles = self._load_profiles()
        if profile_name in profiles:
            del profiles[profile_name]
            self._save_profiles(profiles)
            removed += 1

        pending = self._load_pending()
        to_drop = [pid for pid, rec in pending.items() if rec.get("suggested_name") == profile_name]
        for pid in to_drop:
            del pending[pid]
            removed += 1
        if to_drop:
            self._save_pending(pending)

        self._log("rolled_back", profile_name, None, None)
        return removed

    def prune(self, max_age_days: int = 30, min_matches: int = 2) -> int:
        """Drop pending entries with <min_matches whose oldest obs is older than max_age_days."""
        pending = self._load_pending()
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        dropped = 0
        for pid in list(pending.keys()):
            rec = pending[pid]
            distinct = len({o["session_id"] for o in rec["observations"]})
            if distinct >= min_matches:
                continue
            try:
                oldest_ts = min(
                    datetime.fromisoformat(o["ts_utc"]) for o in rec["observations"]
                )
            except (ValueError, KeyError):
                continue
            if oldest_ts < cutoff:
                del pending[pid]
                dropped += 1
        if dropped:
            self._save_pending(pending)
        return dropped

    def list_pending(self) -> list[dict]:
        """Serializable list of all pending records."""
        pending = self._load_pending()
        out = []
        for pid, rec in pending.items():
            out.append({
                "pending_id": pid,
                "suggested_name": rec["suggested_name"],
                "created_at": rec["created_at"],
                "distinct_sessions": len({o["session_id"] for o in rec["observations"]}),
                "observation_count": len(rec["observations"]),
            })
        return out


# ---------------------------------------------------------------- self-test
def _selftest() -> bool:
    import shutil
    import tempfile

    test_dir = Path(tempfile.mkdtemp(prefix="wsbuild_pp_", dir="/tmp"))
    try:
        pp = ProvisionalProfiles(test_dir)
        ok = True

        # Base embedding + small perturbations all > 0.75 cosine
        rng = np.random.default_rng(42)
        base = rng.normal(size=256)
        base = base / np.linalg.norm(base)

        def perturb(scale=0.005):
            v = base + rng.normal(scale=scale, size=256)
            return (v / np.linalg.norm(v)).tolist()

        emb1 = perturb()
        emb2 = perturb()
        emb3 = perturb()

        # Step 1: add unknown from session A
        pid = pp.add_unknown(emb1, suggested_name="Maria", session_id="sess-A")
        step1 = isinstance(pid, str) and len(pp.list_pending()) == 1
        print(f"step1 add_unknown: {'PASS' if step1 else 'FAIL'}")
        ok = ok and step1

        # Step 2: observe from session B → not promoted (only 2 distinct)
        r2 = pp.observe(pid, emb2, "sess-B")
        step2 = r2.get("promoted") is False and r2.get("matches") == 2
        print(f"step2 observe sess-B (no promote): {'PASS' if step2 else 'FAIL'} -> {r2}")
        ok = ok and step2

        # Idempotency: same session B again should not bump matches
        r2b = pp.observe(pid, emb2, "sess-B")
        step2b = r2b.get("promoted") is False and r2b.get("matches") == 2
        print(f"step2b observe sess-B repeat (idempotent): {'PASS' if step2b else 'FAIL'} -> {r2b}")
        ok = ok and step2b

        # Step 3: observe from session C → promotes
        r3 = pp.observe(pid, emb3, "sess-C")
        step3 = r3.get("promoted") is True and r3.get("profile_name") == "Maria"
        print(f"step3 observe sess-C (promote): {'PASS' if step3 else 'FAIL'} -> {r3}")
        ok = ok and step3

        # profiles.json should now contain Maria with expected schema
        profiles = json.loads((test_dir / "profiles.json").read_text())
        step3b = (
            "Maria" in profiles
            and isinstance(profiles["Maria"].get("embedding"), list)
            and profiles["Maria"].get("total_seconds") == 30.0
        )
        print(f"step3b profiles.json shape: {'PASS' if step3b else 'FAIL'}")
        ok = ok and step3b

        # pending should be empty for Maria now
        step3c = len(pp.list_pending()) == 0
        print(f"step3c pending cleared after promotion: {'PASS' if step3c else 'FAIL'}")
        ok = ok and step3c

        # Step 4: rollback Maria
        n = pp.rollback("Maria")
        profiles2 = json.loads((test_dir / "profiles.json").read_text())
        step4 = n >= 1 and "Maria" not in profiles2
        print(f"step4 rollback: {'PASS' if step4 else 'FAIL'} -> removed={n}")
        ok = ok and step4

        # Binding log should have entries
        log_lines = (test_dir / "binding_log.jsonl").read_text().splitlines()
        step5 = len(log_lines) >= 3 and any("promoted" in ln for ln in log_lines) and any("rolled_back" in ln for ln in log_lines)
        print(f"step5 binding_log contents: {'PASS' if step5 else 'FAIL'} -> lines={len(log_lines)}")
        ok = ok and step5

        return ok
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    passed = _selftest()
    print("OK" if passed else "FAILED")
