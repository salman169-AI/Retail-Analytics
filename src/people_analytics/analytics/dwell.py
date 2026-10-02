"""Per-identity, per-zone dwell-time accounting.

A person "dwells" in a zone for every frame their triggering anchor falls inside
it. Frame counts convert to seconds via the source fps. Keyed by the stable
``global_id`` from the re-entry gallery so a person who leaves and returns to a
zone accumulates one continuous dwell total.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass
class ZoneDwell:
    zone: str
    global_id: int
    frames: int
    first_frame: int
    last_frame: int

    def seconds(self, fps: float) -> float:
        return self.frames / fps if fps > 0 else 0.0


class DwellTracker:
    """Accumulates frames-in-zone per (zone, global_id).

    `min_dwell_s` drops identities whose total time in a zone is below it from
    the records and summaries, so someone who only cuts across a corner of the
    zone on their way past isn't counted as a visitor with a 0.4 s dwell.
    """

    def __init__(self, zone_names: list[str], fps: float, min_dwell_s: float = 0.0):
        self.fps = fps
        self._min_frames = min_dwell_s * fps
        self._frames: dict[str, dict[int, int]] = {z: defaultdict(int) for z in zone_names}
        self._first: dict[str, dict[int, int]] = {z: {} for z in zone_names}
        self._last: dict[str, dict[int, int]] = {z: {} for z in zone_names}

    def update(self, zone: str, ids_in_zone: list[int], frame_idx: int) -> None:
        for gid in ids_in_zone:
            gid = int(gid)
            self._frames[zone][gid] += 1
            self._first[zone].setdefault(gid, frame_idx)
            self._last[zone][gid] = frame_idx

    def seconds_so_far(self, zone: str, gid: int) -> float:
        """Running dwell for one identity in one zone (for live on-screen timers)."""
        return self._frames[zone].get(int(gid), 0) / self.fps if self.fps > 0 else 0.0

    def _kept(self, per_id: dict[int, int]) -> dict[int, int]:
        return {gid: f for gid, f in per_id.items() if f >= self._min_frames}

    def records(self) -> list[ZoneDwell]:
        out: list[ZoneDwell] = []
        for zone, per_id in self._frames.items():
            for gid, frames in self._kept(per_id).items():
                out.append(
                    ZoneDwell(zone, gid, frames, self._first[zone][gid], self._last[zone][gid])
                )
        return out

    def zone_summary(self) -> dict[str, dict]:
        """Per-zone aggregates: unique visitors + mean/max dwell seconds."""
        summary: dict[str, dict] = {}
        for zone, per_id in self._frames.items():
            kept = self._kept(per_id)
            secs = [f / self.fps for f in kept.values()] if self.fps > 0 else []
            summary[zone] = {
                "unique_visitors": len(kept),
                "mean_dwell_s": round(sum(secs) / len(secs), 2) if secs else 0.0,
                "max_dwell_s": round(max(secs), 2) if secs else 0.0,
            }
        return summary
