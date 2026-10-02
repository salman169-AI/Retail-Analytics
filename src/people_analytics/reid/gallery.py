"""Long-term identity via a re-entry gallery.

The tracker (BoxMOT) gives short-lived ``tracker_id``s: when a person is lost to
occlusion or walks out and back, the tracker mints a *new* id — an identity
switch that corrupts dwell time. The gallery sits on top and restores continuity:

    global_id -> {running embedding, last_seen, zone_history, active tracker_id}

Each frame we get the live tracks' appearance embeddings. A tracker_id we already
know keeps its global_id (and refreshes its embedding via EMA). A *new* tracker_id
is matched by cosine similarity against recently-departed identities within a time
window; on a hit it reclaims that original global_id, otherwise it mints a new one.

This module is deliberately free of any tracker/torch dependency so it unit-tests
with synthetic embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


@dataclass
class GalleryEntry:
    global_id: int
    embedding: np.ndarray  # L2-normalized running embedding
    last_seen_time: float
    last_seen_frame: int
    active_tracker_id: int | None = None
    zone_history: list = field(default_factory=list)


class ReEntryGallery:
    """Maps ephemeral tracker_ids to stable global_ids using appearance re-ID."""

    def __init__(
        self,
        cosine_thresh: float = 0.55,
        time_window_s: float = 120.0,
        max_gallery_size: int = 500,
        ema_alpha: float = 0.9,
    ):
        self.cosine_thresh = cosine_thresh
        self.time_window_s = time_window_s
        self.max_gallery_size = max_gallery_size
        self.ema_alpha = ema_alpha

        self._entries: dict[int, GalleryEntry] = {}
        self._tid_to_gid: dict[int, int] = {}
        self._next_gid = 1
        self.reentry_count = 0  # how many times a returning identity was restored

    def _new_entry(self, tid: int, emb: np.ndarray, t: float, frame: int) -> int:
        gid = self._next_gid
        self._next_gid += 1
        self._entries[gid] = GalleryEntry(gid, emb, t, frame, active_tracker_id=tid)
        self._tid_to_gid[tid] = gid
        self._evict_if_needed()
        return gid

    def _update_entry(self, gid: int, tid: int, emb: np.ndarray, t: float, frame: int) -> None:
        e = self._entries[gid]
        e.embedding = _normalize(self.ema_alpha * e.embedding + (1 - self.ema_alpha) * emb)
        e.last_seen_time = t
        e.last_seen_frame = frame
        e.active_tracker_id = tid

    def _evict_if_needed(self) -> None:
        if len(self._entries) <= self.max_gallery_size:
            return
        # Drop the oldest inactive entry; never evict a currently-active identity.
        inactive = [e for e in self._entries.values() if e.active_tracker_id is None]
        if not inactive:
            return
        victim = min(inactive, key=lambda e: e.last_seen_time)
        del self._entries[victim.global_id]

    def _match(self, emb: np.ndarray, t: float, claimed: set[int]) -> int | None:
        """Best departed identity within the time window above threshold, else None."""
        best_gid, best_sim = None, self.cosine_thresh
        for gid, e in self._entries.items():
            if e.active_tracker_id is not None or gid in claimed:
                continue
            if t - e.last_seen_time > self.time_window_s:
                continue
            sim = float(np.dot(emb, e.embedding))
            if sim >= best_sim:
                best_gid, best_sim = gid, sim
        return best_gid

    def update(
        self,
        tracker_ids: list[int],
        embeddings: np.ndarray,
        timestamp: float,
        frame_idx: int,
    ) -> list[int]:
        """Return global_ids aligned with `tracker_ids` for the current frame."""
        active_now = set(int(t) for t in tracker_ids)
        # Mark identities whose tracker vanished as departed (still matchable).
        for e in self._entries.values():
            if e.active_tracker_id is not None and e.active_tracker_id not in active_now:
                e.active_tracker_id = None

        embeddings = np.asarray(embeddings, dtype=np.float32)
        gids: list[int] = []
        claimed: set[int] = set()
        for i, tid in enumerate(tracker_ids):
            tid = int(tid)
            emb = _normalize(embeddings[i])
            known = self._tid_to_gid.get(tid)
            # A known mapping is only valid if its global_id isn't already claimed
            # this frame — two live tracks can revive onto one id (BoxMOT reviving a
            # buffered id after a re-entry re-link); the collision must be resolved.
            if known is not None and known in self._entries and known not in claimed:
                self._update_entry(known, tid, emb, timestamp, frame_idx)
                gids.append(known)
                claimed.add(known)
                continue
            match = self._match(emb, timestamp, claimed)
            if match is not None:
                self._tid_to_gid[tid] = match
                self._update_entry(match, tid, emb, timestamp, frame_idx)
                self.reentry_count += 1
                gids.append(match)
                claimed.add(match)
            else:
                gid = self._new_entry(tid, emb, timestamp, frame_idx)
                gids.append(gid)
                claimed.add(gid)
        return gids

    @property
    def num_identities(self) -> int:
        return len(self._entries)
