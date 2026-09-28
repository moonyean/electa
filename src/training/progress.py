"""TTY 진행 막대와 백그라운드 상태 파일이 공유하는 토큰 기준 ETA."""

import sys
import time

from tqdm import tqdm


def progress_stats(total, seen, initial, elapsed, session_remaining):
    """재개 이전 토큰을 이번 세션의 속도 계산에 포함하지 않는다."""
    new_tokens = max(0, seen-initial)
    rate = new_tokens/elapsed if elapsed > 0 and new_tokens else None
    remaining = max(0, total-seen)
    eta = remaining/rate if rate else None
    if remaining == 0:
        eta = 0.0
    return {"target_tokens": total, "progress_percent": 100*seen/total,
            "remaining_tokens": remaining, "session_tokens": new_tokens,
            "effective_tokens_per_second": rate, "eta_seconds": eta,
            "eta_hours": eta/3600 if eta is not None else None,
            "eta_provisional": elapsed < 300 or not new_tokens,
            "session_remaining_seconds": max(0.0, session_remaining)}


class TrainingProgress:
    def __init__(self, total, initial, started, deadline, margin, stream=None):
        self.total, self.initial, self.last_seen = total, initial, initial
        self.started, self.deadline, self.margin = started, deadline, margin
        stream = stream or sys.stderr
        self.bar = tqdm(total=total, initial=initial, desc="Pretrain", unit="tok",
                        unit_scale=True, dynamic_ncols=True, mininterval=1,
                        disable=not stream.isatty(), file=stream,
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}{postfix}]")

    def snapshot(self, seen):
        return progress_stats(self.total, seen, self.initial, time.monotonic()-self.started,
                              self.deadline-time.time()-self.margin)

    def update(self, seen, loss, lr, step_rate):
        stats = self.snapshot(seen)
        eta = "?" if stats["eta_hours"] is None else f"{stats['eta_hours']:.1f}h"
        provisional = "~" if stats["eta_provisional"] else ""
        self.bar.set_postfix_str(f"loss={loss:.4f}, lr={lr:.2g}, {step_rate:,.0f} tok/s, "
                                 f"ETA={provisional}{eta}", refresh=False)
        self.bar.update(seen-self.last_seen)
        self.last_seen = seen
        return stats

    def close(self):
        self.bar.close()
