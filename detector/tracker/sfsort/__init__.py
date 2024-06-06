# SPDX-FileCopyrightText: 2024-present dh031200 <imbird0312@gmail.com>
#
# SPDX-License-Identifier: MIT

from .sfsort import SFSORT


def gen_track_args(width, height, fps):
    return {
        "dynamic_tuning": True,
        "cth": 0.5,
        "high_th": 0.4,
        "high_th_m": 0.1,
        "match_th_first": 0.5,
        "match_th_first_m": 0.05,
        "match_th_second": 0.1,
        "low_th": 0.2,
        "new_track_th": 0.3,
        "new_track_th_m": 0.1,
        "marginal_timeout": (7 * fps // 10),
        "central_timeout": fps,
        "horizontal_margin": width // 10,
        "vertical_margin": height // 10,
        "frame_width": width,
        "frame_height": height,
    }


__all__ = (
    "SFSORT",
    "gen_track_args",
)
