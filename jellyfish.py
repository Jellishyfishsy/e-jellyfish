#!/usr/bin/env python3
"""
ASCII Jellyfish
===============
A softly lit 3D jellyfish - pulsing bell, frilly oral arms, drifting tentacles -
drawn entirely with text characters in your terminal.

    python jellyfish.py                    # run it (maximise the window; a smaller font = more detail)
    python jellyfish.py --palette orchid   # start on another palette
    python jellyfish.py --no-intro         # skip the opening where the tentacles drop in
    python jellyfish.py --snapshot 9       # print the frame at t = 9 s and exit
    python jellyfish.py --bg 000000        # tell it your terminal's background colour (default 0c0c0c on Windows)

Keys:  p  next palette    g  glow on/off    space  pause    q  quit
"""
import argparse
import math
import os
import shutil
import sys
import time

import numpy as np

TAU = 2 * math.pi
SX, SY = 2, 4  # sub-samples per character cell (a cell is about 1:2, so sub-cells are square)

# fill characters, faint -> dense.  / \ | _ - are kept back for outlines, the rim and tentacles.
RAMP = np.array(list(".,:;i!l1rtcjzvx=+<>{}[]nuoaesZXO0Q#%&8B$@"))

PALETTES = [  # "bell" runs from the crown down to the rim
    {"name": "abyss", "bell": [(170, 210, 255), (75, 125, 255), (48, 88, 245), (42, 64, 218), (36, 50, 190)],
     "arms": (175, 140, 255), "tent": (180, 198, 240), "spark": (140, 210, 255)},
    {"name": "orchid", "bell": [(255, 228, 100), (255, 170, 120), (255, 95, 165), (200, 60, 220), (125, 55, 240)],
     "arms": (70, 225, 255), "tent": (255, 182, 212), "spark": (255, 220, 110)},
    {"name": "ember", "bell": [(255, 50, 40), (255, 70, 42), (255, 110, 48), (255, 160, 60), (255, 205, 95)],
     "arms": (182, 145, 255), "tent": (255, 200, 112), "spark": (255, 160, 70)},
    {"name": "coral", "bell": [(255, 220, 95), (250, 120, 75), (240, 70, 62), (220, 40, 62), (185, 30, 72)],
     "arms": (236, 236, 246), "tent": (246, 176, 192), "spark": (255, 120, 150)},
    {"name": "lagoon", "bell": [(205, 255, 240), (110, 240, 215), (50, 200, 205), (30, 150, 205), (35, 90, 205)],
     "arms": (255, 130, 200), "tent": (165, 236, 226), "spark": (120, 255, 210)},
]
PAL_NAMES = [p["name"] for p in PALETTES]
MINI_COLOURS = [(90, 210, 255), (255, 110, 170), (170, 120, 255), (255, 95, 95), (120, 255, 170), (255, 200, 90)]
DUST_COLOURS = [(120, 150, 210), (200, 140, 200), (150, 210, 220), (210, 210, 230)]

# body plan
A_MAX = 0.60 * math.pi  # how far round the profile goes (past the equator = the curled-in skirt)
HB = 0.68               # bell height relative to its radius
LAPPETS = 16            # scallops round the rim
N_CANALS = 16           # radial canals drawn on the lower bell
N_ARMS = 4
ARM_LEN = 1.75          # oral arm length below the rim, in bell radii
TENT_LEN = 2.75         # tentacle length, in bell radii
PERIOD = 2.8            # seconds per beat
SQUEEZE = 0.32          # fraction of a beat spent contracting

LIGHT = np.array([-0.5, 0.72, 0.62])
LIGHT /= np.linalg.norm(LIGHT)
HALF = LIGHT + np.array([0.0, 0.0, 1.0])
HALF /= np.linalg.norm(HALF)

GLOW_GAIN = 0.30  # how much of the body's light spills onto the background
GLOW_MAX = 40     # cap on the light added to a background cell (per channel)
GLOW_STEP = 2     # glow is rounded to this step: smaller = smoother, but more escape codes

PAL_FIRST = 8.0  # first automatic palette change
PAL_HOLD = 6.0   # seconds between changes
PAL_SWEEP = 2.2  # seconds a new palette takes to wash from the crown to the tentacle tips


# ----------------------------------------------------------------------------- helpers
def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def ease_out(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def pulse(t):
    """0 = relaxed, 1 = fully squeezed: a quick squeeze, then a slow release."""
    ph = (np.asarray(t, dtype=float) / PERIOD) % 1.0
    return np.where(ph < SQUEEZE, smooth(ph / SQUEEZE), 1 - smooth((ph - SQUEEZE) / (1 - SQUEEZE)))


def gradient(stops, s):
    stops = np.asarray(stops, dtype=float)
    f = np.clip(s, 0, 1) * (len(stops) - 1)
    i = np.minimum(f.astype(int), len(stops) - 2)
    k = (f - i)[..., None]
    return stops[i] * (1 - k) + stops[i + 1] * k


def flat(colour, s):
    return np.broadcast_to(np.asarray(colour, dtype=float), np.shape(s) + (3,))


def hash01(a, b, seed):
    """Stable pseudo-random value in [0, 1] per integer lattice point."""
    h = a.astype(np.int64) * 73856093 ^ b.astype(np.int64) * 19349663 ^ seed
    h = (h ^ (h >> 13)) * 1274126177
    return ((h ^ (h >> 16)) & 0xFFFF) / 65535.0


def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def box_blur(a, r, axis):
    if r <= 0:
        return a
    n = a.shape[axis]
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r + 1, r)
    cs = np.cumsum(np.pad(a, pad), axis=axis)
    hi = np.take(cs, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
    lo = np.take(cs, np.arange(n), axis=axis)
    return (hi - lo) / (2 * r + 1)


def line_glyphs(col, row):
    """Choose | / \\ _ - for each point of a polyline from its local screen direction."""
    dc = np.gradient(col, axis=-1)
    dr = np.gradient(row, axis=-1)
    slope = np.abs(dr) / (np.abs(dc) + 1e-9)
    flatg = np.where(row % 1.0 > 0.55, "_", "-")
    return np.where(slope >= 2.4, "|", np.where(slope >= 0.42, np.where(dc * dr > 0, "\\", "/"), flatg))


def edge_glyphs(gx, gy, cov, aspect):
    """Outline character for a silhouette cell; (gx, gy) points into the shape."""
    slope = np.abs(gx) / (aspect * np.abs(gy) + 1e-9)
    flatg = np.where(gy > 0, np.where(cov < 0.5, "_", "-"), np.where(cov < 0.5, "-", "_"))
    return np.where(slope >= 2.4, "|", np.where(slope >= 0.42, np.where(gx * gy > 0, "/", "\\"), flatg))


class View:
    """Body space -> camera space (sway, then a slight upward look) -> character cells."""

    def __init__(self, cx, cy, aspect, pitch, roll):
        self.cx, self.cy, self.aspect = cx, cy, aspect
        self.cp, self.sp = math.cos(pitch), math.sin(pitch)
        self.cr, self.sr = math.cos(roll), math.sin(roll)

    def rot(self, v):
        x, y, z = v[..., 0], v[..., 1], v[..., 2]
        x1 = x * self.cr - y * self.sr
        y1 = x * self.sr + y * self.cr
        return x1, y1 * self.cp + z * self.sp, -y1 * self.sp + z * self.cp

    def project(self, x, y):
        return self.cx + x * self.aspect, self.cy - y


class Colours:
    """Palette state; each change washes down the body from the crown to the tentacle tips."""

    def __init__(self, start, auto):
        self.cur = self.prev = start
        self.changed = -1e9
        self.auto = auto
        self.next_at = PAL_FIRST
        self.count = 0

    def switch(self, t):
        self.prev = self.cur
        self.cur = (self.cur + 1) % len(PALETTES)
        self.changed = t
        self.next_at = t + PAL_HOLD
        self.count += 1

    def update(self, t):
        if self.auto and t >= self.next_at:
            self.switch(t)

    @property
    def name(self):
        return PAL_NAMES[self.cur]

    def get(self, t, s, fn):
        """fn(palette, s) -> colours; s runs 0 (crown) .. ~3 (tentacle tips)."""
        new = fn(PALETTES[self.cur], s)
        p = (t - self.changed) / PAL_SWEEP
        if p >= 2.0:
            return new
        old = fn(PALETTES[self.prev], s)
        k = smooth((p * 3.4 - np.asarray(s)) * 1.3)[..., None]
        return old + (new - old) * k


class MiniJelly:
    """A small, far-away jelly drawn as a little sprite."""

    def __init__(self, rng, w, h, colour):
        self.w = int(rng.integers(4, 7))
        self.x = rng.uniform(0.04, 0.96) * w
        self.y = rng.uniform(0, h)
        self.v = rng.uniform(0.15, 0.45)
        self.ph = rng.uniform(0, TAU)
        self.n = int(rng.integers(3, 6))
        self.colour = np.array(colour, dtype=float) * rng.uniform(0.6, 0.85)

    def draw(self, ch, fg, t):
        h, w = ch.shape
        span = h + self.n + 6
        top = int(round((self.y - self.v * t) % span - (self.n + 3)))
        left = int(round(self.x + 1.5 * math.sin(0.35 * t + self.ph) - self.w / 2))
        squeezed = float(pulse(t * 0.9 + self.ph * 3)) > 0.55
        ww = self.w
        rows = [" ." + "-" * (ww - 4) + ". ",
                (" (" + ":" * (ww - 4) + ") ") if squeezed else ("(" + ":" * (ww - 2) + ")")]
        shift = int(t * 3 + self.ph * 5)
        for k in range(self.n):
            rows.append(" " + "".join("()"[(i + k + shift) % 2] for i in range(ww - 2)) + " ")
        for k, text in enumerate(rows):
            r = top + k
            if not 0 <= r < h:
                continue
            fade = 1.0 if k < 2 else 0.85 - 0.12 * (k - 2)
            for i, g in enumerate(text):
                c = left + i
                if g != " " and 0 <= c < w:
                    ch[r, c] = g
                    fg[r, c] = self.colour * fade


# ----------------------------------------------------------------------------- scene
class Scene:
    def __init__(self, w, h, tentacles, aspect):
        self.W, self.H = w, h - 1  # last row is the status bar
        self.aspect = aspect
        self.R = max(3.0, min(0.26 * self.H, 0.30 * w / aspect))
        self.base_cx = w / 2.0
        self.base_cy = 0.05 * self.H + 0.8 * self.R
        rng = np.random.default_rng(7)
        self.rng = rng
        self.n_tent = tentacles
        self.t_len = rng.uniform(0.78, 1.0, tentacles)
        self.t_ph = rng.uniform(0, TAU, tentacles)
        self.t_amp = rng.uniform(0.7, 1.3, tentacles)
        self.t_jit = rng.uniform(-0.12, 0.12, tentacles)
        n_mini = max(3, min(7, w * self.H // 1300))
        self.minis = [MiniJelly(rng, w, self.H, MINI_COLOURS[i % len(MINI_COLOURS)]) for i in range(n_mini)]
        n_dust = max(20, w * self.H // 80)
        self.dust = (rng.uniform(0, w, n_dust), rng.uniform(0, self.H, n_dust),
                     rng.uniform(0.05, 0.3, n_dust), rng.uniform(0, TAU, n_dust),
                     np.array(DUST_COLOURS, dtype=float)[rng.integers(0, len(DUST_COLOURS), n_dust)])
        self.sp_pos = np.zeros((0, 2))
        self.sp_vel = np.zeros((0, 2))
        self.sp_age = np.zeros(0)
        self.sp_life = np.zeros(0)
        self.sp_col = np.zeros((0, 3))
        self.beat_index = 0
        self.beats = 0
        self.letters = 0

    # --- motion -------------------------------------------------------------
    def pose(self, t):
        c = float(pulse(t))
        spin = 0.22 * t
        roll = 0.07 * math.sin(0.37 * t) + 0.03 * math.sin(0.83 * t + 1.0)
        pitch = 0.12 + 0.04 * math.sin(0.29 * t)
        cx = self.base_cx + 0.06 * self.R * self.aspect * math.sin(0.21 * t)
        cy = self.base_cy - 0.10 * self.R * c + 0.05 * self.R * math.sin(0.25 * t)
        return c, spin, View(cx, cy, self.aspect, pitch, roll)

    @staticmethod
    def intro(t):
        return min(1.0, max(t, 0.0) / 0.8), ease_out((t - 0.5) / 2.2), ease_out((t - 1.2) / 2.8)

    def bell_surface(self, t, c, spin, s, ph):
        R = self.R
        a = s * A_MAX
        r = R * np.sin(a) ** 0.8
        y = HB * R * np.cos(a)
        r = r * (1 - 0.2 * c * s ** 1.5) * (1 + 0.012 * np.sin(11 * s - 4.5 * t))
        y = y * (1 + 0.1 * c)
        lap = smooth((s - 0.8) / 0.2)
        wav = np.cos(LAPPETS * ph)
        r = r * (1 + 0.04 * lap * wav)
        y = y - 0.05 * R * lap * (0.5 + 0.5 * wav)
        x, y, z = np.broadcast_arrays(r * np.cos(ph + spin), y, r * np.sin(ph + spin))
        return np.stack([x, y, z], -1)

    # --- sparks ---------------------------------------------------------------
    def step(self, t, dt, colours):
        colours.update(t)
        idx = int(t // PERIOD)
        if idx > self.beat_index:
            self.beat_index = idx
            if t > 2.0:
                self.beats += 1
                self.spawn(t, colours)
        if self.sp_age.size:
            self.sp_pos += self.sp_vel * dt
            self.sp_vel *= max(0.0, 1 - 1.4 * dt)
            self.sp_vel[:, 1] -= 0.9 * dt  # sparks drift upwards
            self.sp_age += dt
            alive = self.sp_age < self.sp_life
            self.sp_pos, self.sp_vel = self.sp_pos[alive], self.sp_vel[alive]
            self.sp_age, self.sp_life, self.sp_col = self.sp_age[alive], self.sp_life[alive], self.sp_col[alive]

    def spawn(self, t, colours):
        c, spin, view = self.pose(t)
        ph = np.linspace(0, TAU, 64, endpoint=False)[None, :]
        rim = self.bell_surface(t, c, spin, np.array([[1.0]]), ph)[0]
        x, y, _ = view.rot(rim)
        col, row = view.project(x, y)
        n = int(self.rng.integers(10, 18))
        pick = self.rng.integers(0, col.size, n)
        pos = np.stack([col[pick], row[pick]], -1) + self.rng.normal(0, 0.4, (n, 2))
        d = np.stack([(pos[:, 0] - view.cx) / self.aspect, pos[:, 1] - view.cy], -1)
        d = unit(d)
        speed = self.rng.uniform(2.0, 6.0, n)[:, None]
        vel = d * speed * np.array([self.aspect, 1.0]) + np.array([0.0, 1.5])
        colour = colours.get(t, np.zeros(n), lambda p, s: flat(p["spark"], s))
        self.sp_pos = np.vstack([self.sp_pos, pos])
        self.sp_vel = np.vstack([self.sp_vel, vel])
        self.sp_age = np.concatenate([self.sp_age, np.zeros(n)])
        self.sp_life = np.concatenate([self.sp_life, self.rng.uniform(1.6, 3.4, n)])
        self.sp_col = np.vstack([self.sp_col, colour])

    # --- drawing --------------------------------------------------------------
    def render(self, t, colours, glow=True):
        W, H, R, A = self.W, self.H, self.R, self.aspect
        c, spin, view = self.pose(t)
        fade, arm_grow, drop = self.intro(t)
        fills, lines = [], []

        # bell -------------------------------------------------------------
        n_s = max(20, int(1.7 * R * SY * 1.2))
        n_p = max(48, int(TAU * R * A * SX * 1.05))
        s = np.linspace(0.004, 1.0, n_s)[:, None]
        ph = np.linspace(0.0, TAU, n_p, endpoint=False)[None, :]
        P = self.bell_surface(t, c, spin, s, ph)
        nrm = unit(np.cross(np.roll(P, -1, 1) - np.roll(P, 1, 1), np.gradient(P, axis=0)))
        X, Y, Z = view.rot(P)
        nx, ny, nz = view.rot(nrm)
        front = nz > 0
        sg = np.where(front, 1.0, -1.0)
        nx, ny, nz = nx * sg, ny * sg, nz * sg
        lam = np.clip(nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2], 0, None)
        spec = np.clip(nx * HALF[0] + ny * HALF[1] + nz * HALF[2], 0, None) ** 28
        fres = (1 - np.abs(nz)) ** 2
        sv = s[:, 0]
        depth_down = (1 - np.cos(sv * A_MAX)) / (1 - math.cos(A_MAX))  # 0 at the crown, 1 at the rim
        outer = colours.get(t, sv, lambda p, q: gradient(p["bell"], depth_down))[:, None, :]
        inner = colours.get(t, sv, lambda p, q: gradient(p["bell"], 0.55 + 0.45 * depth_down))[:, None, :] * 0.85
        rgb = np.where(front[..., None],
                       outer * (0.30 + 0.70 * lam + 0.25 * fres)[..., None] + (spec * 235)[..., None],
                       inner * (0.30 + 0.45 * lam + 0.20 * fres)[..., None]) * fade
        bright = np.where(front, 0.12 + 0.60 * lam + 0.55 * spec + 0.30 * fres,
                          0.07 + 0.28 * lam + 0.12 * fres + 0.08 * (1 - s)) * fade
        ts, tp = max(8, int(1.9 * R)), max(16, int(TAU * R * A / 1.2))
        tex = hash01(np.floor(s * ts) + 0 * ph, np.floor(ph / TAU * tp) + 0 * s, 11)
        col, row = view.project(X, Y)
        fills.append((col, row, Z, bright, rgb, tex))

        rim_rgb = rgb[-1:] * 0.75 + 255 * 0.25 * fade
        lines.append((col[-1:], row[-1:], Z[-1:], np.minimum(1.3, bright[-1:] * 1.2 + 0.2 * fade), rim_rgb, 0.12 * R))
        j = (np.arange(N_CANALS) * n_p) // N_CANALS
        i0 = int(0.6 * n_s)
        lines.append((col[i0:, j].T, row[i0:, j].T, Z[i0:, j].T, bright[i0:, j].T * 0.8,
                      rgb[i0:, j].transpose(1, 0, 2) * 0.9, 0.12 * R))

        rim_y = HB * R * math.cos(A_MAX) * (1 + 0.1 * c)

        # oral arms --------------------------------------------------------
        if arm_grow > 0.02:
            alen = ARM_LEN * R * arm_grow
            n_u = max(10, int((alen + 0.5 * R) * SY * 1.1))
            n_v = max(5, int(0.42 * R * A * SX * 1.0))
            u = np.linspace(0, 1, n_u)[:, None]
            v = np.linspace(-1, 1, n_v)[None, :]
            lagc = pulse(t - 0.3 - 0.7 * u)
            base = colours.get(t, 0.9 + 0.9 * u[:, 0], lambda p, q: flat(p["arms"], q))[:, None, :]
            base = base + (255 - base) * 0.12 * (1 - u[..., None])
            for k in range(N_ARMS):
                ang = 0.85 * spin + k * TAU / N_ARMS + 1.8 * u + 0.3 * math.sin(0.7 * t + 1.9 * k)
                ca, sa = np.cos(ang), np.sin(ang)
                rad = R * (0.06 + 0.12 * u) * (1 + 0.25 * np.sin(3.0 * u - 1.4 * t + k))
                width = R * 0.21 * (1 - 0.45 * u) * (1 + 0.3 * np.sin(12 * u - 3.0 * t + 2.1 * k))
                ruffle = R * 0.07 * np.sin(8 * v + 10 * u - 4.2 * t + k) * np.abs(v) ** 1.5
                sway = R * 0.22 * u ** 2 * math.sin(0.45 * t + 1.3 * k)
                yc = rim_y + 0.5 * R - (alen + 0.5 * R) * u * (1 - 0.1 * lagc)
                px = (rad + v * width) * ca - ruffle * sa + sway
                pz = (rad + v * width) * sa + ruffle * ca
                py = yc + 0.04 * R * np.sin(6 * v - 3 * u + 2.5 * t)
                Pa = np.stack(np.broadcast_arrays(px, py, pz), -1)
                na = unit(np.cross(np.gradient(Pa, axis=0), np.gradient(Pa, axis=1)))
                Xa, Ya, Za = view.rot(Pa)
                ax, ay, az = view.rot(na)
                lam_a = np.abs(ax * LIGHT[0] + ay * LIGHT[1] + az * LIGHT[2])
                spec_a = np.abs(ax * HALF[0] + ay * HALF[1] + az * HALF[2]) ** 20
                fres_a = (1 - np.abs(az)) ** 2
                rgb_a = (base * (0.48 + 0.6 * lam_a + 0.2 * fres_a)[..., None] + (spec_a * 150)[..., None]) * fade
                b_a = (0.2 + 0.6 * lam_a + 0.4 * spec_a + 0.2 * fres_a) * fade
                tex_a = hash01(np.floor(u * n_u / 3) + 0 * v, np.floor((v + 1) * n_v / 4) + 100 * k + 0 * u, 5)
                ca_, ra_ = view.project(Xa, Ya)
                fills.append((ca_, ra_, Za, b_a, rgb_a, tex_a))

        # tentacles --------------------------------------------------------
        if drop > 0.01:
            n = self.n_tent
            m = max(24, int(TENT_LEN * R * 3.2))
            u = np.linspace(0, 1, m)[None, :]
            ang = spin + TAU * np.arange(n)[:, None] / n + self.t_jit[:, None] - 0.3 * u
            lagc = pulse(t - 1.2 * u)
            length = TENT_LEN * R * self.t_len[:, None] * drop
            r = R * math.sin(A_MAX) ** 0.8 * 0.95 * (1 - 0.2 * lagc) + 0.30 * R * u ** 1.3
            amp = self.t_amp[:, None] * R * u ** 1.2
            w1 = amp * 0.16 * np.sin(5.5 * u - 1.9 * t + self.t_ph[:, None])
            w2 = amp * 0.12 * np.cos(4.0 * u - 1.3 * t + 1.7 * self.t_ph[:, None])
            px = (r + w1) * np.cos(ang) - w2 * np.sin(ang)
            pz = (r + w1) * np.sin(ang) + w2 * np.cos(ang)
            py = rim_y - 0.02 * R - length * u * (1 - 0.08 * lagc)
            Pt = np.stack(np.broadcast_arrays(px, py, pz), -1)
            Xt, Yt, Zt = view.rot(Pt)
            ct, rt = view.project(Xt, Yt)
            depth = np.clip(0.5 + 0.5 * Zt / R, 0, 1)
            b_t = (0.35 + 0.75 * (1 - u) ** 0.8) * (0.5 + 0.5 * depth) * fade
            base = colours.get(t, 1.0 + 1.6 * u, lambda p, q: flat(p["tent"], q))
            rgb_t = base * (0.5 + 0.6 * b_t)[..., None]
            lines.append((ct, rt, Zt, b_t, rgb_t, 0.06 * R))
            # a second, fainter strand alongside the upper part makes the roots look thicker
            k = max(4, int(0.4 * m))
            thick = (0.35 + 0.02 * R) * A * (1 - u[:, :k] / 0.4)
            side = np.where(self.t_ph[:, None] > math.pi, 1.0, -1.0)
            lines.append((ct[:, :k] + side * thick, rt[:, :k], Zt[:, :k] - 0.01, b_t[:, :k] * 0.75,
                          rgb_t[:, :k] * 0.75, 0.06 * R))

        # compose ------------------------------------------------------------
        ch = np.full((H, W), " ", dtype="<U1")
        fg = np.zeros((H, W, 3))
        self.draw_dust(ch, fg, t)
        for mj in self.minis:
            mj.draw(ch, fg, t)

        cov, cb, crgb, cz, gx, gy, ctex = self.resolve_fill(fills)
        present = cov >= 0.12
        empty = np.pad(~present, 1, constant_values=True)
        lonely = empty[:-2, 1:-1] | empty[2:, 1:-1] | empty[1:-1, :-2] | empty[1:-1, 2:]
        edge = present & ((cov < 0.8) | lonely) & (np.hypot(gx, gy) > 0.25)
        level = np.rint(np.clip(cb, 0, 1.2) ** 1.1 * (len(RAMP) - 1) * 0.95 + (ctex - 0.5) * 11.0)
        level = np.clip(level, 2, len(RAMP) - 1).astype(int)
        main = np.where(edge, edge_glyphs(gx, gy, cov, A), RAMP[level])
        lum = np.where(present, np.where(edge, cb * 1.15 + 0.1, cb), 0.0)
        mrgb = np.where(edge[..., None], crgb * 1.12, crgb)

        lidx, lglyph, lrgb, lb = self.resolve_lines(lines, cz, cov)
        main = main.reshape(-1)
        mrgb = mrgb.reshape(-1, 3)
        lum = lum.reshape(-1)
        present = present.reshape(-1)
        main[lidx] = lglyph
        mrgb[lidx] = lrgb
        lum[lidx] = lb
        present[lidx] = True
        main, mrgb = main.reshape(H, W), mrgb.reshape(H, W, 3)
        lum, present = lum.reshape(H, W), present.reshape(H, W)

        ch = np.where(present, main, ch)
        fg = np.where(present[..., None], mrgb, fg)
        self.letters = int(present.sum())
        self.draw_sparks(ch, fg, present & (lum > 0.3))

        light = None
        if glow:  # soft light spilling from the body onto the background, added on top of it later
            src = mrgb * np.clip(lum, 0, 1)[..., None] * present[..., None]
            ry = max(1, int(round(R * 0.3)))
            rx = max(1, int(round(ry * A)))  # same reach across and down on screen
            for _ in range(3):  # three box passes ~ a smooth gaussian, so cell blocks don't show
                src = box_blur(box_blur(src, rx, 1), ry, 0)
            light = np.minimum(src * GLOW_GAIN, GLOW_MAX)
        return ch, fg, light

    def resolve_fill(self, fills):
        W, H, A = self.W, self.H, self.aspect
        col = np.concatenate([f[0].ravel() for f in fills])
        row = np.concatenate([f[1].ravel() for f in fills])
        z = np.concatenate([f[2].ravel() for f in fills])
        b = np.concatenate([f[3].ravel() for f in fills])
        rgb = np.concatenate([f[4].reshape(-1, 3) for f in fills])
        tex = np.concatenate([f[5].ravel() for f in fills])
        fw, fh = W * SX, H * SY
        fx = np.floor(col * SX).astype(np.int64)
        fy = np.floor(row * SY).astype(np.int64)
        ok = (fx >= 0) & (fx < fw) & (fy >= 0) & (fy < fh)
        idx = (fy * fw + fx)[ok]
        z, b, rgb, tex = z[ok], b[ok], rgb[ok], tex[ok]
        n = fw * fh
        zbuf = np.full(n, -np.inf)
        np.maximum.at(zbuf, idx, z)  # z-buffer: nearest sample per sub-cell wins
        sel = np.flatnonzero(z >= zbuf[idx])
        pix = idx[sel]
        occ = np.zeros(n)
        occ[pix] = 1.0
        fb = np.zeros(n)
        fb[pix] = b[sel]
        frgb = np.zeros((n, 3))
        frgb[pix] = rgb[sel]
        fz = np.full(n, -1e9)
        fz[pix] = z[sel]
        ft = np.zeros(n)
        ft[pix] = tex[sel]

        o4 = occ.reshape(H, SY, W, SX)
        cnt = o4.sum((1, 3))
        den = np.maximum(cnt, 1)
        cov = cnt / (SX * SY)
        cb = fb.reshape(H, SY, W, SX).sum((1, 3)) / den
        crgb = frgb.reshape(H, SY, W, SX, 3).sum((1, 3)) / den[..., None]
        cz = fz.reshape(H, SY, W, SX).max((1, 3))
        ctex = ft.reshape(H, SY, W, SX).sum((1, 3)) / den
        # which way the silhouette runs through each cell (vector points into the shape)
        gx = (o4[..., SX - 1].mean(1) - o4[..., 0].mean(1)) / 0.5
        gy = (o4[:, SY // 2:].mean((1, 3)) - o4[:, :SY // 2].mean((1, 3))) / (A / 2)
        pc = np.pad(cov, 1)
        gx = gx + (pc[1:-1, 2:] - pc[1:-1, :-2]) / 2
        gy = gy + (pc[2:, 1:-1] - pc[:-2, 1:-1]) / (2 * A)
        return cov, cb, crgb, cz, gx, gy, ctex

    def resolve_lines(self, lines, cz, cov):
        W, H = self.W, self.H
        cols, rows, zs, bs, rgbs, gs, biases = [], [], [], [], [], [], []
        for col, row, z, b, rgb, bias in lines:
            gs.append(line_glyphs(col, row).ravel())
            cols.append(col.ravel())
            rows.append(row.ravel())
            zs.append(z.ravel())
            bs.append(np.broadcast_to(b, col.shape).ravel())
            rgbs.append(np.broadcast_to(rgb, col.shape + (3,)).reshape(-1, 3))
            biases.append(np.full(col.size, bias))
        col, row, z = np.concatenate(cols), np.concatenate(rows), np.concatenate(zs)
        b, rgb, g, bias = np.concatenate(bs), np.concatenate(rgbs), np.concatenate(gs), np.concatenate(biases)
        c = np.floor(col).astype(np.int64)
        r = np.floor(row).astype(np.int64)
        ok = (c >= 0) & (c < W) & (r >= 0) & (r < H)
        idx = (r * W + c)[ok]
        z, b, rgb, g, bias = z[ok], b[ok], rgb[ok], g[ok], bias[ok]
        vis = (z >= cz.ravel()[idx] - bias) | (cov.ravel()[idx] < 0.35)
        idx, z, b, rgb, g = idx[vis], z[vis], b[vis], rgb[vis], g[vis]
        zbuf = np.full(W * H, -np.inf)
        np.maximum.at(zbuf, idx, z)
        sel = np.flatnonzero(z >= zbuf[idx])
        return idx[sel], g[sel], rgb[sel], b[sel]

    def draw_dust(self, ch, fg, t):
        x, y, v, ph, colour = self.dust
        h, w = ch.shape
        rows = np.floor((y - v * t) % h).astype(int)
        cols = np.floor(x + 0.6 * np.sin(0.3 * t + ph)).astype(int) % w
        tw = 0.5 + 0.5 * np.sin(1.7 * t + 3 * ph)
        ch[rows, cols] = np.where(tw > 0.93, "+", np.where(tw > 0.35, ".", " "))
        fg[rows, cols] = colour * (0.25 + 0.45 * tw)[:, None]

    def draw_sparks(self, ch, fg, blocked):
        if not self.sp_age.size:
            return
        h, w = ch.shape
        c = np.floor(self.sp_pos[:, 0]).astype(int)
        r = np.floor(self.sp_pos[:, 1]).astype(int)
        ok = (c >= 0) & (c < w) & (r >= 0) & (r < h)
        c, r = c[ok], r[ok]
        life = (self.sp_age / self.sp_life)[ok]
        keep = ~blocked[r, c]
        c, r, life = c[keep], r[keep], life[keep]
        ch[r, c] = np.where(life < 0.25, "*", np.where(life < 0.6, "+", "."))
        fg[r, c] = self.sp_col[ok][keep] * (1 - life)[:, None] ** 0.6


# ----------------------------------------------------------------------------- output
_FG, _BG = {}, {}


def bg_code(rgb):
    return "\x1b[48;2;%d;%d;%dm" % tuple(int(v) for v in rgb)


def parse_bg(text):
    if text == "auto":
        return (12, 12, 12) if os.name == "nt" else (0, 0, 0)  # Windows Terminal / conhost default is #0c0c0c
    value = text.lstrip("#")
    if len(value) != 6:
        raise argparse.ArgumentTypeError("use a hex colour like 0c0c0c")
    try:
        return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        raise argparse.ArgumentTypeError("use a hex colour like 0c0c0c") from None


def to_ansi(ch, fg, light, color, base):
    """Every cell is painted with the base background plus any glow, so the glow can only
    ever lighten the background - it never leaves a ring darker than the terminal's own."""
    h, w = ch.shape
    if not color:
        return "\n".join("".join(row) for row in ch.tolist())
    q = (np.clip(fg, 0, 255).astype(np.int64) >> 3) << 3
    fgk = ((q[..., 0] << 16) | (q[..., 1] << 8) | q[..., 2]).tolist()
    b = np.broadcast_to(np.asarray(base, dtype=np.int64), (h, w, 3))
    if light is not None:
        b = b + (np.maximum(light, 0) // GLOW_STEP).astype(np.int64) * GLOW_STEP
    b = np.minimum(b, 255)
    bgk = ((b[..., 0] << 16) | (b[..., 1] << 8) | b[..., 2]).tolist()
    chars = ch.tolist()
    out = []
    for r in range(h):
        parts = []
        last_f = last_b = -1
        crow, frow, brow = chars[r], fgk[r], bgk[r]
        for i in range(w):
            k = brow[i]
            if k != last_b:
                code = _BG.get(k)
                if code is None:
                    code = _BG[k] = "\x1b[48;2;%d;%d;%dm" % (k >> 16, (k >> 8) & 255, k & 255)
                parts.append(code)
                last_b = k
            g = crow[i]
            if g != " ":
                k = frow[i]
                if k != last_f:
                    code = _FG.get(k)
                    if code is None:
                        code = _FG[k] = "\x1b[38;2;%d;%d;%dm" % (k >> 16, (k >> 8) & 255, k & 255)
                    parts.append(code)
                    last_f = k
            parts.append(g)
        out.append("".join(parts))
    return "\n".join(out)


def status_line(scene, colours, t, w, color, base, fps=None):
    left = " beats %-4d tentacles %-4d colours %-4d sparks %-5d letters %s" % (
        scene.beats, scene.n_tent, colours.count, scene.sp_age.size, format(scene.letters, ","))
    right = "%s  t %05.2fs   p palette  g glow  space pause  q quit " % (
        colours.name, t) if fps is None else "%s  %4.1f fps  t %05.2fs   p palette  g glow  space pause  q quit " % (
        colours.name, fps, t)
    text = (left + " " * max(1, w - 1 - len(left) - len(right)) + right)[:w - 1]
    return ("\x1b[0m" + bg_code(base) + "\x1b[38;2;95;105;130m" + text + "\x1b[0m") if color else text


class Keys:
    """Non-blocking single-key reads on Windows and Unix."""

    def __init__(self):
        self.win = os.name == "nt"
        self.old = None
        if self.win:
            import msvcrt
            self.msvcrt = msvcrt
        elif sys.stdin.isatty():
            import termios
            import tty
            self.termios = termios
            self.old = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())

    def read(self):
        if self.win:
            return self.msvcrt.getwch() if self.msvcrt.kbhit() else None
        if self.old is None:
            return None
        import select
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    def close(self):
        if self.old is not None:
            self.termios.tcsetattr(sys.stdin, self.termios.TCSADRAIN, self.old)


def terminal_size(override):
    if override:
        w, h = override.lower().split("x")
        return int(w), int(h)
    size = shutil.get_terminal_size((120, 40))
    return size.columns, size.lines


def main():
    ap = argparse.ArgumentParser(description="ASCII jellyfish")
    ap.add_argument("--palette", choices=PAL_NAMES, default="abyss")
    ap.add_argument("--tentacles", type=int, default=14)
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--aspect", type=float, default=2.0, help="character cell height / width (default 2.0)")
    ap.add_argument("--no-cycle", action="store_true", help="stay on one palette")
    ap.add_argument("--no-glow", action="store_true", help="no soft background glow")
    ap.add_argument("--bg", type=parse_bg, default="auto",
                    help="background colour as hex, e.g. 0c0c0c; match your terminal's (auto: 0c0c0c on Windows, else 000000)")
    ap.add_argument("--no-intro", action="store_true", help="start fully grown")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--size", help="force WxH, e.g. 160x50")
    ap.add_argument("--snapshot", type=float, metavar="T", help="print the frame at time T and exit")
    args = ap.parse_args()

    if os.name == "nt":
        os.system("")  # enable ANSI escape handling in the Windows console
    color, glow = not args.no_color, not (args.no_glow or args.no_color)
    colours = Colours(PAL_NAMES.index(args.palette), auto=not args.no_cycle)
    t0 = 5.0 if args.no_intro else 0.0
    if args.no_intro:
        colours.next_at = t0 + PAL_FIRST

    w, h = terminal_size(args.size)
    scene = Scene(w, h, args.tentacles, args.aspect)

    if args.snapshot is not None:
        t, dt = t0, 1 / 30
        while t < t0 + args.snapshot:
            t = min(t + dt, t0 + args.snapshot)
            scene.step(t, dt, colours)
        ch, fg, light = scene.render(t, colours, glow)
        sys.stdout.write(to_ansi(ch, fg, light, color, args.bg) + "\n"
                         + status_line(scene, colours, t, w, color, args.bg) + "\n")
        return

    clear = ("\x1b[0m" + (bg_code(args.bg) if color else "")) + "\x1b[2J"  # clear to the base colour
    out = sys.stdout
    out.write("\x1b[?1049h\x1b[?25l" + clear)  # alternate screen, hide cursor
    keys = Keys()
    t, paused = t0, False
    last = time.perf_counter()
    fps = args.fps
    try:
        while True:
            now = time.perf_counter()
            dt = min(now - last, 0.1)
            last = now
            fps = fps * 0.92 + 0.08 / max(dt, 1e-3)

            k = keys.read()
            if k in ("q", "Q", "\x1b"):
                break
            if k in ("p", "P"):
                colours.switch(t)
            if k in ("g", "G"):
                glow = color and not glow
            if k == " ":
                paused = not paused

            if not args.size:
                nw, nh = terminal_size(None)
                if (nw, nh) != (w, h):
                    w, h = nw, nh
                    scene = Scene(w, h, args.tentacles, args.aspect)
                    out.write(clear)

            if not paused:
                t += dt
                scene.step(t, dt, colours)
            ch, fg, light = scene.render(t, colours, glow)
            out.write("\x1b[H" + to_ansi(ch, fg, light, color, args.bg) + "\n"
                      + status_line(scene, colours, t, w, color, args.bg, fps))
            out.flush()

            spare = 1 / args.fps - (time.perf_counter() - now)
            if spare > 0:
                time.sleep(spare)
    except KeyboardInterrupt:
        pass
    finally:
        keys.close()
        out.write("\x1b[0m\x1b[?25h\x1b[?1049l")
        out.flush()


if __name__ == "__main__":
    main()
