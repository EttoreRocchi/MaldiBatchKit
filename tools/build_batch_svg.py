"""Generate docs/batch_correction.svg, the animated figure of the README.

Usage:
    python tools/build_batch_svg.py [OUTPUT]

Needs only the standard library. The scatter plot is a schematic drawn from
synthetic points (fixed seed, so the output is reproducible), not an embedding
of real spectra.

The animation is pure SMIL: GitHub shows README images through <img>, where
JavaScript and external files do not run. For the same reason colours are
mid-tones readable on both themes: inside <img>, prefers-color-scheme follows
the operating system, not the GitHub theme.
"""

import math
import random
import sys
from pathlib import Path

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "docs" / "batch_correction.svg"

W, H = 800, 320
DUR = 7.5  # seconds per loop

# Per-point timeline (seconds), before the per-point offset is added.
T_HOLD_A = 1.2  # + offset; effective hold A per point = 1.5 s (loop wraps)
T_MOVE_AB = 2.0
T_HOLD_B = 2.5
T_MOVE_BA = 1.5
MAX_OFFSET = 0.28  # per-point stagger, must keep end < DUR

SITE_COLORS = ["#E69F00", "#56B4E9", "#009E73"]  # Okabe-Ito
# Darker shade of each site colour for marker outlines (theme-independent).
SITE_STROKES = ["#B07A00", "#2F8BC0", "#00775A"]
SITE_NAMES = ["Site A", "Site B", "Site C"]
SPECIES_NAMES = ["Species 1", "Species 2", "Species 3"]
N_PER = 6  # points per (site, species) -> 54 total

# Plot region
PX0, PX1, PY0, PY1 = 44, 556, 52, 288

# State A: blobs by site; small species sub-structure inside each blob.
SITE_CENTERS = [(150, 128), (302, 226), (448, 126)]
SPECIES_SUBOFF = [(-13, -9), (13, -8), (0, 14)]
# State B: clusters by species; sites fully mixed.
SPECIES_CENTERS = [(140, 214), (300, 112), (458, 212)]

rng = random.Random(20240611)


def clip(x, y, pad=12):
    """Keep a point inside the plot region."""
    return (min(max(x, PX0 + pad), PX1 - pad), min(max(y, PY0 + pad), PY1 - pad))


points = []  # dict(site, sp, a=(x,y), b=(x,y))
for site in range(3):
    for sp in range(3):
        for _ in range(N_PER):
            cx, cy = SITE_CENTERS[site]
            ox, oy = SPECIES_SUBOFF[sp]
            ax = cx + ox + rng.gauss(0, 15)
            ay = cy + oy + rng.gauss(0, 13)
            bx0, by0 = SPECIES_CENTERS[sp]
            bx = bx0 + rng.gauss(0, 22)
            by = by0 + rng.gauss(0, 18)
            points.append({"site": site, "sp": sp, "a": [ax, ay], "b": [bx, by]})

rng.shuffle(points)  # interleave draw order so no colour sits on top


def relax(key, min_d=13.0, iters=200):
    """Push overlapping markers apart with pairwise repulsion."""
    for _ in range(iters):
        moved = False
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                p, q = points[i][key], points[j][key]
                dx, dy = q[0] - p[0], q[1] - p[1]
                d = math.hypot(dx, dy)
                if d < min_d:
                    if d < 1e-6:
                        dx, dy, d = 1.0, 0.0, 1.0
                    push = (min_d - d) / 2 * 0.6
                    ux, uy = dx / d, dy / d
                    p[0] -= ux * push
                    p[1] -= uy * push
                    q[0] += ux * push
                    q[1] += uy * push
                    moved = True
        for pt in points:
            pt[key] = list(clip(*pt[key]))
        if not moved:
            break


relax("a")
relax("b")

# Per-point stagger offsets (deterministic)
for pt in points:
    pt["off"] = rng.uniform(0, MAX_OFFSET)


def f(v):
    """Format a coordinate with one decimal, dropping a trailing .0."""
    s = f"{v:.1f}"
    return s[:-2] if s.endswith(".0") else s


def kt(t):
    """Convert a time in seconds to a SMIL keyTimes value."""
    s = f"{t / DUR:.4f}".rstrip("0").rstrip(".")
    return s or "0"


EASE = ".65 0 .35 1"
HOLD = "0 0 1 1"


def marker(sp, color=None, stroke=None):
    """Return the SVG shape of a species, centred on the origin."""
    fill = f' fill="{color}"' if color else ""
    if stroke:
        fill += f' stroke="{stroke}"'
    if sp == 0:
        return f'<circle r="6"{fill}/>'
    if sp == 1:
        s = 14.0
        h = s * math.sqrt(3) / 2
        pts = f"0,{f(-2 * h / 3)} {f(s / 2)},{f(h / 3)} {f(-s / 2)},{f(h / 3)}"
        return f'<polygon points="{pts}"{fill}/>'
    s = 10.6
    return f'<rect x="{f(-s / 2)}" y="{f(-s / 2)}" width="{s}" height="{s}" rx="1.5"{fill}/>'


def ellipse_for(pts, pad=16):
    """Return centre and radii of an ellipse around the points."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    cx = (max(xs) + min(xs)) / 2
    cy = (max(ys) + min(ys)) / 2
    rx = (max(xs) - min(xs)) / 2 + pad
    ry = (max(ys) - min(ys)) / 2 + pad
    return cx, cy, rx, ry


def opacity_anim(times_vals):
    """Return an opacity animation from (time, value) pairs."""
    ts = ";".join(kt(t) for t, _ in times_vals)
    vs = ";".join(str(v) for _, v in times_vals)
    return (
        f'<animate attributeName="opacity" dur="{DUR}s" repeatCount="indefinite" '
        f'keyTimes="{ts}" values="{vs}"/>'
    )


def build() -> str:
    """Return the SVG document as a string."""
    out = []
    o = out.append
    o(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
        f'role="img" aria-labelledby="t d">'
    )
    o('<title id="t">MaldiBatchKit batch-effect correction</title>')
    o(
        '<desc id="d">Animated scatter plot of 54 MALDI-TOF spectra from three acquisition sites '
        "(colours) and three species (marker shapes). Before correction the samples form three "
        "clusters, one per site. After MaldiBatchKit the site colours are mixed and the samples "
        "form three clusters, one per species: the site effect is removed and the species signal "
        "is preserved. The animation loops between the two states.</desc>"
    )
    o(
        "<style>"
        'text{font-family:system-ui,-apple-system,"Segoe UI",Helvetica,Arial,sans-serif}'
        ".tx{fill:#737b85}.mu{fill:#7d8590}.ax{stroke:#7d8590}.nt{fill:#737b85}"
        ".hb{fill:#818b98;stroke:#818b98}"
        ".h1{font-size:16px;font-weight:600}.h2{font-size:15px}"
        ".lg{font-size:13px}.lh{font-size:11px;font-weight:600;letter-spacing:.08em}"
        ".al{font-size:11px}"
        "@media (prefers-color-scheme:dark){"
        ".tx{fill:#7d8590}.mu{fill:#848d97}.nt{fill:#7d8590}.hb{fill:#8b949e;stroke:#8b949e}}"
        "</style>"
    )

    o(
        f'<g class="ax" stroke-width="1" stroke-opacity=".5" fill="none" stroke-linecap="round">'
        f'<path d="M{PX0} {PY0}V{PY1}H{PX1}"/></g>'
    )
    o(f'<text class="mu al" x="{PX1}" y="{PY1 + 16}" text-anchor="end">UMAP 1</text>')
    o(
        f'<text class="mu al" transform="translate({PX0 - 10} {PY0}) rotate(-90)" '
        f'text-anchor="end">UMAP 2</text>'
    )

    mean_off = MAX_OFFSET / 2
    t_ab0 = T_HOLD_A + mean_off  # forward move starts (avg)
    t_ab1 = t_ab0 + T_MOVE_AB  # forward move ends
    t_ba0 = t_ab1 + T_HOLD_B  # back move starts
    t_ba1 = t_ba0 + T_MOVE_BA  # back move ends

    # State A: soft coloured halo per site.
    site_hulls = []
    for s in range(3):
        cx, cy, rx, ry = ellipse_for([p["a"] for p in points if p["site"] == s])
        site_hulls.append(
            f'<ellipse cx="{f(cx)}" cy="{f(cy)}" rx="{f(rx)}" ry="{f(ry)}" '
            f'fill="{SITE_COLORS[s]}"/>'
        )
    o(
        '<g fill-opacity=".13">'
        + "".join(site_hulls)
        + opacity_anim(
            [
                (0, 1),
                (t_ab0, 1),
                (t_ab0 + 0.8, 0),
                (t_ba0 + 0.6, 0),
                (t_ba1, 1),
                (DUR, 1),
            ]
        )
        + "</g>"
    )

    # State B: neutral dashed outline per species.
    sp_hulls = []
    for sp in range(3):
        cx, cy, rx, ry = ellipse_for([p["b"] for p in points if p["sp"] == sp], pad=17)
        sp_hulls.append(
            f'<ellipse cx="{f(cx)}" cy="{f(cy)}" rx="{f(rx)}" ry="{f(ry)}"/>'
        )
    o(
        '<g class="hb" fill-opacity=".07" stroke-opacity=".55" stroke-width="1" '
        'stroke-dasharray="3 4" opacity="0">'
        + "".join(sp_hulls)
        + opacity_anim(
            [
                (0, 0),
                (t_ab1 - 0.6, 0),
                (t_ab1 + 0.1, 1),
                (t_ba0, 1),
                (t_ba0 + 0.5, 0),
                (DUR, 0),
            ]
        )
        + "</g>"
    )

    o('<g stroke-width="1" stroke-linejoin="round" fill-opacity=".92">')
    for pt in points:
        off = pt["off"]
        a = f"{f(pt['a'][0])} {f(pt['a'][1])}"
        b = f"{f(pt['b'][0])} {f(pt['b'][1])}"
        t1 = T_HOLD_A + off
        t2 = t1 + T_MOVE_AB
        t3 = t2 + T_HOLD_B
        t4 = t3 + T_MOVE_BA
        assert t4 < DUR - 1e-3
        keytimes = ";".join(kt(t) for t in (0, t1, t2, t3, t4, DUR))
        o(
            f'<g transform="translate({a})">{marker(pt["sp"], SITE_COLORS[pt["site"]], SITE_STROKES[pt["site"]])}'
            f'<animateTransform attributeName="transform" type="translate" dur="{DUR}s" '
            f'repeatCount="indefinite" calcMode="spline" keyTimes="{keytimes}" '
            f'values="{a};{a};{b};{b};{a};{a}" '
            f'keySplines="{HOLD};{EASE};{HOLD};{EASE};{HOLD}"/></g>'
        )
    o("</g>")

    cap_y = 30
    o(
        f'<g opacity="1"><text x="{PX0}" y="{cap_y}">'
        f'<tspan class="tx h1">Before correction</tspan>'
        f'<tspan class="mu h2" dx="8">samples cluster by acquisition site</tspan></text>'
        + opacity_anim(
            [
                (0, 1),
                (t_ab0 + 0.35, 1),
                (t_ab0 + 0.7, 0),
                (t_ba0 + 0.6, 0),
                (t_ba0 + 0.95, 1),
                (DUR, 1),
            ]
        )
        + "</g>"
    )
    o(
        f'<g opacity="0"><text x="{PX0}" y="{cap_y}">'
        f'<tspan class="tx h1">After MaldiBatchKit</tspan>'
        f'<tspan class="mu h2" dx="8">site effect removed, species preserved</tspan></text>'
        + opacity_anim(
            [
                (0, 0),
                (t_ab0 + 0.75, 0),
                (t_ab0 + 1.15, 1),
                (t_ba0 + 0.25, 1),
                (t_ba0 + 0.55, 0),
                (DUR, 0),
            ]
        )
        + "</g>"
    )

    LX = 600
    o('<g class="lg">')
    o(f'<text class="mu lh" x="{LX}" y="74">ACQUISITION SITE</text>')
    for i, (name, col) in enumerate(zip(SITE_NAMES, SITE_COLORS, strict=True)):
        y = 100 + i * 24
        o(
            f'<circle cx="{LX + 7}" cy="{y - 4.5}" r="6" fill="{col}"/>'
            f'<text class="tx" x="{LX + 22}" y="{y}">{name}</text>'
        )
    o(f'<text class="mu lh" x="{LX}" y="186">SPECIES</text>')
    for i, name in enumerate(SPECIES_NAMES):
        y = 212 + i * 24
        o(
            f'<g class="nt" transform="translate({LX + 7} {y - 4.5})">'
            f"{marker(i)}</g>"
            f'<text class="tx" x="{LX + 22}" y="{y}">{name}</text>'
        )
    o("</g>")

    o("</svg>")
    return "\n".join(out) + "\n"


def main() -> None:
    """Write the SVG to the path given on the command line, or to DEFAULT_OUT."""
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    svg = build()
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(svg)
    print(f"Wrote {out} ({len(svg.encode('utf-8'))} bytes)")


if __name__ == "__main__":
    main()
