"""
Render "SREE" as an EXTRUDED 3D wordmark rasterized to ASCII,
and emit it as an SVG that animates on GitHub.

SMIL only -- GitHub runs SVG animations in <img>, but never JS.

Based on the original AVIVASHISHTA29 implementation.
"""

import argparse
import html
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont


HERE = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# GEOMETRY / GRID
# ============================================================

# More columns = more horizontal detail
COLS = int(os.environ.get("WORDMARK_COLS", 80))

# ROWS is calculated automatically by fit()
ROWS = 0

# Extra vertical margin
ROW_MARGIN = int(
    os.environ.get("WORDMARK_ROW_MARGIN", 5)
)

# Smaller cells = smaller ASCII characters
CELL_W = 6.0
CELL_H = 10.0


# ============================================================
# WINDOWS FONT
# ============================================================

FONT_PATH = os.environ.get(
    "WORDMARK_FONT",
    r"C:\Windows\Fonts\arialbd.ttf"
)

# Arial Bold is a normal .ttf file
FONT_INDEX = int(
    os.environ.get(
        "WORDMARK_FONT_INDEX",
        0
    )
)


# ============================================================
# WORDMARK TEXT
# ============================================================

TEXT = os.environ.get(
    "WORDMARK_TEXT",
    "SREE"
)


# ============================================================
# SOURCE MASK / 3D SETTINGS
# ============================================================

# Higher resolution source mask
MASK_H = 400

# Letter spacing
TRACKING = 0.14

# Line spacing
LINE_GAP = 1.20

# Extrusion depth
DEPTH_FRAC = 0.34


# ============================================================
# CAMERA / PERSPECTIVE
# ============================================================

TILT_DEG = float(
    os.environ.get(
        "WORDMARK_TILT",
        4.0
    )
)

CAM_DIST = 6.0
FOCAL = 4.15

# Controls how much of the available grid is occupied
FIT = 0.92


# ============================================================
# ASCII CHARACTER RAMP
# ============================================================

# sparse/dim -> dense/bright
RAMP = " .':-=+*csS%@"


# ============================================================
# LIGHTING
# ============================================================

LIGHT = np.array(
    [-0.15, -0.45, -1.00]
)

LIGHT = LIGHT / np.linalg.norm(LIGHT)

AMBIENT = 0.22
FOG = 0.34
FOG_SPAN = 0.55


# ============================================================
# SVG PALETTE
# ============================================================

BG = "#0d1117"
BG2 = "#111722"
FRAME = "#30363d"
TITLE_TEXT = "#7d8590"
INK = "#c9d1d9"

PAD = 18
TITLEBAR_H = 28


# ============================================================
# BUILD 3D VOXEL SHELL
# ============================================================

def build_shell():
    """
    Rasterize TEXT and return:

        points Nx3
        normals Nx3
    """

    probe = TEXT.replace(
        "\n",
        ""
    )

    font_size = MASK_H

    # Find a font size that fits within MASK_H
    for _ in range(40):

        font = ImageFont.truetype(
            FONT_PATH,
            font_size,
            index=FONT_INDEX
        )

        l, t, r, b = font.getbbox(
            probe
        )

        if b - t <= MASK_H:
            break

        font_size = int(
            font_size * 0.92
        )

    h = b - t

    track = int(
        round(
            TRACKING * font_size
        )
    )

    lines = TEXT.split(
        "\n"
    )

    line_h = int(
        round(
            h * LINE_GAP
        )
    )

    def line_w(s):

        return (
            sum(
                font.getlength(c)
                for c in s
            )
            + track * (
                len(s) - 1
            )
        )

    total_w = int(
        round(
            max(
                line_w(s)
                for s in lines
            )
        )
    ) + 8

    total_h = (
        line_h * (
            len(lines) - 1
        )
        + h
        + 8
    )

    # Create grayscale mask
    img = Image.new(
        "L",
        (
            total_w,
            total_h
        ),
        0
    )

    d = ImageDraw.Draw(
        img
    )

    # Draw each letter
    for li, s in enumerate(lines):

        pen = (
            4.0
            + (
                total_w
                - 8
                - line_w(s)
            ) / 2.0
        )

        base = (
            -t
            + 4
            + li * line_h
        )

        for ch in s:

            d.text(
                (pen, base),
                ch,
                font=font,
                fill=255
            )

            pen += (
                font.getlength(ch)
                + track
            )

    # Convert to binary mask
    mask = np.array(
        img
    ) > 127

    # Find actual occupied region
    xs_any = np.nonzero(
        mask.any(0)
    )[0]

    ys_any = np.nonzero(
        mask.any(1)
    )[0]

    mask = mask[
        ys_any[0]:ys_any[-1] + 1,
        xs_any[0]:xs_any[-1] + 1
    ]

    H, W = mask.shape

    # Calculate extrusion depth
    depth = max(
        4,
        int(
            round(
                H * DEPTH_FRAC
            )
        )
    )

    cy, cx = np.nonzero(
        mask
    )

    pts = []
    nrm = []


    # ========================================================
    # FRONT CAP
    # ========================================================

    front = np.stack(
        [
            cx,
            cy,
            np.full_like(
                cx,
                -0.6,
                dtype=float
            )
        ],
        1
    )

    pts.append(
        front
    )

    nrm.append(
        np.tile(
            [0.0, 0.0, -1.0],
            (
                len(front),
                1
            )
        )
    )


    # ========================================================
    # BACK CAP
    # ========================================================

    back = np.stack(
        [
            cx,
            cy,
            np.full_like(
                cx,
                depth
            )
        ],
        1
    ).astype(
        float
    )

    pts.append(
        back
    )

    nrm.append(
        np.tile(
            [0.0, 0.0, 1.0],
            (
                len(back),
                1
            )
        )
    )


    # ========================================================
    # SIDE WALLS
    # ========================================================

    pad = np.pad(
        mask,
        1
    )

    empty_r = ~pad[
        1:-1,
        2:
    ]

    empty_l = ~pad[
        1:-1,
        :-2
    ]

    empty_d = ~pad[
        2:,
        1:-1
    ]

    empty_u = ~pad[
        :-2,
        1:-1
    ]

    edge = mask & (
        empty_r
        | empty_l
        | empty_d
        | empty_u
    )

    ey, ex = np.nonzero(
        edge
    )

    nx = (
        empty_r[
            ey,
            ex
        ].astype(float)
        -
        empty_l[
            ey,
            ex
        ].astype(float)
    )

    ny = (
        empty_d[
            ey,
            ex
        ].astype(float)
        -
        empty_u[
            ey,
            ex
        ].astype(float)
    )

    ln = np.sqrt(
        nx * nx
        + ny * ny
    )

    ln[
        ln == 0
    ] = 1.0

    nx = nx / ln
    ny = ny / ln

    zsteps = np.linspace(
        0,
        depth,
        max(
            3,
            depth // 2
        )
    )

    for z in zsteps:

        pts.append(
            np.stack(
                [
                    ex,
                    ey,
                    np.full_like(
                        ex,
                        z,
                        dtype=float
                    )
                ],
                1
            )
        )

        nrm.append(
            np.stack(
                [
                    nx,
                    ny,
                    np.zeros_like(nx)
                ],
                1
            )
        )


    # ========================================================
    # COMBINE POINTS / NORMALS
    # ========================================================

    P = np.concatenate(
        pts
    ).astype(
        np.float32
    )

    N = np.concatenate(
        nrm
    ).astype(
        np.float32
    )

    # Center
    P[:, 0] -= W / 2.0
    P[:, 1] -= H / 2.0
    P[:, 2] -= depth / 2.0

    # Normalize
    P /= float(W)

    return P, N


# ============================================================
# ROTATION
# ============================================================

def rot_y(a):

    c, s = (
        math.cos(a),
        math.sin(a)
    )

    return np.array(
        [
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ],
        np.float32
    )


def rot_x(a):

    c, s = (
        math.cos(a),
        math.sin(a)
    )

    return np.array(
        [
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ],
        np.float32
    )


# ============================================================
# PROJECT 3D OBJECT
# ============================================================

def project(
    P,
    N,
    yaw
):

    """
    Rotate + perspective divide.
    """

    M = (
        rot_x(
            math.radians(
                TILT_DEG
            )
        )
        @ rot_y(yaw)
    )

    p = P @ M.T
    n = N @ M.T

    # Back-face culling
    vis = n[:, 2] < 0.0

    p = p[vis]
    n = n[vis]

    z = (
        p[:, 2]
        + CAM_DIST
    )

    f = (
        FOCAL / z
    )

    lam = n @ LIGHT

    inten = (
        AMBIENT
        + (
            1 - AMBIENT
        )
        * np.clip(
            lam,
            0,
            1
        )
    )

    # Depth fog
    t = np.clip(
        (
            z - CAM_DIST
        )
        / FOG_SPAN,
        -1.0,
        1.0
    )

    inten *= (
        1.0
        - FOG
        * (
            t + 1.0
        )
        / 2.0
    )

    idx = np.clip(
        (
            inten
            * (
                len(RAMP) - 1
            )
        )
        .round()
        .astype(int),
        1,
        len(RAMP) - 1
    )

    return (
        p[:, 0] * f,
        p[:, 1] * f,
        z,
        idx
    )


# ============================================================
# FIT TO GRID
# ============================================================

def fit(
    projected
):

    """
    Width-driven scale + offset.

    ROWS is calculated automatically from the
    projected wordmark dimensions.
    """

    global ROWS

    xs = np.concatenate(
        [
            q[0]
            for q in projected
        ]
    )

    ys = np.concatenate(
        [
            q[1]
            for q in projected
        ]
    )

    x0, x1 = (
        xs.min(),
        xs.max()
    )

    y0, y1 = (
        ys.min(),
        ys.max()
    )

    ar = (
        CELL_W / CELL_H
    )

    scale = (
        FIT
        * (COLS - 1)
        / (x1 - x0)
    )

    ROWS = int(
        math.ceil(
            (
                y1 - y0
            )
            * ar
            * scale
        )
    ) + 1 + (
        2 * ROW_MARGIN
    )

    cx = (
        (COLS - 1) / 2.0
        -
        (
            x0 + x1
        )
        / 2.0
        * scale
    )

    cy = (
        (ROWS - 1) / 2.0
        -
        (
            y0 + y1
        )
        / 2.0
        * scale
        * ar
    )

    return (
        scale,
        cx,
        cy
    )


# ============================================================
# RASTERIZE
# ============================================================

def rasterize(
    q,
    scale,
    cx,
    cy
):

    """
    Z-buffer splat one projected frame.
    """

    x, y, z, idx = q

    col = np.round(
        cx
        + x * scale
    ).astype(int)

    row = np.round(
        cy
        + y
        * scale
        * (
            CELL_W
            / CELL_H
        )
    ).astype(int)

    ok = (
        (col >= 0)
        & (col < COLS)
        & (row >= 0)
        & (row < ROWS)
    )

    col = col[ok]
    row = row[ok]
    z = z[ok]
    idx = idx[ok]

    grid = np.zeros(
        (
            ROWS,
            COLS
        ),
        np.int8
    )

    order = np.argsort(
        -z
    )

    grid[
        row[order],
        col[order]
    ] = idx[order]

    return [
        "".join(
            RAMP[i]
            for i in r
        )
        for r in grid
    ]


# ============================================================
# SVG OUTPUT
# ============================================================

def emit(
    frames,
    mode,
    out,
    dur,
    reveal
):

    art_w = (
        COLS * CELL_W
    )

    art_h = (
        ROWS * CELL_H
    )

    canvas_w = (
        art_w
        + PAD * 2
    )

    canvas_h = (
        TITLEBAR_H
        + art_h
        + PAD
    )

    art_top = (
        TITLEBAR_H
        + PAD * 0.3
    )

    # Font size follows CELL_H
    fs = (
        CELL_H * 0.92
    )

    n = len(
        frames
    )

    p = [

        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{canvas_w:.0f}" '
        f'height="{canvas_h:.0f}" '
        f'viewBox="0 0 {canvas_w:.0f} '
        f'{canvas_h:.0f}" '
        f'font-family="ui-monospace, '
        f'SFMono-Regular, Menlo, Consolas, monospace">',

        '<defs>'
        '<linearGradient id="wbg" '
        'x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0" '
        f'stop-color="{BG2}"/>'
        f'<stop offset="1" '
        f'stop-color="{BG}"/>'
        '</linearGradient>'
        '</defs>',

        f'<rect width="{canvas_w:.0f}" '
        f'height="{canvas_h:.0f}" '
        f'rx="12" '
        f'fill="url(#wbg)"/>',

        f'<rect x="0.5" y="0.5" '
        f'width="{canvas_w-1:.0f}" '
        f'height="{canvas_h-1:.0f}" '
        f'rx="12" '
        f'fill="none" '
        f'stroke="{FRAME}" '
        f'stroke-width="1"/>',

        f'<line x1="0" '
        f'y1="{TITLEBAR_H}" '
        f'x2="{canvas_w:.0f}" '
        f'y2="{TITLEBAR_H}" '
        f'stroke="{FRAME}"/>',
    ]


    # ========================================================
    # TERMINAL DOTS
    # ========================================================

    for i, dot in enumerate(
        [
            "#ff5f56",
            "#ffbd2e",
            "#27c93f"
        ]
    ):

        p.append(
            f'<circle '
            f'cx="{PAD + i*15}" '
            f'cy="{TITLEBAR_H/2}" '
            f'r="4.5" '
            f'fill="{dot}"/>'
        )


    # ========================================================
    # TERMINAL TITLE
    # ========================================================

    p.append(
        f'<text '
        f'x="{canvas_w/2:.0f}" '
        f'y="{TITLEBAR_H/2 + 4:.0f}" '
        f'fill="{TITLE_TEXT}" '
        f'font-size="11.5" '
        f'text-anchor="middle">'
        f'sreenikesh@github: ~$ '
        f'./wordmark.sh --3d'
        f'</text>'
    )


    # ========================================================
    # FRAME GENERATOR
    # ========================================================

    def frame_g(
        rows,
        extra=""
    ):

        out_rows = []

        for ry, line in enumerate(
            rows
        ):

            s = line.rstrip()

            if not s.strip():
                continue

            lead = (
                len(s)
                - len(
                    s.lstrip(" ")
                )
            )

            body = s[
                lead:
            ]

            x = (
                PAD
                + lead * CELL_W
            )

            y = (
                art_top
                + ry * CELL_H
                + CELL_H * 0.78
            )

            out_rows.append(
                f'<text '
                f'xml:space="preserve" '
                f'x="{x:.1f}" '
                f'y="{y:.1f}" '
                f'font-size="{fs:.1f}" '
                f'textLength="'
                f'{len(body)*CELL_W:.1f}" '
                f'lengthAdjust="spacing">'
                f'{html.escape(body)}'
                f'</text>'
            )

        return (
            f'<g fill="{INK}"{extra}>'
            + "".join(
                out_rows
            )
            + "</g>"
        )


    # ========================================================
    # STATIC MODE
    # ========================================================

    if mode == "static":

        p.append(
            frame_g(
                frames[0]
            )
        )

        p.append(
            "</svg>"
        )

        with open(
            out,
            "w"
        ) as fh:

            fh.write(
                "".join(p)
            )

        print(
            "wrote",
            out
        )

        return


    # ========================================================
    # INTRO WIPE
    # ========================================================

    p.append(
        f'<clipPath id="wipe">'
        f'<rect x="{PAD}" '
        f'y="{art_top:.1f}" '
        f'height="{art_h:.1f}" '
        f'width="0">'
        f'<animate '
        f'attributeName="width" '
        f'from="0" '
        f'to="{art_w:.0f}" '
        f'begin="0s" '
        f'dur="{reveal:.2f}s" '
        f'fill="freeze"/>'
        f'</rect>'
        f'</clipPath>'
    )


    p.append(
        f'<g clip-path="url(#wipe)">'
        f'{frame_g(frames[0])}'
        f'<set '
        f'attributeName="opacity" '
        f'to="0" '
        f'begin="{reveal:.2f}s"/>'
        f'</g>'
    )


    # ========================================================
    # SCANNING CURSOR
    # ========================================================

    p.append(
        f'<rect x="{PAD}" '
        f'y="{art_top+2:.1f}" '
        f'width="{CELL_W*1.6:.1f}" '
        f'height="{art_h-4:.1f}" '
        f'fill="{INK}" '
        f'opacity="0.16">'
        f'<animate '
        f'attributeName="x" '
        f'from="{PAD}" '
        f'to="{PAD+art_w:.0f}" '
        f'begin="0s" '
        f'dur="{reveal:.2f}s" '
        f'fill="freeze"/>'
        f'<set '
        f'attributeName="opacity" '
        f'to="0" '
        f'begin="{reveal:.2f}s"/>'
        f'</rect>'
    )


    # ========================================================
    # ONCE MODE
    # ========================================================

    if mode == "once":

        step = (
            dur / n
        )

        for i, rows in enumerate(
            frames
        ):

            begin = (
                reveal
                + i * step
            )

            sets = (
                f'<set '
                f'attributeName="opacity" '
                f'to="1" '
                f'begin="{begin:.3f}s"/>'
            )

            if i != n - 1:

                sets += (
                    f'<set '
                    f'attributeName="opacity" '
                    f'to="0" '
                    f'begin="'
                    f'{begin+step:.3f}s"/>'
                )

            p.append(
                frame_g(
                    rows,
                    ' opacity="0"'
                ).replace(
                    "</g>",
                    sets
                    + "</g>"
                )
            )


    # ========================================================
    # LOOPING MODES
    # ========================================================

    else:

        for i, rows in enumerate(
            frames
        ):

            if i == 0:

                vals = "1;0"

                kt = (
                    f"0;{1/n:.5f}"
                )

            else:

                vals = "0;1;0"

                kt = (
                    f"0;{i/n:.5f};"
                    f"{(i+1)/n:.5f}"
                )

            anim = (
                f'<animate '
                f'attributeName="opacity" '
                f'calcMode="discrete" '
                f'values="{vals}" '
                f'keyTimes="{kt}" '
                f'dur="{dur:.2f}s" '
                f'begin="{reveal:.2f}s" '
                f'repeatCount="indefinite"/>'
            )

            p.append(
                frame_g(
                    rows,
                    ' opacity="0"'
                ).replace(
                    "</g>",
                    anim
                    + "</g>"
                )
            )


    # ========================================================
    # WRITE SVG
    # ========================================================

    p.append(
        "</svg>"
    )

    svg = "".join(
        p
    )

    with open(
        out,
        "w"
    ) as fh:

        fh.write(
            svg
        )

    print(
        f"wrote {out} "
        f"{len(svg)/1024:.1f} KB "
        f"{n} frames "
        f"{canvas_w:.0f}x{canvas_h:.0f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--mode",
        choices=[
            "spin",
            "once",
            "rock",
            "static"
        ],
        default="rock"
    )

    ap.add_argument(
        "--out",
        default=None
    )

    ap.add_argument(
        "--frames",
        type=int,
        default=None
    )

    ap.add_argument(
        "--dur",
        type=float,
        default=None
    )

    ap.add_argument(
        "--reveal",
        type=float,
        default=1.6
    )

    ap.add_argument(
        "--preview",
        action="store_true",
        help="print frames to stdout instead"
    )

    a = ap.parse_args()


    # Build 3D shell
    P, N = build_shell()


    # Rest pose
    rest = math.radians(
        -13
    )


    # ========================================================
    # SPIN
    # ========================================================

    if a.mode == "spin":

        nf = (
            a.frames
            or 36
        )

        yaws = [
            rest
            + 2 * math.pi * i / nf
            for i in range(nf)
        ]

        dur = (
            a.dur
            or 7.0
        )


    # ========================================================
    # ONCE
    # ========================================================

    elif a.mode == "once":

        nf = (
            a.frames
            or 32
        )

        yaws = [
            rest
            + 2 * math.pi * i / nf
            for i in range(nf)
        ] + [
            rest
        ]

        dur = (
            a.dur
            or 3.6
        )


    # ========================================================
    # ROCK
    # ========================================================

    else:

        nf = (
            a.frames
            or 20
        )

        amp = math.radians(
            11
        )

        yaws = [
            rest
            + amp
            * math.sin(
                2 * math.pi * i / nf
            )
            for i in range(nf)
        ]

        dur = (
            a.dur
            or 5.0
        )


    # ========================================================
    # PROJECT FRAMES
    # ========================================================

    proj = [
        project(
            P,
            N,
            y
        )
        for y in yaws
    ]


    # ========================================================
    # FIT TO GRID
    # ========================================================

    scale, cx, cy = fit(
        proj
    )


    # ========================================================
    # RASTERIZE FRAMES
    # ========================================================

    frames = [
        rasterize(
            q,
            scale,
            cx,
            cy
        )
        for q in proj
    ]


    # ========================================================
    # ASCII PREVIEW
    # ========================================================

    if a.preview:

        for row in frames[0]:

            print(
                row.rstrip()
            )

        return


    # ========================================================
    # OUTPUT PATH
    # ========================================================

    out = (
        a.out
        or os.path.join(
            HERE,
            "..",
            f"wordmark-{a.mode}.svg"
        )
    )


    # ========================================================
    # EMIT SVG
    # ========================================================

    emit(
        frames,
        a.mode,
        out,
        dur,
        a.reveal
    )


if __name__ == "__main__":
    main()