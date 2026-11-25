"""RELION-style plot of the per-frame B-factors and scales (needs matplotlib).

Top: B and scale per frame (``w_f(k) = exp(B_f k^2 / 4 + C_f)``, max B = 0 and max
scale = 1): the estimate with bootstrap SDs, Grant & Grigorieff's weights fitted the
same way, and the default curve. Bottom: each frame's measured weight at three
resolutions (points, three neighbouring bins pooled), the fitted model (lines) and
G&G (dashed), relative to the mean of frames 2-9.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"estimate": "#2a78d6", "gg": "#eb6834", "default": "#1baf7a"}
RES_COLORS = ["#2a78d6", "#eb6834", "#6250d6"]


# pylint: disable=too-many-locals,too-many-statements
def plot_frame_bfactors(
    signal_npz: str | Path, out_path: str | Path, title: str = ""
) -> Path:
    """Plot ``frame_signal.npz`` (written by ``estimate_frame_weights``).

    Parameters
    ----------
    signal_npz : str | Path
        The estimator's ``frame_signal.npz``.
    out_path : str | Path
        The image to write (format from the suffix).
    title : str
        Text after the figure's title.

    Returns
    -------
    Path
        `out_path`.
    """
    # pylint: disable=import-outside-toplevel
    try:
        import matplotlib
        import matplotlib.ticker

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as err:
        raise ImportError(
            "the B-factor plot needs matplotlib: pip install 'ripple[plot]'"
        ) from err

    d = np.load(signal_npz)
    bf, cf = d["bfactor"], d["log_scale"]
    nf = len(bf)
    frames = np.arange(nf)
    b_sd = d["bootstrap_bfactor"].std(0) if d["bootstrap_bfactor"].size else None
    s_sd = (
        np.exp(d["bootstrap_log_scale"]).std(0)
        if d["bootstrap_log_scale"].size
        else None
    )
    default_b = d["default_bfactor"] - d["default_bfactor"].max()
    default_c = d["default_log_scale"] - d["default_log_scale"].max()

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig = plt.figure(figsize=(11, 7.4), constrained_layout=True)
    top, bottom = fig.subfigures(2, 1)  # type: ignore[misc, unused-ignore]
    ax_b, ax_c = top.subplots(1, 2)
    ax_b.errorbar(
        frames,
        bf,
        yerr=b_sd,
        fmt="o",
        ms=3,
        lw=0.8,
        color=COLORS["estimate"],
        label="estimated (bootstrap SD)",
    )
    ax_b.plot(
        frames, d["gg_bfactor"], "--", lw=1.5, color=COLORS["gg"], label="G&G, same fit"
    )
    ax_b.plot(
        frames, default_b, "-", lw=1.2, color=COLORS["default"], label="default curve"
    )
    ax_c.errorbar(
        frames, np.exp(cf), yerr=s_sd, fmt="o", ms=3, lw=0.8, color=COLORS["estimate"]
    )
    ax_c.plot(frames, np.exp(d["gg_log_scale"]), "--", lw=1.5, color=COLORS["gg"])
    ax_c.plot(frames, np.exp(default_c), "-", lw=1.2, color=COLORS["default"])
    ax_b.set_xlabel("Frame")
    ax_b.set_ylabel("B-factor (Å²)")
    ax_b.set_title("Per-frame B-factor", loc="left", fontsize=10, color=INK)
    ax_c.set_xlabel("Frame")
    ax_c.set_ylabel("Scale factor exp(C)")
    ax_c.set_title("Per-frame scale factor", loc="left", fontsize=10, color=INK)
    ax_b.legend(frameon=False, fontsize=8, loc="lower left")
    for ax in (ax_b, ax_c):
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)

    ax_w, ax_e = bottom.subplots(1, 2)
    kb, use = d["frequency"], d["bins_used"]
    mean, amp, gg = d["mean"], d["amplitude"], d["gg_weights"]
    hi = 1 / kb[use[-1]]
    for target, col in zip((20.0, 8.0, max(4.5, hi * 1.15)), RES_COLORS, strict=True):
        i = int(np.argmin(np.abs(1 / kb[use] - target)))
        lo_i, hi_i = max(0, i - 1), i + 2
        sel = use[lo_i:hi_i]
        k2 = kb[sel] ** 2
        meas = mean[:, sel].sum(1)
        model = (
            amp[None, lo_i:hi_i] * np.exp(bf[:, None] * k2[None, :] / 4 + cf[:, None])
        ).sum(1)
        ggm = (gg[:, sel] * (mean[:, sel].mean(0) / gg[:, sel].mean(0))[None, :]).sum(1)
        norm = model[2:10].mean() if nf > 9 else model.mean()
        ax_w.plot(frames, meas / norm, "o", ms=2.5, color=col, alpha=0.6)
        ax_w.plot(
            frames,
            model / norm,
            "-",
            lw=1.8,
            color=col,
            label=f"{1 / kb[sel].mean():.1f} Å",
        )
        ax_w.plot(
            frames,
            ggm / (ggm[2:10].mean() if nf > 9 else ggm.mean()),
            "--",
            lw=1.2,
            color=col,
        )
    ax_w.axhline(0, color=MUTED, lw=0.6)
    ax_w.set_xlabel("Frame")
    ax_w.set_ylabel("Weight (frames 2-9 = 1)")
    ax_w.set_title("Measured weight per frame", loc="left", fontsize=10, color=INK)
    ax_w.legend(
        frameon=False,
        fontsize=8,
        title="points measured, line fit, dashed G&G",
        title_fontsize=7,
        loc="upper right",
    )
    ax_w.grid(True, color=GRID, lw=0.6)
    ax_w.set_axisbelow(True)

    if "envelope_frequency" in d and d["envelope_frequency"].size:
        ke, he = d["envelope_frequency"], d["envelope_relative"]
        ok = (ke > 1 / 60.0) & (ke < kb[use[-1]] * 1.3)
        ref = he[np.argmin(np.abs(1 / ke - 40.0))]
        ax_e.plot(1 / ke[ok], he[ok] / ref, "-", lw=1.8, color=COLORS["estimate"])
        ax_e.set_xscale("log")
        ticks = [
            t
            for t in (50, 30, 20, 10, 6, 4, 3, 2)
            if (1 / ke[ok]).min() <= t <= (1 / ke[ok]).max()
        ]
        ax_e.set_xticks(ticks)
        ax_e.set_xticklabels([str(t) for t in ticks])
        ax_e.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax_e.invert_xaxis()
        ax_e.axhline(1, color=MUTED, lw=0.6)
        ax_e.set_xlabel("Resolution (Å)")
        ax_e.set_ylabel("Envelope / search template's (40 Å = 1)")
        ax_e.set_title(
            "New template envelope (matched + correction)",
            loc="left",
            fontsize=10,
            color=INK,
        )
        ax_e.grid(True, color=GRID, lw=0.6)
    else:
        ax_e.set_visible(False)
    fig.suptitle(
        "Per-frame signal in the template correlation, RELION's model "
        f"w(f, k) = exp(B_f k²/4 + C_f){'; ' + title if title else ''}",
        fontsize=10,
        color=INK,
    )
    out = Path(out_path)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
