"""Render a hook-grasp rollout recorded by ``play.py`` (``<tag>_rollout.npz``) as a 3D matplotlib animation.

No Isaac Sim / RTX needed: only numpy, matplotlib and imageio (+ imageio-ffmpeg for mp4; falls back to GIF).
It runs inside the Isaac Lab container (called by ``play.py --video_type traj``) or anywhere else:

    python scripts/hook_grasp/render_rollout.py videos/final_rollout.npz [--fps 30 --every 2 --view follow --png]

Left: 3D view (basket walls, target bar, drone arms, wire, hook coloured by attached state, hook trail).
Right: hook-to-bar distance, attached flag and action norm, with a cursor at the current time.
"""

import argparse
import os

import numpy as np

HOOK_FREE = "#7f7f7f"
HOOK_ATTACHED = "#2ca02c"
BAR_COLOR = "#d62728"
DRONE_COLOR = "#1f77b4"
WIRE_COLOR = "#444444"
WALL_COLOR = "#c8a165"


def quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    """(…, 4) XYZW quaternion (Isaac Lab 3.0 convention) -> (…, 3, 3) rotation matrix."""
    x, y, z, w = np.moveaxis(q / np.linalg.norm(q, axis=-1, keepdims=True), -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def wall_polygons(basket_pos, basket_quat, hx, hy, H):
    """The 4 vertical wall panels of the basket as (4, 4, 3) world-frame corner lists."""
    walls_bf = np.array([
        [[hx, -hy, 0], [hx, hy, 0], [hx, hy, H], [hx, -hy, H]],
        [[-hx, -hy, 0], [-hx, hy, 0], [-hx, hy, H], [-hx, -hy, H]],
        [[-hx, hy, 0], [hx, hy, 0], [hx, hy, H], [-hx, hy, H]],
        [[-hx, -hy, 0], [hx, -hy, 0], [hx, -hy, H], [-hx, -hy, H]],
    ])
    R = quat_xyzw_to_matrix(basket_quat)
    return walls_bf @ R.T + basket_pos


def hook_bar_distance(hook, bar_center, bar_axis, half_len):
    """Distance from the hook to the target bar segment, (T,)."""
    rel = hook - bar_center
    s = np.clip(np.sum(rel * bar_axis, axis=-1), -half_len, half_len)
    closest = bar_center + s[:, None] * bar_axis
    return np.linalg.norm(hook - closest, axis=-1)


def _open_writer(out_path, fps):
    import imageio.v2 as imageio

    if out_path.endswith(".mp4"):
        try:
            return imageio.get_writer(out_path, fps=fps, codec="libx264", quality=7, macro_block_size=1), out_path
        except Exception as exc:  # no ffmpeg backend (imageio-ffmpeg missing)
            print(f"[render] mp4 writer unavailable ({exc}); falling back to GIF")
            out_path = out_path[:-4] + ".gif"
    return imageio.get_writer(out_path, mode="I", duration=1.0 / fps, loop=0), out_path


def render(npz_path, out_path=None, fps=30, every=2, view="follow", trail=80, png=False, dpi=100):
    """Render ``npz_path`` to ``out_path`` (default: ``<npz without _rollout>_traj.mp4``). Returns the path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    d = np.load(npz_path, allow_pickle=False)
    T = d["drone_pos"].shape[0]
    dt = float(d["dt"])
    t = np.arange(T) * dt
    hx, hy, H = float(d["basket_half_x"]), float(d["basket_half_y"]), float(d["basket_wall_height"])
    half_len = float(d["bar_half_len"])
    drone, rotors, wire, hook = d["drone_pos"], d["rotor_pos"], d["wire_pos"], d["hook_pos"]
    attached, done = d["attached"].astype(bool), d["done"].astype(bool)
    bar_c, bar_a = d["bar_center"], d["bar_axis"]
    dist = hook_bar_distance(hook, bar_c, bar_a, half_len)
    act_norm = np.linalg.norm(d["action"], axis=-1)
    caught_steps = np.flatnonzero(d["newly_caught"].astype(bool))

    if out_path is None:
        base = npz_path[:-4]
        base = base[: -len("_rollout")] if base.endswith("_rollout") else base
        out_path = base + "_traj.mp4"
    out_dir = os.path.dirname(out_path) or "."

    # fixed view: bounding box of everything that moves, plus the basket
    pts = np.concatenate([drone, hook, wire.reshape(-1, 3), d["basket_pos"]], axis=0)
    lo, hi = pts.min(0) - 0.3, pts.max(0) + 0.3
    fixed_c, fixed_r = (lo + hi) / 2, (hi - lo).max() / 2

    fig = plt.figure(figsize=(12.8, 7.2), dpi=dpi)
    gs = fig.add_gridspec(3, 3, width_ratios=[1.6, 1, 0.05], hspace=0.45, wspace=0.25,
                          left=0.02, right=0.97, top=0.93, bottom=0.08)
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_d, ax_a, ax_u = (fig.add_subplot(gs[i, 1]) for i in range(3))

    # static time series with a moving cursor
    ax_d.plot(t, dist, color=BAR_COLOR, lw=1.2)
    ax_d.set_ylabel("hook-bar [m]")
    ax_d.set_ylim(0, min(dist.max() * 1.05 + 1e-3, 2.0))
    ax_a.fill_between(t, 0, attached.astype(float), color=HOOK_ATTACHED, alpha=0.5, step="post")
    ax_a.set_ylim(-0.05, 1.05)
    ax_a.set_yticks([0, 1], ["free", "attached"])
    ax_u.plot(t, act_norm, color=DRONE_COLOR, lw=1.0)
    ax_u.set_ylabel("|action|")
    ax_u.set_xlabel("time [s]")
    for ax in (ax_d, ax_a, ax_u):
        ax.set_xlim(0, t[-1])
        ax.grid(alpha=0.3)
        for k in caught_steps:
            ax.axvline(t[k], color=HOOK_ATTACHED, lw=0.8, ls="--")
        for k in np.flatnonzero(done):
            ax.axvline(t[k], color="k", lw=0.8, ls=":")
    cursors = [ax.axvline(0, color="k", lw=1.0) for ax in (ax_d, ax_a, ax_u)]

    writer, out_path = _open_writer(out_path, fps)
    png_steps = set()
    if png:
        png_steps = {int(k) for k in np.linspace(0, T - 1, 5)} | {int(k) for k in caught_steps}
        png_steps = {k - k % every for k in png_steps}
    title = os.path.basename(str(d["checkpoint"])) if "checkpoint" in d else ""

    # episode start index of every step (the hook trail must not jump across resets). done[k] means the
    # episode ended at step k and the state recorded at k is already the reset one.
    ep_start = np.zeros(T, dtype=int)
    for k in range(1, T):
        ep_start[k] = k if done[k] else ep_start[k - 1]

    n_frames = 0
    for k in range(0, T, every):
        ax3d.cla()
        walls = wall_polygons(d["basket_pos"][k], d["basket_quat"][k], hx, hy, H)
        ax3d.add_collection3d(Poly3DCollection(walls, facecolor=WALL_COLOR, edgecolor="#8a6d3b", alpha=0.25))
        bar = np.stack([bar_c[k] - half_len * bar_a[k], bar_c[k] + half_len * bar_a[k]])
        ax3d.plot(*bar.T, color=BAR_COLOR, lw=4)
        for r in rotors[k]:
            ax3d.plot(*np.stack([drone[k], r]).T, color=DRONE_COLOR, lw=2.5)
        ax3d.scatter(*rotors[k].T, color=DRONE_COLOR, s=25)
        chain = np.concatenate([drone[k][None], wire[k], hook[k][None]], axis=0)
        ax3d.plot(*chain.T, color=WIRE_COLOR, lw=1.2)
        s0 = max(ep_start[k], k - trail)
        ax3d.plot(*hook[s0 : k + 1].T, color=HOOK_ATTACHED if attached[k] else HOOK_FREE, lw=0.8, alpha=0.6)
        ax3d.scatter(*hook[k], color=HOOK_ATTACHED if attached[k] else HOOK_FREE, s=60, depthshade=False)
        ax3d.plot(*np.stack([drone[k], drone[k] * [1, 1, 0]]).T, color="#bbbbbb", lw=0.6, ls=":")  # height line

        if view == "follow":
            c, r = (drone[k] + bar_c[k]) / 2, max(0.9, 0.6 * np.linalg.norm(drone[k] - bar_c[k]) + 0.4)
        else:
            c, r = fixed_c, fixed_r
        ax3d.set_xlim(c[0] - r, c[0] + r)
        ax3d.set_ylim(c[1] - r, c[1] + r)
        ax3d.set_zlim(max(0.0, c[2] - r), max(0.0, c[2] - r) + 2 * r)
        ax3d.set_box_aspect((1, 1, 1))
        ax3d.set_xlabel("x [m]")
        ax3d.set_ylabel("y [m]")
        ax3d.set_zlabel("z [m]")
        ax3d.set_title(f"t = {t[k]:5.2f} s   {'ATTACHED' if attached[k] else 'free'}", fontsize=11,
                       color=HOOK_ATTACHED if attached[k] else "k")
        for cur in cursors:
            cur.set_xdata([t[k], t[k]])
        fig.suptitle(f"hook-grasp rollout  {title}", fontsize=12)

        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        writer.append_data(frame)
        n_frames += 1
        if k in png_steps:
            fig.savefig(os.path.join(out_dir, f"{os.path.basename(out_path).rsplit('.', 1)[0]}_t{t[k]:05.2f}.png"))
    writer.close()
    plt.close(fig)
    print(f"[render] wrote {out_path} ({n_frames} frames, {fps} fps)")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("npz", help="<tag>_rollout.npz written by play.py")
    parser.add_argument("--out", default=None, help="Output .mp4 (or .gif). Default: <tag>_traj.mp4 next to the npz.")
    parser.add_argument("--fps", type=int, default=30, help="Video fps.")
    parser.add_argument("--every", type=int, default=2, help="Render one frame every N control steps (dt = 0.01 s).")
    parser.add_argument("--view", choices=["follow", "fixed"], default="follow", help="Camera framing.")
    parser.add_argument("--trail", type=int, default=80, help="Hook trail length in steps.")
    parser.add_argument("--png", action="store_true", help="Also save a few key frames (start, catches, end) as PNG.")
    args = parser.parse_args()
    render(args.npz, args.out, fps=args.fps, every=args.every, view=args.view, trail=args.trail, png=args.png)


if __name__ == "__main__":
    main()
