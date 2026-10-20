# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Roll out a trained hook-grasp checkpoint on the FINAL task and visualize it.

    python scripts/hook_grasp/play.py --headless --checkpoint logs/rsl_rl/hook_grasp/<run>/model_19999.pt [--video_type traj]

Always writes ``<out_dir>/<tag>_rollout.npz`` (per-step drone/rotor/wire/hook/basket/bar positions of env
``--env_id``, attached flag, actions, rewards), ``<out_dir>/<tag>_summary.txt`` and ``<out_dir>/<tag>_viewer.html``
(interactive 3D player, open it in a web browser; see ``export_viewer.py``). Then, depending on ``--video_type``:

* ``traj`` (default): 3D matplotlib animation of the npz -> ``<tag>_traj.mp4`` (``render_rollout.py``). Needs no
  RTX GPU, so it works on the Izar V100s. Can also be re-rendered anywhere with
  ``python scripts/hook_grasp/render_rollout.py <tag>_rollout.npz``.
* ``isaac``: the original Isaac Sim RTX follow-camera video -> ``<tag>_isaac.mp4``. Needs an RTX-capable GPU
  (GeForce/Quadro RTX, L40, A40...); on a V100/A100/H100 the frames come out black.
* ``both``: both of the above.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Video of a trained hook-grasp PPO policy.")
parser.add_argument("--task", type=str, default="Isaac-flycrane-hook-grasp-v0", help="Task id.")
parser.add_argument("--checkpoint", type=str, required=True, help="rsl_rl model_<iter>.pt checkpoint.")
parser.add_argument("--num_envs", type=int, default=1, help="Parallel environments (the camera follows env 0).")
parser.add_argument("--steps", type=int, default=900, help="Control steps to record.")
parser.add_argument("--frame_every", type=int, default=3, help="--video_type isaac: capture a frame every N steps.")
parser.add_argument("--fps", type=int, default=20, help="--video_type isaac: video fps.")
parser.add_argument("--seed", type=int, default=7, help="Seed.")
parser.add_argument("--out_dir", type=str, default="videos", help="Output directory.")
parser.add_argument("--tag", type=str, default="hook_grasp", help="Output file basename.")
parser.add_argument(
    "--video_type", choices=["traj", "isaac", "both"], default="traj",
    help="traj: matplotlib 3D animation from recorded states (no RTX needed); isaac: Isaac Sim RTX camera video"
    " (RTX GPU only, black on V100); both.",
)
parser.add_argument("--env_id", type=int, default=0, help="Env recorded in the npz (the Isaac camera follows env 0).")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
use_isaac_video = args_cli.video_type in ("isaac", "both")
use_traj_video = args_cli.video_type in ("traj", "both")
# offscreen RTX rendering (needed for env.render() under --headless); skipped for --video_type traj
args_cli.enable_cameras = use_isaac_video
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

import gymnasium as gym
import numpy as np
import torch

from rsl_rl.runners import OnPolicyRunner

import MARL_mav_carry_ext.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rsl_rl_env import HookGraspVecEnv  # noqa: E402

from MARL_mav_carry_ext.assets.flyhook import FLYHOOK_WIRE_BODY  # noqa: E402


class RolloutRecorder:
    """Per-step states of one env, in its env-local frame, for render_rollout.py."""

    def __init__(self, base, env_id: int):
        self.base, self.e = base, env_id
        wire_ids, wire_names = base.robot.find_bodies(FLYHOOK_WIRE_BODY)
        # chain order drone -> hook, by the link number at the end of the body name
        order = sorted(range(len(wire_ids)), key=lambda j: int("".join(c for c in wire_names[j] if c.isdigit()) or 0))
        self.wire_ids = [wire_ids[j] for j in order]
        self.rows = {k: [] for k in (
            "drone_pos", "drone_quat", "rotor_pos", "wire_pos", "hook_pos", "basket_pos", "basket_quat",
            "bar_center", "bar_axis", "attached", "newly_caught", "done", "action", "reward",
        )}

    def record(self, action, reward, done):
        b, e = self.base, self.e
        state = b.robot.data.body_com_state_w.torch[e]                      # (num_bodies, 13), world frame
        origin = b.scene.env_origins[e]

        def pos(ids):
            return (state[ids, :3] - origin).cpu().numpy()

        r = self.rows
        r["drone_pos"].append(pos(b._falcon_idx).reshape(3))
        r["drone_quat"].append(state[b._falcon_idx, 3:7].reshape(4).cpu().numpy())
        r["rotor_pos"].append(pos(b._falcon_rotor_idx).reshape(-1, 3))
        r["wire_pos"].append(pos(self.wire_ids).reshape(-1, 3))
        r["hook_pos"].append(pos(b._hook_idx).reshape(3))
        r["basket_pos"].append(b.basket_pos_w[e].cpu().numpy())
        r["basket_quat"].append(b.basket_quat_w[e].cpu().numpy())
        r["bar_center"].append(b.bar_center_w[e].cpu().numpy())
        r["bar_axis"].append(b.bar_axis_w[e].cpu().numpy())
        r["attached"].append(bool(b.attached[e].any()))
        r["newly_caught"].append(bool(b._newly_caught[e].any()))
        r["done"].append(bool(done[e]))
        r["action"].append(action[e].cpu().numpy())
        r["reward"].append(float(reward[e]))

    def save(self, path, checkpoint, seed):
        cfg = self.base.cfg
        np.savez_compressed(
            path,
            **{k: np.asarray(v) for k, v in self.rows.items()},
            dt=float(self.base.step_dt),
            basket_half_x=float(cfg.basket_half_x),
            basket_half_y=float(cfg.basket_half_y),
            basket_wall_height=float(cfg.basket_wall_height),
            bar_half_len=float(cfg.bar_half_len),
            quat_convention="xyzw",
            checkpoint=str(checkpoint),
            seed=int(seed),
        )


def main():
    os.makedirs(args_cli.out_dir, exist_ok=True)
    device = args_cli.device if args_cli.device is not None else "cuda:0"
    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    env_cfg.disable_curricula()
    env_cfg.hold_success_steps = 10_000_000   # keep holding until the episode times out (nicer to watch)

    # follow-camera on the drone body
    env_cfg.viewer.origin_type = "asset_body"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.body_name = "Falcon_base_link_inertia"
    env_cfg.viewer.eye = (2.2, 2.2, 0.6)
    env_cfg.viewer.lookat = (0.0, 0.0, -0.4)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if use_isaac_video else None)
    base = env.unwrapped
    vec_env = HookGraspVecEnv(base)

    agent_cfg: dict = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg["device"] = device
    runner = OnPolicyRunner(vec_env, agent_cfg, log_dir=None, device=device)
    print(f"[play] loading {args_cli.checkpoint}")
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=device)

    recorder = RolloutRecorder(base, args_cli.env_id)
    e = args_cli.env_id
    obs, _ = vec_env.get_observations()
    frames = []
    catches = detaches = attached_steps = 0
    for step in range(args_cli.steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, rewards, dones, _ = vec_env.step(actions)
            recorder.record(actions, rewards, dones)
        catches += int(base._newly_caught[e].any())
        detaches += int(base._just_detached[e].any())
        attached_steps += int(base.attached[e].any())
        if use_isaac_video and step % args_cli.frame_every == 0:
            frame = base.render()
            if frame is not None:
                frames.append(np.asarray(frame))

    summary = "\n".join([
        f"checkpoint     : {args_cli.checkpoint}",
        f"env_id / video : {e} / {args_cli.video_type}",
        f"steps          : {args_cli.steps}",
        f"catches        : {catches}",
        f"detaches       : {detaches}",
        f"attached steps : {attached_steps} / {args_cli.steps}",
    ])
    print("\n" + summary)
    with open(os.path.join(args_cli.out_dir, f"{args_cli.tag}_summary.txt"), "w") as f:
        f.write(summary + "\n")
    npz = os.path.join(args_cli.out_dir, f"{args_cli.tag}_rollout.npz")
    recorder.save(npz, args_cli.checkpoint, args_cli.seed)
    print(f"[play] wrote {npz}")
    try:  # interactive 3D viewer (numpy only, always produced)
        from export_viewer import export
        export(npz)
    except Exception as exc:
        print(f"[play] could not write the interactive viewer ({type(exc).__name__}: {exc})")

    if use_isaac_video:
        if not frames:
            print("[play] no Isaac frames captured (rendering unavailable on this machine?)")
        else:
            import imageio.v2 as imageio
            if max(int(f.max()) for f in frames) < 10:
                print("[play] WARNING: all Isaac frames are black. The RTX renderer needs an RTX-capable GPU "
                      "(V100/A100/H100 are not); use --video_type traj instead.")
            mp4 = os.path.join(args_cli.out_dir, f"{args_cli.tag}_isaac.mp4")
            imageio.mimsave(mp4, frames, fps=args_cli.fps)
            print(f"[play] wrote {mp4} ({len(frames)} frames)")

    if use_traj_video:
        try:
            from render_rollout import render
            render(npz, os.path.join(args_cli.out_dir, f"{args_cli.tag}_traj.mp4"))
        except Exception as exc:  # e.g. matplotlib missing in the container: the npz is still there
            print(f"[play] cannot render the trajectory video here ({type(exc).__name__}: {exc}). Render the npz elsewhere with:\n"
                  f"       python scripts/hook_grasp/render_rollout.py {npz}")
    sys.stdout.flush()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(0)   # skip the headless Isaac Sim teardown (can hang)
