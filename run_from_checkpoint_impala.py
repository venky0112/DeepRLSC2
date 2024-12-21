import argparse
import sys
import os
import torch
import multiprocessing
from absl import flags
from pysc2.env import sc2_env
from torchbeast.core import file_writer, environment
from SC_Utils.game_utils import IMPALA_ObsProcesser_v2
from AC_modules.IMPALA import IMPALA_AC_v2

# Define absl.flags
FLAGS = flags.FLAGS
# flags.DEFINE_string("sc2_run_config", None, "Which run_config to use to spawn the binary")


def init_sc2_environment(game_params, map_name, resolution=32):
    """Initialize StarCraft II environment."""
    race = sc2_env.Race.terran
    agent = sc2_env.Agent(race, "TestAgent")
    agent_interface_format = sc2_env.AgentInterfaceFormat(
        feature_dimensions=sc2_env.Dimensions(screen=resolution, minimap=resolution)
    )
    env_config = dict(
        map_name=map_name,
        players=[agent],
        agent_interface_format=[agent_interface_format],
        step_mul=8,
        game_steps_per_episode=0,
    )
    return sc2_env.SC2Env(**env_config)


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load checkpoint and return the last training state."""
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint.get("step", 0)
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")


def train(flags, model, optimizer, step):
    """Run training loop."""
    plogger = file_writer.FileWriter(xpid=flags["xpid"], xp_args=flags, rootdir=flags["savedir"])
    for current_step in range(step, flags["total_steps"]):
        # Perform training steps (implement actual training logic here).
        if current_step % 1000 == 0:
            print(f"Step: {current_step}")

        # Save checkpoint periodically
        if current_step % 5000 == 0:
            checkpoint_path = os.path.join(flags["savedir"], flags["xpid"], "model.tar")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": current_step,
                },
                checkpoint_path,
            )
            print(f"Checkpoint saved at {checkpoint_path}")


def run():
    # Parse argparse arguments first
    parser = argparse.ArgumentParser(description="Run IMPALA training from a checkpoint")
    parser.add_argument("--xpid", type=str, required=True, help="Experiment ID of the checkpoint to resume")
    parser.add_argument("--map_name", type=str, default="MoveToBeacon", help="StarCraft II minigame map name")
    parser.add_argument("--total_steps", type=int, default=200000, help="Total steps to train")
    parser.add_argument("--savedir", type=str, default="./logs/torchbeast", help="Directory where the checkpoint is saved")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate for resumed training")
    parser.add_argument("--num_actors", type=int, default=4, help="Number of actor processes")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--unroll_length", type=int, default=60, help="Unroll length (time dimension)")
    parser.add_argument("--disable_cuda", action="store_true", help="Disable CUDA")
    args, unknown_flags = parser.parse_known_args()

    # Parse absl.flags with unknown args
    FLAGS(sys.argv[:1] + unknown_flags)

    # Path to the checkpoint
    checkpoint_dir = os.path.join(args.savedir, args.xpid)
    checkpoint_path = os.path.join(checkpoint_dir, "model.tar")

    # Initialize SC2 environment
    game_params = {
        "env": {"feature_screen": 32, "feature_minimap": 32, "action_space": "FEATURES"},
        "obs_processer": {"select_all": True},
    }
    sc_env = init_sc2_environment(game_params, args.map_name)

    # Wrap the SC2 environment with the IMPALA observation processor
    obs_processor = IMPALA_ObsProcesser_v2(env=sc_env, action_table=None, **game_params["obs_processer"])
    env = environment.Environment(sc_env, obs_processor)

    # Initialize model and optimizer
    device = "cuda" if torch.cuda.is_available() and not args.disable_cuda else "cpu"
    model = IMPALA_AC_v2(env=env, device=device, **game_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Load checkpoint
    step = load_checkpoint(checkpoint_path, model, optimizer)

    # Configure training parameters
    flags = {
        "num_actors": args.num_actors,
        "batch_size": args.batch_size,
        "unroll_length": args.unroll_length,
        "savedir": args.savedir,
        "total_steps": args.total_steps,
        "xpid": args.xpid,
        "device": device,
    }

    # Resume training
    print(f"Resuming training from step {step}")
    train(flags, model, optimizer, step)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    run()
