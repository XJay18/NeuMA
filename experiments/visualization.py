import time
import argparse
from modules.vis.renderer import Renderer

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", type=str, required=True,
        help="Path to the config file."
    )
    parser.add_argument(
        "--port", type=int, default=8890,
        help="Port number for the viewer."
    )
    parser.add_argument(
        "--work-dir", "-d", type=str, default="results",
        help="Path to the working directory."
    )
    parser.add_argument(
        "--eval-steps", "-s", type=int, default=400,
        help="Number of simulation steps."
    )
    parser.add_argument(
        "--skip-frames", "-f", type=int, default=5,
        help="Number of skip frames."
    )
    parser.add_argument(
        "--up-axis", "-up", choices=["x", "y", "z"], default="y",
        help="Up axis."
    )

    args = parser.parse_args()
    return args

def main(args):
    renderer = Renderer.init_from_config_file(
        path=args.config,
        work_dir=args.work_dir,
        port=args.port,
        eval_steps=args.eval_steps,
        skip_frames=args.skip_frames,
        up_axis=args.up_axis
    )

    print(f"Please visit http://localhost:{renderer.viewer.server.get_port()} to view the rendering ...")

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    args = parse_args()
    main(args)
