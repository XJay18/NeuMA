import torch
from torchvision.utils import save_image

import os
import random
import trimesh
import argparse
import numpy as np
from pathlib import Path
from tqdm.autonotebook import tqdm
from omegaconf import DictConfig, OmegaConf

from modules.tune.regist.register import Register
from modules.tune.utils import (
    diff_rasterization,
    uniform_sampling,
    volumetric_sampling,
    surface_sampling
)
from modules.tune.dataset.neuma_dataset import VideoDataset
from modules.d3gs.scene.gaussian_model import GaussianModel
from modules.d3gs.utils.loss_utils import l1_loss, l2_loss, ssim

ASSETS_PATH = Path(__file__).parent / "assets"
PIXEL_LOSSES = {
    "l1": l1_loss,
    "l2": l2_loss
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", type=str, required=True,
        help="Path to the config file."
    )
    args = parser.parse_args()
    return args


def transform_pcd(points, scale, origin, rotation, translation):
    """
    Transform a point cloud by scaling, rotating, and translating it.
    """
    points = (points - origin) * scale
    points = np.dot(points, rotation.T)
    points = points + translation
    return points


def regist_gaussians(cfg: DictConfig):
    data_root: Path = ASSETS_PATH / cfg.sim_data_name
    data_root.mkdir(exist_ok=True)

    if (
        (data_root / 'registered_params.npz').is_file()
        and (data_root / 'registered_kernels.ply').is_file() 
    ):
        print("===================================")
        print(f"Registration for Gaussians already finished. Skip.\n")
        print(f"\nRegistration finished.")
        print("===================================")

    else:
        print(OmegaConf.to_yaml(cfg))

        print("\n===================================")
        print(f'Registering Gaussian kernels ...\n')

        # init

        seed = cfg.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        debug = cfg.debug

        torch_device = torch.device(f'cuda:{cfg.gpu}')
        torch.backends.bencmark = True

        # background

        force_mask_data = cfg.video_data.data.get("read_mask_only", False)
        if force_mask_data:
            # Force to use black background when loading mask data
            cfg.video_data.data.white_background = False
            print(f"[Warning] Force to use black background when loading mask data")

        background = (
            torch.tensor([1, 1, 1], dtype=torch.float32, device=torch_device)
            if cfg.video_data.data.get("white_background", False)   # default to black background
            else torch.tensor([0, 0, 0], dtype=torch.float32, device=torch_device)
        )


        OmegaConf.save(cfg, data_root / 'config.yaml', resolve=True)

        if debug:
            debug_root: Path = data_root / 'debug'
            debug_root.mkdir(exist_ok=True)
        
        # data

        #    -- video data
        cfg.video_data.device = f"cuda:{cfg.gpu}"       # NOTE: manually setting !!!
        dataset = VideoDataset(cfg.video_data)
        used_views = dataset.views if cfg.register.get("views", "all") == 'all' else cfg.register.views
        used_views = sorted(used_views)
        first_step = dataset.steps[0]
        pixel_loss = PIXEL_LOSSES[cfg.register.get("pixel_loss", "l1")]
        lambda_ssim_loss = cfg.register.get("lambda_ssim_loss", 0.0)
        print(f"[register] Using views: {used_views}")
        print(f"[register] Using first step: {first_step}")
        print(f"[register] Using pixel loss: {cfg.register.get('pixel_loss', 'l1')}")
        print(f"[register] Lambda ssim loss: {lambda_ssim_loss}")

        #   -- gaussian model
        gaussians = GaussianModel(cfg.gaussian.sh_degree)
        gaussians.load_ply(cfg.gaussian.kernels_path)
        means3D = gaussians.get_xyz
        f_rest = gaussians._features_rest
        scales = gaussians._scaling
        rots = gaussians.get_rotation

        print(f"[register] Training register ...")

        # register

        register = Register(cfg.register)
        register.training_setup()

        # log

        ema_loss_for_log = 0.0
        progress_bar = tqdm(range(1, cfg.register.num_iter + 1), position=0, leave=True, desc='Registering gaussians ...')

        # register

        for i in progress_bar:

            loss = 0.
            render_list = list()
            gt_list = list()

            pack = register(
                points=means3D.clone().detach(),
                scales=scales.clone().detach(),
                rotations=rots.clone().detach(),
                f_rest=f_rest.clone().detach()
            )
            gaussians._xyz = pack["points"]
            gaussians._scaling = pack["scales"]
            gaussians._rotation = pack["rotations"]
            gaussians._features_rest = pack["f_rest"]

            for view in used_views:
                render = diff_rasterization(
                    gaussians._xyz, None, gaussians,
                    dataset.getCameras(view, first_step), background,
                    scaling_modifier=cfg.gaussian.get('scaling_modifier', 1.0),
                    force_mask_data=force_mask_data
                )
                gt = dataset.getCameras(view, first_step).original_image.to(means3D.device)

                loss += (1.0 - lambda_ssim_loss) * pixel_loss(render, gt) + lambda_ssim_loss * (1.0 - ssim(render, gt))

                if (i == 1 or i % 500 == 0) and debug:
                    render_vis = render.clone().detach().cpu()
                    gt_vis = gt.clone().detach().cpu()
                    render_vis = render_vis[:, 100:500, 720:1100]
                    gt_vis = gt_vis[:, 100:500, 720:1100]
                    render_list.append(render_vis)
                    gt_list.append(gt_vis)

            loss.backward()
            register.optimizer.step()
            register.optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if i % 10 == 0:
                    progress_bar.set_postfix_str(
                        f"Loss {ema_loss_for_log:.7f} | "
                        f"r [{register.get_euler[0].item():.2f}, {register.get_euler[1].item():.2f}, {register.get_euler[2].item():.2f}] | "
                        f"t [{register.t[0].item():.3f}, {register.t[1].item():.3f}, {register.t[2].item():.3f}] | "
                        f"s {register.s.item():.4f}"
                    )
                
                if (i == 1 or i % 500 == 0) and debug:
                    render_batch = torch.stack(render_list, dim=0)
                    gt_batch = torch.stack(gt_list, dim=0)
                    debug_batch = torch.cat([render_batch, gt_batch], dim=0)
                    save_image(debug_batch, debug_root / f"regist_iter_{i}.png", nrow=len(used_views))
            
            register.scheduler.step()

        np.savez_compressed(
            data_root / "registered_params.npz",
            r=register.get_rotmat.detach().cpu().numpy(),
            t=register.t.detach().cpu().numpy(),
            s=register.s.detach().cpu().numpy(),
            o=pack["origin"].detach().cpu().numpy()
        )
        gaussians.save_ply(data_root / "registered_kernels.ply")
        print(f"\nRegistration finished. Loss: {ema_loss_for_log:.7f}")
        print("===================================")


def regist_particles(cfg: DictConfig):
    save_dir: Path = ASSETS_PATH / cfg.sim_data_name
    if (save_dir / "registered_particles.ply").is_file():
        print("\n===================================")
        print(f"Registration for Particles already finished. Skip.\n")

        transformed_particles = trimesh.load(save_dir / "registered_particles.ply").vertices
    else:
        print("\n===================================")
        print(f'Registering Particles ...\n')

        mesh_path = Path(cfg.particle_data.mesh_path)
        print(f'Extracting particles from mesh file [{mesh_path}] ...')
        os.system(f"cp {mesh_path} {save_dir}/mesh{mesh_path.suffix}")
        mesh: trimesh.Trimesh = trimesh.load(mesh_path, force='mesh')
        if not mesh.is_watertight:
            raise ValueError(f'Invalid mesh from [{mesh_path}]: not watertight')
        mesh_sample_mode = cfg.particle_data.get("mesh_sample_mode", "volumetric")
        mesh_sample_resolution = cfg.particle_data["mesh_sample_resolution"]
        if mesh_sample_mode == "uniform":
            particles = uniform_sampling(mesh, mesh_sample_resolution)
        elif mesh_sample_mode == "volumetric":
            particles = volumetric_sampling(mesh, mesh_sample_resolution, save_dir)
        elif mesh_sample_mode == "surface":
            particles = surface_sampling(mesh, mesh_sample_resolution)
        else:
            raise ValueError(f"Unsupported mesh sample mode: {mesh_sample_mode}")
        
        transform = np.load(save_dir / "registered_params.npz")
        scale = transform["s"]
        rotation = transform["r"]
        translation = transform["t"]
        origin = transform["o"]

        transformed_particles = transform_pcd(particles, scale, origin, rotation, translation)
        transformed_pcd = trimesh.PointCloud(transformed_particles)
        transformed_pcd.export(save_dir / "registered_particles.ply")

    print(f"\nRegistration finished. Registed particles: {transformed_particles.shape}")
    print("===================================")


if __name__ == "__main__":
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    cfg = DictConfig(cfg)
    regist_gaussians(cfg)
    regist_particles(cfg)
