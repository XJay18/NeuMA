import os
import torch
from torch import nn
from torchvision.utils import save_image
import random
import trimesh
import argparse
import warp as wp
import numpy as np
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from tqdm.autonotebook import trange

from modules.nclaw.utils import (
    denormalize_points_helper_func
)
from modules.nclaw.sim import (
    MPMModelBuilder,
    MPMForwardSim,
    MPMStateInitializer,
    MPMStaticsInitializer,
    MPMInitData,
)
from modules.nclaw.material import (
    InvariantFullMetaElasticity,
    InvariantFullMetaPlasticity
)

from modules.d3gs.scene.gaussian_model import GaussianModel
from modules.tune.dataset.neuma_dataset import VideoDataset
from modules.tune.utils import (
    save_video_mediapy,
    prepare_simulation_data,
    compute_bindings_xyz,
    compute_bindings_F,
    diff_rasterization
)

ASSETS_PATH = Path(__file__).parent / "assets"
RESULT = "results"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", type=str, required=True,
        help="Path to the config file."
    )
    parser.add_argument(
        "--eval_steps", "-s", type=int, default=600,
        help="Number of simulation steps."
    )
    parser.add_argument(
        "--skip_frames", "-f", type=int, default=1,
        help="Number of skip frames when packing the video."
    )
    parser.add_argument(
        "--init_frame", type=int, default=None,
        help="Only load the camera parameters for the init frame to save time."
    )
    parser.add_argument(
        "--load_lora", "-l", type=str, default=None,
        help="Load lora weights from the given path."
    )
    parser.add_argument(
        "--remove_images", "-ri", action="store_true",
        help="Whether to remove images after packing video."
    )
    parser.add_argument(
        "--video_name", "-vn", type=str, required=True,
        help="Save video name."
    )
    parser.add_argument(
        "--sim_dt", "-dt", type=float, default=None,
        help="Time step for simulation."
    )
    parser.add_argument(
        "--debug_views", "-dv", nargs='+', default=[],
        help="Views for rendering."
    )
    parser.add_argument(
        "--save_particles", "-sp", type=str, default=None,
        help="Specify the folder name for saving simulated particles."
    )
    parser.add_argument(
        "--change_base_model", "-cbm", type=str, default=None,
        help="Replace the base model for rendering."
    )
    parser.add_argument(
        "--skip", type=int, default=1,
        help="Skip index."
    )
    parser.add_argument(
        "--dataset_path", type=str, default=None,
        help="Rewrite video dataset path."
    )
    parser.add_argument(
        "--transform_file", type=str, default=None,
        help="Rewrite transform_file name, e.g., 'eval_dynamic.json'."
    )
    parser.add_argument(
        "--alpha", type=float, default=None,
        help="Rewrite alpha value of the trained material adaptor."
    )

    args = parser.parse_args()
    return args


def eval(cfg: DictConfig):
    
    # init

    seed = cfg.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    wp.init()
    wp_device = wp.get_device(f'cuda:{cfg.gpu}')
    wp.ScopedTimer.enabled = False
    wp.set_module_options({'fast_math': False})

    torch_device = torch.device(f'cuda:{cfg.gpu}')
    torch.backends.cudnn.benchmark = True

    requires_grad = False

    # background

    background = (
        torch.tensor([1, 1, 1], dtype=torch.float32, device=torch_device)
        if cfg.video_data.data.get("white_background", False)   # default to black background
        else torch.tensor([0, 0, 0], dtype=torch.float32, device=torch_device)
    )

    # path

    if cfg.alpha is not None:
        cfg.constitution.lora.alpha = cfg.alpha

    root: Path = Path(cfg.root)
    exp_root: Path = root / cfg.name
    assert exp_root.exists(), f"Experiment {exp_root} does not exist."

    tune_root: Path = exp_root / 'finetune' # fine-tuned weights
    data_root: Path = ASSETS_PATH / cfg.sim_data_name

    # eval images
    image_root: Path = Path(RESULT) / cfg.name / f'images_{cfg.video_name}'
    image_root.mkdir(exist_ok=True, parents=True)

    # eval videos
    video_root: Path = Path(RESULT) / cfg.name / 'videos'
    video_root.mkdir(exist_ok=True, parents=True)

    if cfg.save_particles is not None:
        state_root: Path = Path(RESULT) / cfg.name / f'states_{cfg.save_particles}'
        state_root.mkdir(parents=True, exist_ok=True)

    # data

    if cfg.particle_data.get('particles_path') is not None:

        prepare_simulation_data(
            kernels_path=Path(cfg.gaussian.kernels_path),
            particles_path=Path(cfg.particle_data.particles_path),
            save_dir=data_root,
            sh_degree=cfg.gaussian.sh_degree,
            opacity_thres=cfg.gaussian.opacity_thres,
            particles_downsample_factor=cfg.particle_data.downsample_factor,
            confidence=cfg.gaussian.confidence,
            max_particles=cfg.gaussian.max_particles
        )

    elif cfg.particle_data.get('mesh_path') is not None:

        prepare_simulation_data(
            kernels_path=Path(cfg.gaussian.kernels_path),
            mesh_path=Path(cfg.particle_data.mesh_path),
            mesh_sample_mode=cfg.particle_data.mesh_sample_mode,
            mesh_sample_resolution=cfg.particle_data.mesh_sample_resolution,
            save_dir=data_root,
            sh_degree=cfg.gaussian.sh_degree,
            opacity_thres=cfg.gaussian.opacity_thres,
            particles_downsample_factor=1,
            confidence=cfg.gaussian.confidence,
            max_particles=cfg.gaussian.max_particles
        )

    cfg.video_data.device = f"cuda:{cfg.gpu}"           # NOTE: manually setting !!!
    cfg.video_data.data.init_frame = cfg.init_frame     # NOTE: manually setting !!!
    cfg.video_data.data.used_views = cfg.debug_views    # NOTE: manually setting !!!
    if cfg.dataset_path is not None:
        cfg.video_data.data.path = cfg.dataset_path
        print(f'Rewrite video dataset path to\n\t{cfg.dataset_path}')
    if cfg.transform_file is not None:
        cfg.video_data.data.transformsfile = cfg.transform_file
    dataset = VideoDataset(cfg.video_data)
    first_step = dataset.steps[0]

    #    -- binding data
    bind_data = torch.load(data_root / 'bindings.pt')
    # bindings: torch.Tensor = bind_data['bindings'].to(torch_device).float()
    bindings: torch.Tensor = torch.sparse_coo_tensor(
        bind_data['bindings_ind'], bind_data['bindings_val'], bind_data['bindings_size']
    ).to(torch_device).float()
    n_particles: torch.Tensor = bind_data['n_particles'].to(torch_device).float()

    has_particles = n_particles > 0
    print(f'Using data name [{cfg.sim_data_name}]')
    print(f'#Gaussians with particle bindings: {has_particles.sum()}')
    print(f'#Avg particles: {n_particles.mean()}')
    print(f'#Max particles: {n_particles.max()}')

    # gaussians

    gaussians = GaussianModel(cfg.gaussian.sh_degree)
    gaussians.load_ply(data_root / f"kernels.ply", requires_grad=False)

    # init velocity

    init_x_and_v = torch.load(tune_root / 'init.pt', map_location="cpu")
    dataset.set_init_x_and_v(init_x=init_x_and_v['init_x'], init_v=init_x_and_v['init_v'])

    elasticity: nn.Module = InvariantFullMetaElasticity(cfg.constitution.elasticity)
    elasticity.to(torch_device)
    elasticity.requires_grad_(requires_grad)
    elasticity.eval()
    
    plasticity: nn.Module = InvariantFullMetaPlasticity(cfg.constitution.plasticity)
    plasticity.to(torch_device)
    plasticity.requires_grad_(requires_grad)
    plasticity.eval()

    # load pretrained weights

    if cfg.get("change_base_model", None) is not None:
        ckpt_path = cfg.change_base_model
    else:
        ckpt_path = cfg.pretrained_ckpt
    pretrained = torch.load(ckpt_path, map_location=torch_device)
    elasticity.load_state_dict(pretrained['elasticity'])
    plasticity.load_state_dict(pretrained['plasticity'])
    print(f'Loaded pretrained weights from {ckpt_path}')

    if cfg.get("load_lora", None) is not None:
        cfg.constitution.load_lora = cfg.load_lora

    if cfg.constitution.get("load_lora", None) is not None:
        elasticity.init_lora_layers(r=cfg.constitution.lora.r, lora_alpha=cfg.constitution.lora.alpha)
        plasticity.init_lora_layers(r=cfg.constitution.lora.r, lora_alpha=cfg.constitution.lora.alpha)
        lora = torch.load(tune_root / cfg.constitution.load_lora, map_location=torch_device)
        elasticity.load_state_dict(lora['elasticity'], strict=False)
        plasticity.load_state_dict(lora['plasticity'], strict=False)
        print(f'Loaded lora weights from {tune_root / cfg.constitution.load_lora}')
    
    # warp

    eval_steps = cfg.eval_steps

    if cfg.sim_dt is not None:
        cfg.sim.dt = cfg.sim_dt                                     # NOTE: manually setting !!!
    model = MPMModelBuilder().parse_cfg(cfg.sim).finalize(wp_device, requires_grad)
    sim = MPMForwardSim(model)
    state_initializer = MPMStateInitializer(model)
    statics_initializer = MPMStaticsInitializer(model)
    cfg.particle_data.span = [0, eval_steps]                        # NOTE: manually setting !!!
    cfg.particle_data.shape.name = cfg.sim_data_name + "/particles" # NOTE: manually setting !!!
    init_data = MPMInitData.get(cfg.particle_data)
    print(f"[eval] Using dt: {model.constant.dt}")
    print(f"[eval] Using eps: {model.constant.eps}")
    print(f"[eval] Using bound: {model.constant.bound}")

    state_initializer.add_group(init_data)
    statics_initializer.add_group(init_data)

    state, _ = state_initializer.finalize()
    statics = statics_initializer.finalize()

    #    -- assertion
    assert init_data.pos.shape[0] == dataset.get_init_x.shape[0], \
        f"Shape mismatch: init_data {init_data.pos.shape[0]} dataset {dataset.get_init_x.shape[0]}"

    #    -- good to go
    x, v, C, F, _ = dataset.get_init_material_data()
    state.from_torch(x=x, v=v, C=C, F=F)

    #    -- first step
    for view in dataset.views:
        if view in cfg.get("debug_views", list()):
            # rasterize deformed kernels
            render = diff_rasterization(
                gaussians.get_xyz, None, gaussians,
                dataset.getCameras(view, first_step), background,
                scaling_modifier=cfg.gaussian.get('scaling_modifier', 1.0)
            )
            save_image(render, image_root / f"{view}_{first_step:03d}.png")

    de_x = denormalize_points_helper_func(x, init_data.size, init_data.center)

    de_x_prev = de_x.clone().detach()
    g_prev = gaussians.get_xyz.clone().detach()

    for step in trange(1, eval_steps + 1):
        stress = elasticity(F)
        state.from_torch(stress=stress)
        x, v, C, F = sim(statics, state)
        F = plasticity(F)
        state.from_torch(F=F)
        statics_initializer.update(statics, step)

        de_x = denormalize_points_helper_func(x, init_data.size, init_data.center)

        means3D = compute_bindings_xyz(de_x, de_x_prev, g_prev, bindings)
        deform_grad = compute_bindings_F(F, bindings)

        for view in dataset.views:
            if view in cfg.get("debug_views", list()):
                # rasterize deformed kernels
                render = diff_rasterization(
                    means3D, deform_grad, gaussians,
                    dataset.getCameras(view, first_step), background,
                    scaling_modifier=cfg.gaussian.get('scaling_modifier', 1.0)
                )
                save_image(render, image_root / f"{view}_{first_step + step:03d}.png")

        if cfg.save_particles is not None:
            t = trimesh.PointCloud(vertices=x.clone().detach().cpu())
            t.export(state_root / f'{first_step + step:03d}.ply')

        de_x_prev = de_x.clone().detach()
        g_prev = means3D.clone().detach()

    # pack video

    fps = 30

    for view in dataset.views:
        if view in cfg.get("debug_views", list()):
            save_video_mediapy(
                image_root, f"{view}_*.png",
                video_root / f"{cfg.video_name}_{view}.mp4",
                cfg.skip_frames, fps=fps
            )

    if cfg.remove_images:
        os.system(f"rm -rf {image_root}")


if __name__ == "__main__":
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    cfg.update(vars(args))
    cfg = DictConfig(cfg)

    with torch.no_grad():
        eval(cfg)
