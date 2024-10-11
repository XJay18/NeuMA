import os
import torch
from torch import nn
from torchvision.utils import save_image
import random
import trimesh
import argparse
import warp as wp
import numpy as np
from typing import List
from pathlib import Path
from tqdm.autonotebook import trange
from omegaconf import DictConfig, OmegaConf

from modules.nclaw.utils import (
    denormalize_points,
    sample_vel
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
    InvariantFullMetaPlasticity,
    ComposeMaterial
)

from modules.d3gs.scene.gaussian_model import GaussianModel
from modules.d3gs.utils.transform_utils import (
    scale_gaussians,
    translate_gaussians
)
from modules.tune.dataset.neuma_dataset import CameraDataset
from modules.tune.utils import (
    save_video_mediapy,
    prepare_simulation_data,
    preprocess_for_rasterization,
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
        "--remove_images", "-ri", action="store_true",
        help="Whether to remove images after packing video."
    )
    parser.add_argument(
        "--video_name", "-vn", type=str, required=True,
        help="Save video name."
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
        "--dataset_path", type=str, default=None,
        help="Rewrite video dataset path."
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

    image_root: Path = Path(RESULT) / 'inference' / f'images_{cfg.video_name}'  # eval images
    image_root.mkdir(exist_ok=True, parents=True)

    video_root: Path = Path(RESULT) / 'inference_videos'  # eval videos
    video_root.mkdir(exist_ok=True)

    if cfg.save_particles is not None:
        state_root: Path = Path(RESULT) / 'inference_states' / f'states_{cfg.save_particles}'
        state_root.mkdir(parents=True, exist_ok=True)

    # warp

    eval_steps = cfg.eval_steps
    model = MPMModelBuilder().parse_cfg(cfg.sim).finalize(wp_device, requires_grad)
    sim = MPMForwardSim(model)
    state_initializer = MPMStateInitializer(model)
    statics_initializer = MPMStaticsInitializer(model)

    # env

    if cfg.dataset_path is not None:
        cfg.video_data.data.path = cfg.dataset_path
        print(f'Rewrite video dataset path to\n\t{cfg.dataset_path}')
    if len(cfg.get("debug_views", list())) > 0:
        cfg.video_data.data.used_views = cfg.debug_views
    dataset = CameraDataset(cfg.video_data)
    first_step = dataset.steps[0]

    elasticities = list()
    plasticities = list()

    obj_gaussians: List[GaussianModel] = list()
    sec_gaussians = list()
    obj_bindings = list()
    obj_k_prev = list()
    obj_p_prev = list()
    obj_p_curr = list()
    obj_deform_grad = list()
    obj_scalings = list()

    points_all = list()

    for obj_cfg in cfg.objects:

        data_root: Path = ASSETS_PATH / obj_cfg.sim_data_name
        print(f'\nLoad data for {obj_cfg.sim_data_name} ...')

        #    -- prepare data for each obj
        if obj_cfg.particle_data.get('particles_path') is not None:

            prepare_simulation_data(
                kernels_path=Path(obj_cfg.gaussian.kernels_path),
                particles_path=Path(obj_cfg.particle_data.particles_path),
                save_dir=data_root,
                sh_degree=obj_cfg.gaussian.sh_degree,
                opacity_thres=obj_cfg.gaussian.opacity_thres,
                particles_downsample_factor=obj_cfg.particle_data.downsample_factor,
                confidence=obj_cfg.gaussian.confidence,
                max_particles=obj_cfg.gaussian.max_particles
            )
        
        elif obj_cfg.particle_data.get('mesh_path') is not None:

            prepare_simulation_data(
                kernels_path=Path(obj_cfg.gaussian.kernels_path),
                mesh_path=Path(obj_cfg.particle_data.mesh_path),
                mesh_sample_mode=obj_cfg.particle_data.mesh_sample_mode,
                mesh_sample_resolution=obj_cfg.particle_data.mesh_sample_resolution,
                save_dir=data_root,
                sh_degree=obj_cfg.gaussian.sh_degree,
                opacity_thres=obj_cfg.gaussian.opacity_thres,
                particles_downsample_factor=1,
                confidence=obj_cfg.gaussian.confidence,
                max_particles=obj_cfg.gaussian.max_particles
            )
        
        obj_point = np.array(trimesh.load(data_root / "particles.ply").vertices)
        points_all.append(obj_point)

        #    -- binding data
        bind_data = torch.load(data_root / 'bindings.pt')
        # bindings: torch.Tensor = bind_data['bindings'].to(torch_device).float()
        bindings: torch.Tensor = torch.sparse_coo_tensor(
            bind_data['bindings_ind'], bind_data['bindings_val'], bind_data['bindings_size']
        ).to(torch_device).float()
        n_particles: torch.Tensor = bind_data['n_particles'].to(torch_device).float()

        has_particles = n_particles > 0
        print(f'#Gaussians with particle bindings: {has_particles.sum()}')
        print(f'#Avg particles: {n_particles.mean()}')
        print(f'#Max particles: {n_particles.max()}, index: {torch.argmax(n_particles)}')
        obj_bindings.append(bindings)

        # gaussians

        gaussians = GaussianModel(obj_cfg.gaussian.sh_degree)
        gaussians.load_ply(data_root / f"kernels.ply", requires_grad=False)
        obj_gaussians.append(gaussians)
        sec_gaussians.append(gaussians.get_xyz.shape[0])
        obj_scalings.append(obj_cfg.gaussian.get('scaling_modifier', 1.0))

        # material

        elasticity: nn.Module = InvariantFullMetaElasticity(obj_cfg.constitution.elasticity)
        elasticity.to(torch_device)

        plasticity: nn.Module = InvariantFullMetaPlasticity(obj_cfg.constitution.plasticity)
        plasticity.to(torch_device)

        #    -- load pretrained weights
        ckpt_path = obj_cfg.pretrained_ckpt
        pretrained = torch.load(ckpt_path, map_location=torch_device)
        elasticity.load_state_dict(pretrained['elasticity'])
        plasticity.load_state_dict(pretrained['plasticity'])
        print(f'Loaded pretrained weights from {ckpt_path}')

        #    -- load lora weights
        if obj_cfg.constitution.get("load_lora", None) is not None:
            elasticity.init_lora_layers(r=obj_cfg.constitution.lora.r, lora_alpha=obj_cfg.constitution.lora.alpha)
            plasticity.init_lora_layers(r=obj_cfg.constitution.lora.r, lora_alpha=obj_cfg.constitution.lora.alpha)
            lora = torch.load(obj_cfg.constitution.load_lora, map_location=torch_device)
            elasticity.load_state_dict(lora['elasticity'], strict=False)
            plasticity.load_state_dict(lora['plasticity'], strict=False)
            print(f'Loaded lora weights from {obj_cfg.constitution.load_lora}')
    
        obj_cfg.particle_data.span = [0, eval_steps]                            # NOTE: manually setting !!!
        obj_cfg.particle_data.shape.name = obj_cfg.sim_data_name + "/particles" # NOTE: manually setting !!!
        init_data = MPMInitData.get(obj_cfg.particle_data)

        #    -- init velocity
        if obj_cfg.particle_data.get("vel") is not None:
            print(f'Use initial velocity: {obj_cfg.particle_data.vel} ...')
            lin_vel = np.array(obj_cfg.particle_data.vel.lin_vel)
            ang_vel = np.array(obj_cfg.particle_data.vel.ang_vel)
        else:
            print(f'Randomly sample initial velocity ...')
            lin_vel, ang_vel = sample_vel(seed=42)
        init_data.set_lin_vel(lin_vel)
        init_data.set_ang_vel(ang_vel)

        state_initializer.add_group(init_data)
        statics_initializer.add_group(init_data)
        elasticities.append(elasticity)
        plasticities.append(plasticity)

    state, sections = state_initializer.finalize()
    statics = statics_initializer.finalize()
    x, v, C, F, stress = state.to_torch()

    elasticity = ComposeMaterial(elasticities, sections)
    elasticity.to(torch_device)
    elasticity.requires_grad_(requires_grad)
    elasticity.eval()

    plasticity = ComposeMaterial(plasticities, sections)
    plasticity.to(torch_device)
    plasticity.requires_grad_(requires_grad)
    plasticity.eval()

    #    -- first step
    if not cfg.get('denormalize', False):
        # if not denormalize, this means the visualization is performed in the simulation bounds,
        # so we need to transform the gaussians from the original bounds to the simulation bounds
        # for visualization
        for i, group_data in enumerate(state_initializer.groups):
            scale_gaussians(obj_gaussians[i], group_data.size[0], torch.zeros(3).to(torch_device))
            translate_gaussians(obj_gaussians[i], torch.from_numpy(group_data.center).float().to(torch_device))

    first_means3D = torch.cat([g.get_xyz for g in obj_gaussians], dim=0)
    first_cov3D = torch.cat([g.get_covariance(s) for g, s in zip(obj_gaussians, obj_scalings)], dim=0)
    first_opa3D = torch.cat([g.get_opacity for g in obj_gaussians], dim=0)
    first_shs = torch.cat([g.get_features for g in obj_gaussians], dim=0)

    for view in dataset.views:
        if view in cfg.get("debug_views", list()):
            # rasterize deformed kernels
            render = diff_rasterization(
                first_means3D, None, None,
                dataset.getCameras(view, first_step), background,
                obj_gaussians[0].active_sh_degree,  # NOTE: assume all gaussians have the same sh degree
                first_cov3D,
                first_opa3D,
                first_shs
            )
            save_image(render, image_root / f"{view}_{0:03d}.png")

    de_x = denormalize_points(x, sections, state_initializer) if cfg.get('denormalize', False) else x

    for sec_x in torch.split(de_x, sections, dim=0):
        obj_p_prev.append(sec_x.clone().detach())
    for gaussians in obj_gaussians:
        obj_k_prev.append(gaussians.get_xyz.clone().detach())

    for step in trange(1, eval_steps + 1):
        stress = elasticity(F)
        state.from_torch(stress=stress)
        x, v, C, F = sim(statics, state)
        F = plasticity(F)
        state.from_torch(F=F)
        statics_initializer.update(statics, step)

        de_x = denormalize_points(x, sections, state_initializer) if cfg.get('denormalize', False) else x

        for sec_x, sec_F in zip(
            torch.split(de_x, sections, dim=0),
            torch.split(F, sections, dim=0)
        ):
            obj_p_curr.append(sec_x)
            obj_deform_grad.append(sec_F)

        pack = preprocess_for_rasterization(
            obj_gaussians=obj_gaussians,
            obj_deform_grad=obj_deform_grad,
            obj_kernels_prev=obj_k_prev,
            obj_particles_curr=obj_p_curr,
            obj_particles_prev=obj_p_prev,
            obj_bindings=obj_bindings,
            obj_scalings=obj_scalings
        )

        # clear the states
        obj_k_prev = list()
        obj_p_prev = list()
        obj_p_curr = list()
        obj_deform_grad = list()

        for view in dataset.views:
            if view in cfg.get("debug_views", list()):
                # rasterize deformed kernels
                render = diff_rasterization(
                    pack['means3D'], pack['deform_grad'], None,
                    dataset.getCameras(view, first_step), background,
                    pack['active_sh_degree'],
                    pack['cov3D'],
                    pack['opacity'],
                    pack['shs']
                )
                save_image(render, image_root / f"{view}_{step:03d}.png")

        if cfg.save_particles is not None:
            t = trimesh.PointCloud(vertices=x.clone().detach().cpu())
            t.export(state_root / f'{first_step + step:03d}.ply')

        for sec_x in torch.split(de_x, sections, dim=0):
            obj_p_prev.append(sec_x.clone().detach())
        for sec_k in torch.split(pack['means3D'], sec_gaussians, dim=0):
            obj_k_prev.append(sec_k.clone().detach())
        
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
