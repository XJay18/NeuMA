import os
import sys
import random
import trimesh
import torch
from torch import nn
import warp as wp
import numpy as np
from typing import List, Tuple
from pathlib import Path
from tqdm.autonotebook import trange
from nerfview import CameraState
from omegaconf import DictConfig
from collections import namedtuple

from modules.nclaw.utils import sample_vel

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

from modules.d3gs.scene.cameras import MiniCam
from modules.d3gs.scene.gaussian_model import GaussianModel
from modules.d3gs.utils.graphics_utils import (
    focal2fov, getProjectionMatrix
)
from modules.d3gs.utils.transform_utils import (
    scale_gaussians, translate_gaussians
)
from modules.tune.utils import (
    prepare_simulation_data,
    preprocess_for_rasterization,
    diff_rasterization
)

ASSETS_PATH = Path(__file__).parent.parent.parent / "experiments" / "assets"

NeuMAGaus = namedtuple('NeuMAGaus', ['mean', 'cov', 'opacity', 'sh', 'defgrad'])

class NeuMAInstance(object):
    def __init__(self, cfg: DictConfig):
        # init

        seed = cfg.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        wp.init()
        self.wp_device = wp.get_device(f'cuda:{cfg.gpu}')
        wp.ScopedTimer.enabled = False
        wp.set_module_options({'fast_math': False})

        self.torch_device = torch.device(f'cuda:{cfg.gpu}')
        torch.backends.cudnn.benchmark = True

        # warp

        cfg.sim.eps = 6e-7  # in case numerical errors
        self.mpm = MPMModelBuilder().parse_cfg(cfg.sim).finalize(self.wp_device, requires_grad=False)

        # env

        self.cfg = cfg

        #    -- use white background for the interactive viewer
        self.background = torch.tensor([1, 1, 1], dtype=torch.float32, device=self.torch_device)

        self.simulate()
    
    def simulate(self, cfg_objects: DictConfig | None = None):
        """ Update objects and re-simulate """

        # override objects

        if cfg_objects is not None:
            self.cfg.objects = cfg_objects

        # init objects
        
        print('Initialize objects for simulation ...')
        x, F, stress, state, sections, statics = self._init_objects()
        
        # simulate

        print('\nStart simulation ...')
        self._simulate(x, F, stress, state, sections, statics)
        print('Simulation done! Ready for rendering ...')
    
    def _init_objects(self):
        """ Initialize objects for simulation """

        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')          # suppress print

        cfg = self.cfg
        elasticities = list()
        plasticities = list()

        self.obj_gaussians: List[GaussianModel] = list()
        self.sec_gaussians = list()
        self.obj_bindings = list()
        self.obj_scalings = list()

        self.points_all = list()
        
        # NOTE: assume all gaussians have the same sh degree
        try:
            self.sh_degree = cfg.objects[0].gaussian.sh_degree
        except:
            self.sh_degree = 3

        self.state_initializer = MPMStateInitializer(self.mpm)
        self.statics_initializer = MPMStaticsInitializer(self.mpm)

        # iterate over objects

        for obj_cfg in cfg.objects:

            #   -- skip non-existing objects
            if not obj_cfg.get('exists', True):
                continue

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
            self.points_all.append(obj_point)

            #    -- binding data
            bind_data = torch.load(data_root / 'bindings.pt')
            # bindings: torch.Tensor = bind_data['bindings'].to(torch_device).float()
            bindings: torch.Tensor = torch.sparse_coo_tensor(
                bind_data['bindings_ind'], bind_data['bindings_val'], bind_data['bindings_size']
            ).to(self.torch_device).float()
            n_particles: torch.Tensor = bind_data['n_particles'].to(self.torch_device).float()

            has_particles = n_particles > 0
            print(f'#Gaussians with particle bindings: {has_particles.sum()}')
            print(f'#Avg particles: {n_particles.mean()}')
            print(f'#Max particles: {n_particles.max()}, index: {torch.argmax(n_particles)}')
            self.obj_bindings.append(bindings)

            # gaussians

            gaussians = GaussianModel(obj_cfg.gaussian.sh_degree)
            gaussians.load_ply(data_root / f"kernels.ply", requires_grad=False)
            self.obj_gaussians.append(gaussians)
            self.sec_gaussians.append(gaussians.get_xyz.shape[0])
            self.obj_scalings.append(obj_cfg.gaussian.get('scaling_modifier', 1.0))

            # material

            elasticity: nn.Module = InvariantFullMetaElasticity(obj_cfg.constitution.elasticity)
            elasticity.to(self.torch_device)

            plasticity: nn.Module = InvariantFullMetaPlasticity(obj_cfg.constitution.plasticity)
            plasticity.to(self.torch_device)

            #    -- load pretrained weights
            ckpt_path = obj_cfg.pretrained_ckpt
            pretrained = torch.load(ckpt_path, map_location=self.torch_device)
            elasticity.load_state_dict(pretrained['elasticity'])
            plasticity.load_state_dict(pretrained['plasticity'])
            print(f'Loaded pretrained weights from {ckpt_path}')

            #    -- load lora weights
            if obj_cfg.constitution.get("load_lora", None) is not None:
                elasticity.init_lora_layers(r=obj_cfg.constitution.lora.r, lora_alpha=obj_cfg.constitution.lora.alpha)
                plasticity.init_lora_layers(r=obj_cfg.constitution.lora.r, lora_alpha=obj_cfg.constitution.lora.alpha)
                lora = torch.load(obj_cfg.constitution.load_lora, map_location=self.torch_device)
                elasticity.load_state_dict(lora['elasticity'], strict=False)
                plasticity.load_state_dict(lora['plasticity'], strict=False)
                print(f'Loaded lora weights from {obj_cfg.constitution.load_lora}')
        
            obj_cfg.particle_data.span = [0, cfg.eval_steps]                            # NOTE: manually setting !!!
            obj_cfg.particle_data.shape.name = obj_cfg.sim_data_name + "/particles"     # NOTE: manually setting !!!
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

            self.state_initializer.add_group(init_data)
            self.statics_initializer.add_group(init_data)
            elasticities.append(elasticity)
            plasticities.append(plasticity)
        
        state, sections = self.state_initializer.finalize()
        statics = self.statics_initializer.finalize()
        x, _, _, F, stress = state.to_torch()

        elasticity = ComposeMaterial(elasticities, sections)
        elasticity.to(self.torch_device)
        elasticity.requires_grad_(False)
        elasticity.eval()
        self.elasticity = elasticity

        plasticity = ComposeMaterial(plasticities, sections)
        plasticity.to(self.torch_device)
        plasticity.requires_grad_(False)
        plasticity.eval()
        self.plasticity = plasticity

        sys.stdout = old_stdout                     # restore print

        return x, F, stress, state, sections, statics
    
    @torch.no_grad()
    def _simulate(self, x, F, stress, state, sections, statics):
        """ Simulate the scene and store the results for rendering """

        # init the states
        self.gaus: List[NeuMAGaus] = list()

        # first step
        for i, group_data in enumerate(self.state_initializer.groups):
            scale_gaussians(self.obj_gaussians[i], group_data.size[0], torch.zeros(3).to(self.torch_device))
            translate_gaussians(self.obj_gaussians[i], torch.from_numpy(group_data.center).float().to(self.torch_device))

        first_means3D = torch.cat([g.get_xyz for g in self.obj_gaussians], dim=0)
        first_cov3D = torch.cat([g.get_covariance(s) for g, s in zip(self.obj_gaussians, self.obj_scalings)], dim=0)
        first_opa3D = torch.cat([g.get_opacity for g in self.obj_gaussians], dim=0)
        first_shs = torch.cat([g.get_features for g in self.obj_gaussians], dim=0)

        self.gaus.append(NeuMAGaus(
            first_means3D.to("cpu"),
            first_cov3D.to("cpu"),
            first_opa3D.to("cpu"),
            first_shs.to("cpu"),
            None
        ))

        # init simulation model

        sim = MPMForwardSim(self.mpm)

        obj_k_prev = list()
        obj_p_prev = list()
        obj_p_curr = list()
        obj_deform_grad = list()

        for sec_x in torch.split(x, sections, dim=0):
            obj_p_prev.append(sec_x.clone().detach())
        for gaussians in self.obj_gaussians:
            obj_k_prev.append(gaussians.get_xyz.clone().detach())
        
        for step in trange(1, self.cfg.eval_steps + 1):
            stress = self.elasticity(F)
            state.from_torch(stress=stress)
            x, _, _, F = sim(statics, state)
            F = self.plasticity(F)
            state.from_torch(F=F)
            self.statics_initializer.update(statics, step)

            for sec_x, sec_F in zip(
                torch.split(x, sections, dim=0),
                torch.split(F, sections, dim=0)
            ):
                obj_p_curr.append(sec_x)
                obj_deform_grad.append(sec_F)

            pack = preprocess_for_rasterization(
                obj_gaussians=self.obj_gaussians,
                obj_deform_grad=obj_deform_grad,
                obj_kernels_prev=obj_k_prev,
                obj_particles_curr=obj_p_curr,
                obj_particles_prev=obj_p_prev,
                obj_bindings=self.obj_bindings,
                obj_scalings=self.obj_scalings
            )

            # clear the states
            obj_k_prev = list()
            obj_p_prev = list()
            obj_p_curr = list()
            obj_deform_grad = list()

            if step % self.cfg.skip_frames == 0:
                self.gaus.append(NeuMAGaus(
                    pack['means3D'].to("cpu"),
                    pack['cov3D'].to("cpu"),
                    pack['opacity'].to("cpu"),
                    pack['shs'].to("cpu"),
                    pack['deform_grad'].to("cpu")
                ))

            for sec_x in torch.split(x, sections, dim=0):
                obj_p_prev.append(sec_x.clone().detach())
            for sec_k in torch.split(pack['means3D'], self.sec_gaussians, dim=0):
                obj_k_prev.append(sec_k.clone().detach())

    @torch.no_grad()
    def render(
        self,
        t: int | None,
        camera_state: CameraState,
        img_wh: Tuple[int, int],
    ):
        width, height = img_wh
        znear, zfar = 0.01, 100.0

        c2w = torch.from_numpy(camera_state.c2w.astype(np.float32)).to(self.torch_device)
        w2c = torch.linalg.inv(c2w)
        world_view_transform = w2c.transpose(0, 1)

        focal = 0.5 * height / np.tan(0.5 * camera_state.fov).item()
        fovx = focal2fov(focal, width)
        fovy = focal2fov(focal, height)

        projection_matrix = getProjectionMatrix(
            znear, zfar, fovx, fovy
        ).transpose(0,1).to(self.torch_device)

        full_proj_transform = (
            world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
        ).squeeze(0)

        viewpoint = MiniCam(
            width, height, fovy, fovx, 
            znear, zfar,
            world_view_transform, full_proj_transform
        )

        t = t or 0

        render = diff_rasterization(
            self.gaus[t].mean.to(self.torch_device),
            self.gaus[t].defgrad.to(self.torch_device) if self.gaus[t].defgrad is not None else None, 
            None,
            viewpoint, self.background,
            self.sh_degree,
            self.gaus[t].cov.to(self.torch_device),
            self.gaus[t].opacity.to(self.torch_device),
            self.gaus[t].sh.to(self.torch_device)
        )

        render = render.permute(1, 2, 0)
        render = torch.clamp(render, 0., 1.)

        return render
