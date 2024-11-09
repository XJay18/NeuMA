import os
import sys
import time
import torch
import mediapy
import imageio
import trimesh
import warp as wp
import numpy as np
from PIL import Image
from pathlib import Path
from natsort import natsorted
from typing import Optional, List, Tuple
from typing_extensions import Literal
from modules.nclaw.sph import volume_sampling
from modules.d3gs.scene.gaussian_model import GaussianModel
from modules.d3gs.gaussian_renderer import get_rasterizer
from modules.d3gs.utils.binding_utils import (
    gaussian_binding,
    gaussian_binding_with_clip_v1
)
from modules.d3gs.utils.simulation_utils import (
    torch2warp_mat33,
    deform_cov_by_F
)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


class Timer(object):
    """Time recorder."""

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


def verbose_points(points: torch.Tensor, tag: str=''):
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    print(f"{tag}\n  x: [{x_min}, {x_max}]")
    print(f"  y: [{y_min}, {y_max}]")
    print(f"  z: [{z_min}, {z_max}]\n")


def save_video_mediapy(
    frame_dir: Path,
    frame_name: str,
    output_path: Path,
    skip_frame: int = 1,
    fps: int = 30,
    white_bg: bool = False,
):
    np_frames = list()
    image_paths = [i for i in frame_dir.glob(frame_name)]
    image_paths = natsorted(image_paths)[::skip_frame]

    for image_path in image_paths:
        image = Image.open(image_path)
        if image.mode == "RGBA":
            background_color = np.array([1, 1, 1]) if white_bg else np.array([0, 0, 0])
            image_rgba = np.array(image)
            norm_rgba = image_rgba / 255.0
            norm_rgba = norm_rgba[:, :, :3] * norm_rgba[:, :, 3:] + (1 - norm_rgba[:, :, 3:]) * background_color
            image_arr = np.array(norm_rgba*255.0, dtype=np.uint8)
        elif image.mode == "RGB":
            image_arr = np.array(image)
        else:
            raise ValueError(f"Unsupported image mode: {image.mode}")
        np_frames.append(image_arr)
    
    mediapy.write_video(output_path, np_frames, fps=fps, qp=18)
    print(f"Video saved to {output_path} with skip frame {skip_frame} and fps {fps}")


# In case mediapy not works correctly, use imageio to save gifs
def save_gif_imageio(
    frame_dir: Path,
    frame_name: str,
    output_path: Path,
    skip_frame: int = 1,
    fps: int = 30,
    white_bg: bool = False,
    resize: Optional[Tuple[int, int]] = None,
):
    np_frames = list()
    image_paths = [i for i in frame_dir.glob(frame_name)]
    image_paths = natsorted(image_paths)[::skip_frame]

    for image_path in image_paths:
        image = Image.open(image_path)
        if resize is not None:
            # resize to 400x400
            image = image.resize(size=resize)
        if image.mode == "RGBA":
            background_color = np.array([1, 1, 1]) if white_bg else np.array([0, 0, 0])
            image_rgba = np.array(image)
            norm_rgba = image_rgba / 255.0
            norm_rgba = norm_rgba[:, :, :3] * norm_rgba[:, :, 3:] + (1 - norm_rgba[:, :, 3:]) * background_color
            image_arr = np.array(norm_rgba*255.0, dtype=np.uint8)
        elif image.mode == "RGB":
            image_arr = np.array(image)
        else:
            raise ValueError(f"Unsupported image mode: {image.mode}")
        np_frames.append(image_arr)
    
    with imageio.get_writer(output_path, mode='I', fps=fps, loop=0) as writer:
        for frame in np_frames:
            writer.append_data(frame)

    print(f"GIF saved to {output_path} with skip frame {skip_frame} and fps {fps}")


def uniform_sampling(mesh: trimesh.Trimesh, resolution: int) -> np.ndarray:
    bounds = mesh.bounds.copy()
    # mesh.vertices = (mesh.vertices - bounds[0]) / (bounds[1] - bounds[0])
    mesh.vertices = mesh.vertices - bounds[0]
    upper_bound = mesh.vertices.max(0)
    dims = np.linspace(np.zeros(3), upper_bound, resolution).T
    grid = np.stack(np.meshgrid(*dims, indexing='ij'), axis=-1).reshape(-1, 3)
    p_x = grid[mesh.contains(grid)]
    # undo normalization
    p_x = p_x + bounds[0]

    return p_x


def volumetric_sampling(mesh: trimesh.Trimesh, resolution: int, asset_path: Path) -> np.ndarray:
    import pyvista

    bounds = mesh.bounds.copy()
    mesh.vertices = (mesh.vertices - bounds.mean(0)) / (bounds[1] - bounds[0]).max() + 0.5
    cache_obj_path = asset_path / f'temp.obj'
    cache_vtk_path = asset_path / f'temp.vtk'
    mesh.export(cache_obj_path)

    radius = 1.0 / resolution * 0.5
    volume_sampling(cache_obj_path, cache_vtk_path, radius, res=(resolution, resolution, resolution))
    pcd: pyvista.PolyData = pyvista.get_reader(str(cache_vtk_path)).read()
    p_x = np.array(pcd.points).copy()

    # undo normalization
    p_x = (p_x - 0.5) * (bounds[1] - bounds[0]).max() + bounds.mean(0)

    cache_obj_path.unlink(missing_ok=True)
    cache_vtk_path.unlink(missing_ok=True)

    return p_x


def surface_sampling(mesh: trimesh.Trimesh, resolution: int) -> np.ndarray:
    # resolution in this case is the number of points to sample
    points = trimesh.sample.sample_surface_even(mesh, resolution // 2)[0]

    noise = np.random.normal(0, 0.001, points.shape)
    points_n1 = points.copy() + noise

    return np.concatenate([points, points_n1], axis=0)


def get_warp_device(device: torch.device) -> wp.context.Device:
    if device.type == 'cuda':
        return wp.get_device(f'cuda:{device.index}')
    else:
        return wp.get_device('cpu')


@torch.no_grad()
def prepare_simulation_data(
    save_dir: Path,
    kernels_path: Path,
    particles_path: Optional[Path] = None,
    mesh_path: Optional[Path] = None,
    mesh_sample_mode: Literal["uniform", "volumetric", "surface"] = "volumetric",
    mesh_sample_resolution: int = 30,
    sh_degree: int = 3,
    opacity_thres: float = 0.02,
    particles_downsample_factor: int = 3,
    confidence: float = 0.95,
    max_particles: int = 10,
):
    if (
        (save_dir / "kernels.ply").is_file() 
        and (save_dir / "particles.ply").is_file()
        and (save_dir / "bindings.pt").is_file()
    ):
        print("===================================")
        print(f"Data already prepared. Skipping data preparation.\n")

    else:
        print("===================================")
        print(f'Start preparing data for simulation.\n')

        gaussians = GaussianModel(sh_degree)
        gaussians.load_ply(kernels_path.as_posix())

        opacity = gaussians.get_opacity
        retain_flag = opacity.squeeze() > opacity_thres

        print(f'Gaussians after pruning low opacity kernels: {retain_flag.sum()}')

        gaussians.load_ply_with_mask(kernels_path.as_posix(), retain_flag.cpu().numpy())
        gaussians.save_ply((save_dir / "kernels.ply").as_posix())

        if particles_path is not None:
            print(f'Extracting particles from pcd file [{particles_path}] ...')
            particles = trimesh.load(particles_path).vertices
        elif mesh_path is not None:
            print(f'Extracting particles from mesh file [{mesh_path}] ...')
            os.system(f"cp {mesh_path} {save_dir}/mesh{mesh_path.suffix}")
            mesh: trimesh.Trimesh = trimesh.load(mesh_path, force='mesh')
            if not mesh.is_watertight:
                print(f'[**WARNING**] Invalid mesh from [{mesh_path}]: not watertight!')
                print(f'[**WARNING**] Please manually check the sampled particles in case of unexpected results!')
            if mesh_sample_mode == "uniform":
                particles = uniform_sampling(mesh, mesh_sample_resolution)
            elif mesh_sample_mode == "volumetric":
                particles = volumetric_sampling(mesh, mesh_sample_resolution, save_dir)
            elif mesh_sample_mode == "surface":
                particles = surface_sampling(mesh, mesh_sample_resolution)
            else:
                raise ValueError(f"Unsupported mesh sample mode: {mesh_sample_mode}")
        else:
            raise ValueError("Either 'particles_path' or 'mesh_path' must be provided.")
        particles = torch.from_numpy(particles).float().cuda()

        # downsample particles
        rand_idx = torch.randperm(particles.shape[0])
        particles = particles[rand_idx][::particles_downsample_factor]
        particles = particles.contiguous()

        # pre comp binding
        print(f'Pre-compute bindings to find gaussians without particle bindings ...')
        flag_mat_pre = gaussian_binding(gaussians, particles, confidence=confidence)

        num_particles_pre = flag_mat_pre.sum(1)
        has_particles_pre = num_particles_pre > 0

        to_clone_means3D = gaussians.get_xyz[~has_particles_pre].requires_grad_(False)
        print(f'Particles to be added: {to_clone_means3D.shape}')

        particles = torch.cat([particles, to_clone_means3D], dim=0)

        del flag_mat_pre, num_particles_pre, has_particles_pre

        # finalize binding
        print(f'Finalize binding computation ...')
        weight_mat = gaussian_binding_with_clip_v1(
            gaussians, particles,
            confidence=confidence,
            max_particles=max_particles
        )
        # NOTE: The size of the flag mat should not exceed INT_MAX = 2_147_483_647
        assert weight_mat.reshape(-1).shape[0] < torch.iinfo(torch.int32).max
        weight_mat_sparse = weight_mat.to_sparse_coo()
        print("COO: ", weight_mat_sparse.indices().shape)

        num_particles = (weight_mat > 0).sum(1)

        # save data
        particles_path = save_dir / "particles.ply"
        point_np = particles.cpu().numpy()
        point_tr = trimesh.PointCloud(vertices=point_np)
        point_tr.export(particles_path)

        torch.save({
            # NOTE: different pytorch version may have different behavior
            # "bindings" : weight_mat_sparse.cpu(),
            # NOTE: use the following form to save the sparse tensor
            "bindings_ind": weight_mat_sparse.indices().cpu(),
            "bindings_val": weight_mat_sparse.values().cpu(),
            "bindings_size": weight_mat_sparse.size(),
            "n_particles": num_particles.cpu()
        }, save_dir / "bindings.pt")

    print(f'\nData preparation done.')
    print("===================================\n")


def diff_rasterization(
    x: torch.Tensor,
    deform_grad: Optional[torch.Tensor],
    gaussians: Optional[GaussianModel],
    view_cam,
    background_color: torch.Tensor,
    gaussians_active_sh: Optional[int] = None,
    guassians_cov: Optional[torch.Tensor] = None,
    gaussians_opa: Optional[torch.Tensor] = None,
    gaussians_shs: Optional[torch.Tensor] = None,
    scaling_modifier: Optional[float] = 1.,
    force_mask_data: Optional[bool] = False
) -> torch.Tensor:  
    device = x.device
    means3D = x

    if gaussians is not None:
        cov3D_precomp = gaussians.get_covariance(scaling_modifier=scaling_modifier)
        opacity = gaussians.get_opacity
        shs = gaussians.get_features
        sh_degree = gaussians.active_sh_degree
    else:
        cov3D_precomp = guassians_cov
        opacity = gaussians_opa
        shs = gaussians_shs
        sh_degree = gaussians_active_sh

    assert means3D.shape[0] == cov3D_precomp.shape[0], \
        f"Shape mismatch: means3D {means3D.shape[0]} cov3D {cov3D_precomp.shape[0]}"

    if deform_grad is not None:
        tensor_F = torch.reshape(deform_grad, (-1, 3, 3))
        wp_F = torch2warp_mat33(tensor_F, dvc=device.type)

        assert cov3D_precomp.shape[0] == tensor_F.shape[0], \
            f"Shape mismatch: cov3D {cov3D_precomp.shape[0]} F {tensor_F.shape[0]}"

        wp_cov3D_precomp = wp.from_torch(
            cov3D_precomp.reshape(-1),
            dtype=wp.float32
        )
        wp_cov3D_deformed = wp.zeros_like(wp_cov3D_precomp)
        wp.launch(
            deform_cov_by_F,
            dim=tensor_F.shape[0],
            inputs=[wp_cov3D_precomp, wp_F, wp_cov3D_deformed],
            device=device.type
        )
        wp.synchronize()

        cov3D_deformed = wp.to_torch(wp_cov3D_deformed).reshape(-1, 6)
    else:
        cov3D_deformed = cov3D_precomp

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    means2D = screenspace_points
    
    rasterizer = get_rasterizer(
        view_cam, sh_degree,
        debug=False, bg_color=background_color,
    )

    if force_mask_data:
        # Rasterize visible Gaussians to image.
        rendered_image, _ = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=None,
            colors_precomp=torch.ones(means3D.shape[0], 3, device=device),
            opacities=opacity,
            scales=None,
            rotations=None,
            cov3D_precomp=cov3D_deformed
            # scales=scales,
            # rotations=rotations,
            # cov3D_precomp=None
        )
    else:
        # Rasterize visible Gaussians to image.
        rendered_image, _ = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=None,
            opacities=opacity,
            scales=None,
            rotations=None,
            cov3D_precomp=cov3D_deformed
            # scales=scales,
            # rotations=rotations,
            # cov3D_precomp=None
        )

    return rendered_image


def compute_bindings_xyz(
    p_curr: torch.Tensor,
    p_prev: torch.Tensor,
    k_prev: torch.Tensor,
    bindings: torch.Tensor,
):
    """Compute updated location of gaussian kernels.

    Args:
        p_curr: Current particles xyz.
        p_prev: Previous particles xyz.
        k_prev: Previous kernels xyz.
        bindings: Binding matrix.
    
    Returns:
        k_curr: Updated kernels xyz.
    """
    delta_x = p_curr - p_prev.detach()

    # calculate means3D
    delta_means3D = torch.sparse.mm(bindings, delta_x)
    delta_means3D = delta_means3D.to_dense()
    k_curr = k_prev.detach() + delta_means3D

    return k_curr


def compute_bindings_F(
    deform_grad: torch.Tensor,
    bindings: torch.Tensor,
):
    """Compute updated deformation gradiant for each gaussian kernel.

    Args:
        deform_grad: Deformation gradient of each particle.
        bindings: Binding matrix.
    
    Returns:
        tensor_F: Deformation gradiant for each gaussian kernel.
    """

    # calculate deformation gradient
    tensor_F = torch.reshape(deform_grad, (-1, 9))
    tensor_F = torch.sparse.mm(bindings, tensor_F)
    tensor_F = tensor_F.to_dense()

    # reshape to (kernels, 3, 3)
    tensor_F = torch.reshape(tensor_F, (-1, 3, 3))
    return tensor_F


def preprocess_for_rasterization(
    obj_gaussians: List[GaussianModel],
    obj_deform_grad: List[torch.Tensor],
    obj_kernels_prev: List[torch.Tensor],
    obj_particles_curr: List[torch.Tensor],
    obj_particles_prev: List[torch.Tensor],
    obj_bindings: List[torch.Tensor],
    obj_scalings: List[float]
):  
    # x, deform_grad, cov, opa, shs
    obj_x = list()
    obj_F = list()
    obj_cov = list()
    obj_opa = list()
    obj_shs = list()

    # compute updated location of gaussian kernels
    for p_curr, p_prev, k_prev, bindings in zip(
        obj_particles_curr, obj_particles_prev, obj_kernels_prev, obj_bindings
    ):
        k_curr = compute_bindings_xyz(p_curr, p_prev, k_prev, bindings)
        obj_x.append(k_curr)

    # compute updated deformation gradiant for each gaussian kernel
    for deform_grad, bindings in zip(obj_deform_grad, obj_bindings):
        tensor_F = compute_bindings_F(deform_grad, bindings)
        obj_F.append(tensor_F)

    for gaussians, scaling in zip(obj_gaussians, obj_scalings):
        obj_cov.append(gaussians.get_covariance(scaling_modifier=scaling))
        obj_opa.append(gaussians.get_opacity)
        obj_shs.append(gaussians.get_features)
    
    out_x = torch.cat(obj_x, dim=0)
    out_F = torch.cat(obj_F, dim=0)
    out_cov = torch.cat(obj_cov, dim=0)
    out_opa = torch.cat(obj_opa, dim=0)
    out_shs = torch.cat(obj_shs, dim=0)

    out_dict = {
        "means3D": out_x,
        "deform_grad": out_F,
        "cov3D": out_cov,
        "opacity": out_opa,
        "shs": out_shs,
        "active_sh_degree": obj_gaussians[0].active_sh_degree
    }

    return out_dict