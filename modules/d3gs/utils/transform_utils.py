import torch
import einops
from e3nn import o3
from typing import Union
from einops import einsum
from typing_extensions import Optional
from modules.d3gs.scene.gaussian_model import GaussianModel
try:
    from modules.d3gs.utils.se3_utils import rotmat_to_quat, quat_to_rotmat
except ImportError:
    print("[Warning] 'rotmat_to_quat' and 'quat_to_rotmat' which depend on 'pytorch3d' are not imported!")


def quaternion_multiply(q0, q1):
    w0, x0, y0, z0 = torch.unbind(q0, dim=-1)
    w1, x1, y1, z1 = torch.unbind(q1, dim=-1)
    return torch.stack((
        -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
        x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
        x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
    ), dim=-1)


def transform_shs_by_quat(shs_feat: torch.Tensor, quaternion: torch.Tensor):
    """
    Transform the spherical harmonics of gaussians by a given quaternion.
    Please refer to the following link for more details:
    https://github.com/graphdeco-inria/gaussian-splatting/issues/176#issuecomment-2147223570
    
    Args:
        shs_feat (torch.Tensor): The spherical harmonics of gaussians to be transformed.
        quaternion (torch.Tensor): The quaternion.

    Returns:
        torch.Tensor: The transformed spherical harmonics of gaussians.
    """
    return transform_shs_by_rotmat(shs_feat, quat_to_rotmat(quaternion))


def transform_shs_by_rotmat(shs_feat: torch.Tensor, rotation_matrix: torch.Tensor):
    """
    Transform the spherical harmonics of gaussians by a given rotation matrix.
    Please refer to the following link for more details:
    https://github.com/graphdeco-inria/gaussian-splatting/issues/176#issuecomment-2147223570

    Args:
        shs_feat (torch.Tensor): The spherical harmonics of gaussians to be transformed.
        rotation_matrix (torch.Tensor): The rotation matrix.

    Returns:
        torch.Tensor: The transformed spherical harmonics of gaussians.
    """

    device = shs_feat.device
    assert shs_feat.shape[-1] == 3, f"SH features must be in RGB format (N, SHS_NUM, 3), but got {shs_feat.shape}"

    if shs_feat.shape[1] <= 1:
        return shs_feat

    ## rotate shs
    P = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]], device=device, dtype=torch.float32)  # switch axes: yzx -> xyz
    permuted_rotation_matrix = torch.linalg.inv(P) @ rotation_matrix @ P
    rot_angles = o3._rotation.matrix_to_angles(permuted_rotation_matrix)

    # Construction coefficient
    D_1 = o3.wigner_D(1, rot_angles[0], - rot_angles[1], rot_angles[2])
    D_2 = o3.wigner_D(2, rot_angles[0], - rot_angles[1], rot_angles[2])
    D_3 = o3.wigner_D(3, rot_angles[0], - rot_angles[1], rot_angles[2])

    # rotation of the shs features
    one_degree_shs = shs_feat[:, 0:3]
    one_degree_shs = einops.rearrange(one_degree_shs, 'n shs_num rgb -> n rgb shs_num')
    one_degree_shs = einsum(
        D_1,
        one_degree_shs,
        "... i j, ... j -> ... i",
    )
    one_degree_shs = einops.rearrange(one_degree_shs, 'n rgb shs_num -> n shs_num rgb')
    shs_feat[:, 0:3] = one_degree_shs

    if shs_feat.shape[1] >= 4:
        two_degree_shs = shs_feat[:, 3:8]
        two_degree_shs = einops.rearrange(two_degree_shs, 'n shs_num rgb -> n rgb shs_num')
        two_degree_shs = einsum(
            D_2,
            two_degree_shs,
            "... i j, ... j -> ... i",
        )
        two_degree_shs = einops.rearrange(two_degree_shs, 'n rgb shs_num -> n shs_num rgb')
        shs_feat[:, 3:8] = two_degree_shs

        if shs_feat.shape[1] >= 9:
            three_degree_shs = shs_feat[:, 8:15]
            three_degree_shs = einops.rearrange(three_degree_shs, 'n shs_num rgb -> n rgb shs_num')
            three_degree_shs = einsum(
                D_3,
                three_degree_shs,
                "... i j, ... j -> ... i",
            )
            three_degree_shs = einops.rearrange(three_degree_shs, 'n rgb shs_num -> n shs_num rgb')
            shs_feat[:, 8:15] = three_degree_shs

    return shs_feat


def translate_gaussians(gaussians: GaussianModel, translation: torch.Tensor):
    """
    Translate the gaussians by a given translation vector. This operation is in-place.

    Args:
        gaussians (GaussianModel): The gaussians to be translated.
        translation (torch.Tensor): The translation vector.
    """
    assert translation.shape == (3,), f"Translation vector must have shape (3,), but got {translation.shape}."
    gaussians._xyz += translation.unsqueeze(0)


def scale_gaussians(gaussians: GaussianModel, scale: Union[torch.Tensor | float], origin: Optional[torch.Tensor] = None):
    """
    Scale the gaussians by a given scale. This operation is in-place.

    Args:
        gaussians (GaussianModel): The gaussians to be scaled.
        scale (torch.Tensor or float): The scale factor.
    """
    if isinstance(scale, float):
        scale = torch.tensor(scale, device=gaussians.get_xyz.device, dtype=torch.float32)
    elif isinstance(scale, torch.Tensor):
        assert scale.shape == (1,) or scale.shape == (), f"Scale factor must have shape (1,) or (), but got {scale.shape}."
    else:
        raise ValueError(f"Scale factor must be a torch.Tensor or a float, but got {type(scale)}.")
    
    if origin is None:
        origin = torch.mean(gaussians.get_xyz, dim=0, keepdim=True)
    gaussians._xyz = scale * (gaussians.get_xyz - origin)
    gaussians._scaling += torch.log(scale)


def rotate_gaussians(gaussians: GaussianModel, rotation_matrix: torch.Tensor):
    """
    Rotate the gaussians by a given rotation matrix. This operation is in-place.

    Args:
        gaussians (GaussianModel): The gaussians to be rotated.
        rotation_matrix (torch.Tensor): The rotation matrix.
    """
    assert rotation_matrix.shape == (3, 3), f"Rotation matrix must have shape (3, 3), but got {rotation_matrix.shape}."
    gaussians._xyz = gaussians.get_xyz @ rotation_matrix.T

    quat = rotmat_to_quat(rotation_matrix)[None, ...]
    quat_updated = quaternion_multiply(gaussians.get_rotation, quat)
    gaussians._rotation = torch.nn.functional.normalize(quat_updated, p=2, dim=-1)

    gaussians._features_rest = transform_shs_by_rotmat(gaussians._features_rest, rotation_matrix)


def translate_transform(points: torch.Tensor, translation: torch.Tensor):
    """
    Translate the gaussians with parameters [location, translation] by a given translation vector.

    Args:
        points (torch.Tensor): The locations of gaussian kernels.
        translation (torch.Tensor): The translation vector.
    
    Returns:
        torch.Tensor: The translated locations of gaussian kernels.
    """
    assert translation.shape == (3,), f"Translation vector must have shape (3,), but got {translation.shape}."
    points += translation.unsqueeze(0)
    return points


def scale_transform(points: torch.Tensor, point_scales: torch.Tensor, scale: Union[torch.Tensor | float], origin: Optional[torch.Tensor] = None):
    """
    Scale the gaussians with parameters [location, scaling] by a given scale.

    Args:
        points (torch.Tensor): The locations of gaussian kernels.
        point_scales (torch.Tensor): The scalings of gaussian kernels.
        scale (torch.Tensor or float): The scale factor.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The scaled locations of gaussian kernels and the updated scalings.
    """
    if isinstance(scale, float):
        scale = torch.tensor(scale, device=points.device, dtype=torch.float32)
    elif isinstance(scale, torch.Tensor):
        assert scale.shape == (1,) or scale.shape == (), f"Scale factor must have shape (1,) or (), but got {scale.shape}."
    else:
        raise ValueError(f"Scale factor must be a torch.Tensor or a float, but got {type(scale)}.")

    if origin is None:
        origin = torch.mean(points, dim=0, keepdim=True)
    points = scale * (points - origin)
    point_scales = point_scales + torch.log(scale)

    return points, point_scales


def rotate_transform(points: torch.Tensor, point_rotations: torch.Tensor, rotation_matrix: torch.Tensor):
    """
    Rotate the gaussians with parameters [location, rotation] by a given rotation matrix.
    Note that the spherical harmonics of gaussians are *not* updated in this function.

    Args:
        points (torch.Tensor): The locations of gaussian kernels.
        point_rotations (torch.Tensor): The rotations of gaussian kernels.
        rotation_matrix (torch.Tensor): The rotation matrix.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The rotated locations of gaussian kernels and the updated rotations.
    """
    assert rotation_matrix.shape == (3, 3), f"Rotation matrix must have shape (3, 3), but got {rotation_matrix.shape}."
    points = points @ rotation_matrix.T

    quat = rotmat_to_quat(rotation_matrix)[None, ...]
    quat_updated = quaternion_multiply(point_rotations, quat)
    point_rotations = torch.nn.functional.normalize(quat_updated, p=2, dim=-1)

    return points, point_rotations
