import torch
import warp as wp
from scipy.stats import chi2
from tqdm.autonotebook import trange
from modules.d3gs.scene.gaussian_model import GaussianModel


def test_point_in_gaussians(point, mean, cov, confidence):
    """
    Test if a point is inside a Gaussian distribution.
    """
    point = torch.from_numpy(point)
    mean = torch.from_numpy(mean)
    cov = torch.from_numpy(cov)

    d = point - mean
    p = d @ torch.linalg.inv(cov) @ d

    return p <= confidence, p


@wp.kernel
def compute_inv_cov(
    cov: wp.array(dtype=float),
    out: wp.array(dtype=float)
):
    tid = wp.tid()

    cov_wp = wp.mat33(
        cov[tid * 6], cov[tid * 6 + 1], cov[tid * 6 + 2],
        cov[tid * 6 + 1], cov[tid * 6 + 3], cov[tid * 6 + 4],
        cov[tid * 6 + 2], cov[tid * 6 + 4], cov[tid * 6 + 5]
    )

    inv_cov_wp = wp.inverse(cov_wp)

    out[tid * 9 + 0] = inv_cov_wp[0, 0]
    out[tid * 9 + 1] = inv_cov_wp[0, 1]
    out[tid * 9 + 2] = inv_cov_wp[0, 2]
    out[tid * 9 + 3] = inv_cov_wp[1, 0]
    out[tid * 9 + 4] = inv_cov_wp[1, 1]
    out[tid * 9 + 5] = inv_cov_wp[1, 2]
    out[tid * 9 + 6] = inv_cov_wp[2, 0]
    out[tid * 9 + 7] = inv_cov_wp[2, 1]
    out[tid * 9 + 8] = inv_cov_wp[2, 2]


@wp.kernel
def test_point_in_gaussians_warp(
    point: wp.array(dtype=wp.vec3),
    mean: wp.array(dtype=float),
    cov: wp.array(dtype=float),
    threshold: wp.float32,
    out: wp.array(dtype=wp.int8),
    out_p: wp.array(dtype=float)
):
    tid = wp.tid()

    mean_wp = wp.vec3(mean[0], mean[1], mean[2])
    cov_wp = wp.mat33(
        cov[0], cov[1], cov[2],
        cov[1], cov[3], cov[4],
        cov[2], cov[4], cov[5]
    )

    d = point[tid] - mean_wp

    inv_cov_wp = wp.inverse(cov_wp)

    # manually calculate p1 = d * inv(cov)
    p11 = d[0] * inv_cov_wp[0, 0] + d[1] * inv_cov_wp[1, 0] + d[2] * inv_cov_wp[2, 0]
    p12 = d[0] * inv_cov_wp[0, 1] + d[1] * inv_cov_wp[1, 1] + d[2] * inv_cov_wp[2, 1]
    p13 = d[0] * inv_cov_wp[0, 2] + d[1] * inv_cov_wp[1, 2] + d[2] * inv_cov_wp[2, 2]

    # mannualy calculate p = p1 * d
    p1 = wp.vec3(p11, p12, p13)
    p = wp.dot(p1, d)

    if p <= threshold:
        out[tid] = wp.int8(1)
    else:
        out[tid] = wp.int8(0)

    out_p[tid] = p


@wp.kernel
def test_point_in_gaussians_with_inv_cov_warp(
    point: wp.array(dtype=wp.vec3),
    mean: wp.array(dtype=float),
    inv_cov: wp.array(dtype=float),
    threshold: wp.float32,
    out: wp.array(dtype=wp.int8),
    out_p: wp.array(dtype=float)
):
    tid = wp.tid()

    mean_wp = wp.vec3(mean[0], mean[1], mean[2])
    inv_cov_wp = wp.mat33(
        inv_cov[0], inv_cov[1], inv_cov[2],
        inv_cov[3], inv_cov[4], inv_cov[5],
        inv_cov[6], inv_cov[7], inv_cov[8]
    )

    d = point[tid] - mean_wp

    # manually calculate p1 = d * inv(cov)
    p11 = d[0] * inv_cov_wp[0, 0] + d[1] * inv_cov_wp[1, 0] + d[2] * inv_cov_wp[2, 0]
    p12 = d[0] * inv_cov_wp[0, 1] + d[1] * inv_cov_wp[1, 1] + d[2] * inv_cov_wp[2, 1]
    p13 = d[0] * inv_cov_wp[0, 2] + d[1] * inv_cov_wp[1, 2] + d[2] * inv_cov_wp[2, 2]

    # mannualy calculate p = p1 * d
    p1 = wp.vec3(p11, p12, p13)
    p = wp.dot(p1, d)

    if p <= threshold:
        out[tid] = wp.int8(1)
    else:
        out[tid] = wp.int8(0)

    out_p[tid] = p


def gaussian_binding(
    gaussians: GaussianModel,
    particles: torch.Tensor,
    confidence: float = 0.95,
    max_particles: int = 10,
):
    print(f'Start binding gaussians with particles ...')
    print(f'  chi: {chi2.ppf(confidence, 3)}')
    print(f'  g: {gaussians.get_xyz.shape} {gaussians.get_xyz.device}')
    print(f'  p: {particles.shape} {particles.device}')

    # initialize (kernels, particles)
    flag_mat = torch.zeros(
        (gaussians.get_xyz.shape[0], particles.shape[0]),
        dtype=torch.bool, device=particles.device
    )    
    point_wp = wp.from_torch(particles, dtype=wp.vec3)

    means = gaussians.get_xyz
    covs = gaussians.get_covariance()
    covs_wp = wp.from_torch(covs.reshape(-1), dtype=wp.float32)
    inv_covs_wp = wp.zeros(covs.shape[0] * 9, dtype=wp.float32, device=point_wp.device)

    # compute inverse of covariance only once
    wp.launch(
        compute_inv_cov,
        dim=covs.shape[0],
        inputs=[covs_wp, inv_covs_wp],
        device=point_wp.device
    )

    inv_covs = wp.to_torch(inv_covs_wp).reshape(-1, 9)

    for k in trange(gaussians.get_xyz.shape[0]):
        mean = means[k]
        mean_wp = wp.from_torch(mean, dtype=wp.float32)
        inv_cov = inv_covs[k]
        inv_cov_wp = wp.from_torch(inv_cov.reshape(-1), dtype=wp.float32)
        out_wp = wp.zeros(particles.shape[0], dtype=wp.int8, device=point_wp.device)
        out_p_wp = wp.zeros(particles.shape[0], dtype=wp.float32, device=point_wp.device)

        wp.launch(
            test_point_in_gaussians_with_inv_cov_warp,
            dim=particles.shape[0],
            inputs=[
                point_wp,
                mean_wp,
                inv_cov_wp,
                wp.float32(chi2.ppf(confidence, 3)),
                out_wp,
                out_p_wp
            ],
            device=point_wp.device
        )

        flag = wp.to_torch(out_wp).bool()
        out_p = wp.to_torch(out_p_wp)
        flag_new = torch.zeros_like(flag)

        if flag.sum() > max_particles:
            # we need to downsample the particles
            # sort particles in the gaussian
            idx = torch.argsort(out_p[flag], descending=False)[:max_particles]
            global_idx = torch.arange(flag.shape[0], device=particles.device)
            global_idx = global_idx[flag][idx]

            flag_new[global_idx] = True
        else:
            flag_new[flag] = True

        flag_mat[k] = flag_new

    return flag_mat

# FIXME: Covariance may be inf, thus we need to cull Gaussians whose covs too small
def gaussian_binding_with_clip_v1(
    gaussians: GaussianModel,
    particles: torch.Tensor,
    confidence: float = 0.95,
    max_particles: int = 10,
):
    print(f'Start binding gaussians with particles (clip v1) ...')
    print(f'  chi: {chi2.ppf(confidence, 3)}')
    print(f'  g: {gaussians.get_xyz.shape} {gaussians.get_xyz.device}')
    print(f'  p: {particles.shape} {particles.device}')

    # initialize (kernels, particles)
    weight_mat = torch.zeros(
        (gaussians.get_xyz.shape[0], particles.shape[0]),
        dtype=torch.float32, device=particles.device
    )
    point_wp = wp.from_torch(particles, dtype=wp.vec3)

    means = gaussians.get_xyz
    covs = gaussians.get_covariance()
    covs_wp = wp.from_torch(covs.reshape(-1), dtype=wp.float32)
    inv_covs_wp = wp.zeros(covs.shape[0] * 9, dtype=wp.float32, device=point_wp.device)

    # compute inverse of covariance only once
    wp.launch(
        compute_inv_cov,
        dim=covs.shape[0],
        inputs=[covs_wp, inv_covs_wp],
        device=point_wp.device
    )

    inv_covs = wp.to_torch(inv_covs_wp).reshape(-1, 9)

    for k in trange(gaussians.get_xyz.shape[0]):
        mean = means[k]
        mean_wp = wp.from_torch(mean, dtype=wp.float32)
        inv_cov = inv_covs[k]
        inv_cov_wp = wp.from_torch(inv_cov.reshape(-1), dtype=wp.float32)
        out_wp = wp.zeros(particles.shape[0], dtype=wp.int8, device=point_wp.device)
        out_p_wp = wp.zeros(particles.shape[0], dtype=wp.float32, device=point_wp.device)

        wp.launch(
            test_point_in_gaussians_with_inv_cov_warp,
            dim=particles.shape[0],
            inputs=[
                point_wp,
                mean_wp,
                inv_cov_wp,
                wp.float32(chi2.ppf(confidence, 3)),
                out_wp,
                out_p_wp
            ],
            device=point_wp.device
        )

        flag = wp.to_torch(out_wp).bool()
        out_p = wp.to_torch(out_p_wp)
        weight = torch.zeros_like(out_p)

        if flag.sum() > max_particles:
            # we need to downsample the particles
            # sort particles in the gaussian
            idx = torch.argsort(out_p[flag], descending=False)[:max_particles]
            global_idx = torch.arange(flag.shape[0], device=particles.device)
            global_idx = global_idx[flag][idx]

            temp_p = out_p[flag][idx].clone()
            temp_p = torch.ones_like(temp_p)
            # compute softmax of temp_p
            temp_weight = torch.softmax(-temp_p, dim=-1)
            weight[global_idx] = temp_weight
        else:
            temp_p = out_p[flag].clone()
            temp_p = torch.ones_like(temp_p)
            # compute softmax of temp_p
            temp_weight = torch.softmax(-temp_p, dim=-1)
            weight[flag] = temp_weight

        if weight.sum() == 0:
            print(f'k: {k}, weight: {weight.sum()} flag: {flag.sum()} out_p: {out_p.argmin()} {out_p.min()}')
            print(f'cov: {covs[k]}')
            print(f'inv_cov: {inv_cov}')
        assert weight.sum() != 0

        weight_mat[k] = weight

    return weight_mat
