import torch
import warp as wp


def torch2warp_mat33(t, copy=False, dvc="cuda"):
    assert t.is_contiguous()
    if t.dtype != torch.float32 and t.dtype != torch.int32:
        raise RuntimeError(
            "Error aliasing Torch tensor to Warp array. Torch tensor must be float32 or int32 type"
        )
    assert t.shape[1] == 3
    a = wp.types.array(
        ptr=t.data_ptr(),
        dtype=wp.mat33,
        shape=t.shape[0],
        copy=copy,
        owner=False,
        requires_grad=t.requires_grad,
        # device=t.device.type)
        device=dvc,
    )
    a.tensor = t
    return a

@wp.kernel
def deform_cov_by_F(
    init_cov: wp.array(dtype=float),
    F: wp.array(dtype=wp.mat33),
    out_cov: wp.array(dtype=float)
):
    tid = wp.tid()

    thread_F = F[tid]

    thread_cov = wp.mat33(
        init_cov[tid * 6], init_cov[tid * 6 + 1], init_cov[tid * 6 + 2],
        init_cov[tid * 6 + 1], init_cov[tid * 6 + 3], init_cov[tid * 6 + 4],
        init_cov[tid * 6 + 2], init_cov[tid * 6 + 4], init_cov[tid * 6 + 5]
    )

    cov = thread_F * thread_cov * wp.transpose(thread_F)

    out_cov[tid * 6] = cov[0, 0]
    out_cov[tid * 6 + 1] = cov[0, 1]
    out_cov[tid * 6 + 2] = cov[0, 2]
    out_cov[tid * 6 + 3] = cov[1, 1]
    out_cov[tid * 6 + 4] = cov[1, 2]
    out_cov[tid * 6 + 5] = cov[2, 2]
