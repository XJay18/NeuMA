# Modified from https://github.com/Colmar-zlicheng/Spring-Gaus/blob/main/lib/models/regist/Register.py

import torch
import torch.nn as nn
from omegaconf import DictConfig
from modules.tune.scheduler import CosineDecayScheduler
from modules.d3gs.utils.se3_utils import (
    rot6d_to_rotmat,
    rot6d_to_quat, quat_to_rot6d,
    euler_to_quat, quat_to_euler
)
from modules.d3gs.utils.transform_utils import (
    scale_transform,
    rotate_transform,
    translate_transform,
    transform_shs_by_quat,
)


class Register(nn.Module):

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.name = type(self).__name__
        self.cfg = cfg

        euler = torch.tensor(cfg.INIT_R, dtype=torch.float32, device="cuda") * torch.pi / 180
        quat = euler_to_quat(euler)
        rot6d = quat_to_rot6d(quat)

        self.r = nn.Parameter(rot6d, requires_grad=True)
        self.t = nn.Parameter(torch.tensor(cfg.INIT_T, dtype=torch.float32, device="cuda"), requires_grad=True)
        self.s = nn.Parameter(torch.tensor(cfg.INIT_S, dtype=torch.float32, device="cuda"), requires_grad=True)

    def training_setup(self):
        l = [{
            'params': [self.r],
            'lr': self.cfg.lr_r,
            "name": "r"
        }, {
            'params': [self.t],
            'lr': self.cfg.lr_t,
            "name": "t"
        }, {
            'params': [self.s],
            'lr': self.cfg.lr_s,
            "name": "s"
        }]
        self.optimizer = torch.optim.RAdam(l, lr=0.0, eps=1e-15)
        self.scheduler: torch.optim.lr_scheduler._LRScheduler = CosineDecayScheduler(self.cfg.scheduler).get_scheduler(self.optimizer, 0.0)

    @property
    def get_scale(self):
        return self.s

    @property
    def get_quat(self):
        return rot6d_to_quat(self.r)

    @property
    def get_euler(self):
        return torch.rad2deg(quat_to_euler(rot6d_to_quat(self.r)))

    @property
    def get_rotmat(self):
        return rot6d_to_rotmat(self.r)

    def forward(
        self,
        points: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        f_rest: torch.Tensor
    ):
        R = rot6d_to_rotmat(self.r)
        origin = torch.mean(points.clone().detach(), dim=0, keepdim=True)

        points, scales = scale_transform(points, scales, self.s)
        points, rotations = rotate_transform(points, rotations, R)
        points = translate_transform(points, self.t)

        f_rest = transform_shs_by_quat(f_rest, self.get_quat)

        out = {
            "points": points,
            "scales": scales,
            "rotations": rotations,
            "f_rest": f_rest,
            "origin": origin
        }
        return out
