import torch
from torch import nn
import numpy as np
from numpy.typing import NDArray
from torch.utils.data import Dataset
from omegaconf import OmegaConf, DictConfig
from typing import Optional, Union
from modules.tune.scheduler import fetch_scheduler
from modules.d3gs.scene.dataset_readers import (
    readRealCaptureCameras,
    readNeuMASyntheticCameras,
)
from modules.d3gs.scene.cameras import PhysCamera
from modules.d3gs.utils.camera_utils import physCameraList_from_camInfos

class CameraDataset(Dataset):
    def __init__(self, cfg: DictConfig, readCameras=True):
        self.eval = cfg.eval
        self.cameras = {}
        if readCameras:
            self.readCameras(cfg)

    def readCameras(self, cfg: DictConfig):
        self.cameras = {}
        mode = "Training" if not self.eval else "Testing"
        camera_type = cfg.camera_type
        read_fn = eval(f"read{camera_type}Cameras")
        print(f"Reading {mode} Data")
        info = read_fn(**cfg.data)
        self.views = info["views"]  # sorted
        self.steps = info["steps"]  # sorted
        self.length = len(self.views) * len(self.steps)

        print(f"Loading {mode} Cameras")
        if cfg.camera.get("data_device") is None:
            cfg.camera.data_device = cfg.device
        print(f'Setting default device for camera data to [{cfg.camera.data_device}]')
        temp_cam_list = physCameraList_from_camInfos(info["cam_infos"], 1.0, cfg.camera)
        for cam in temp_cam_list:
            if cam.view in self.cameras:
                self.cameras[cam.view].update({
                    cam.step: cam
                })
            else:
                self.cameras.update({
                    cam.view: {cam.step: cam}
                })
        print(f"Loaded the Camera Set with {len(self.cameras)} views and {len(self.steps)} steps")
        if len(self.views) < 20:
            print(f"    Views: {self.views}")
        if len(self.steps) < 20:
            print(f"    Steps: {self.steps}")
    
    def getCameras(self, view, step) -> PhysCamera:
        if isinstance(view, int):
            view = self.views[view]
        elif isinstance(view, str):
            pass
        else:
            raise ValueError(f"view must be an integer or a string, but got {view} ({type(view)})")
        return self.cameras[view][step]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # modulate idx to get view and step
        idx = idx % self.length
        view_id = idx // len(self.steps)
        view = self.views[view_id]
        step = idx % len(self.steps)
        
        return self.cameras[view][step]


class VideoDataset(CameraDataset):
    def __init__(self, cfg: DictConfig, readCameras=True):
        super().__init__(cfg, readCameras=False)
        self.device = cfg.device

        self._init_x = None
        self._init_v = None
        self._velocity_opt = None
        self._velocity_sch = None
        if readCameras:
            self.readCameras(cfg)
    
    def get_init_material_data(self):
        init_C = torch.zeros(self._init_x.shape[0], 3, 3).to(self._init_x.device)
        init_F = torch.eye(3).unsqueeze(0).expand(self._init_x.shape[0], 3, 3).to(self._init_x.device)
        init_S = torch.zeros(self._init_x.shape[0], 3, 3).to(self._init_x.device)
        return self.get_init_x, self.get_init_v, init_C, init_F, init_S
    
    @property
    def getVelocityOptimizer(self) -> torch.optim.Optimizer:
        return self._velocity_opt
    
    @property
    def getVelocityScheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        return self._velocity_sch
    
    @property
    def get_init_x(self):
        return self._init_x
    
    @property
    def get_init_v(self):
        # global init_v
        if self._init_v.ndim == 1:
            return self._init_v.unsqueeze(0).expand(self._init_x.shape[0], -1)
        # custom init_v
        elif self._init_v.ndim == 2:
            return self._init_v
    
    def export_init_x_and_v(self, path):
        data = {'init_x': self.get_init_x.cpu(), 'init_v': self.get_init_v.cpu()}
        torch.save(data, path)
        print(f'Saved initial particle data (`x` and `v`) to {path}')
    
    def set_init_x_and_v(self, init_x: Union[NDArray | torch.Tensor], init_v: Optional[Union[NDArray | torch.Tensor]]=None):
        if isinstance(init_x, np.ndarray):
            self._init_x = torch.from_numpy(init_x).to(self.device).float()
        elif isinstance(init_x, torch.Tensor):
            self._init_x = init_x.to(self.device).float()
        else:
            raise ValueError(f"init_x must be a numpy array or a torch tensor, but got {type(init_x)}")

        if init_v is None:
            init_v = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            init_v = torch.from_numpy(init_v)
            self._init_v = nn.Parameter(init_v.to(self.device), requires_grad=True)
        else:
            if isinstance(init_v, np.ndarray):
                self._init_v = torch.from_numpy(init_v).to(self.device).float()
            elif isinstance(init_v, torch.Tensor):
                self._init_v = init_v.requires_grad_(False).to(self.device).float()
            else:
                raise ValueError(f"init_v must be a numpy array or a torch tensor, but got {type(init_v)}")
    
    def init_velocity_optimizer(self, optimizer: torch.optim.Optimizer, lr: float):
        self._velocity_opt = optimizer([self._init_v], lr=lr)
    
    def init_velocity_scheduler(self, scheduler_config: DictConfig, init_lr: float):
        self._velocity_sch = fetch_scheduler(scheduler_config).get_scheduler(self._velocity_opt, init_lr)
    
    def free_velocity_optimizer(self):
        self._velocity_opt = None
    
    def free_velocity_scheduler(self):
        self._velocity_sch = None
    
    def freeze_velocity(self):
        self._init_v.requires_grad = False


if __name__ == "__main__":
    cfg_file = "experiments/config/debug_video_dataset.yaml"
    config = OmegaConf.load(cfg_file)
    config = DictConfig(config)
    vd = VideoDataset(config)

    print(len(vd))
    for i in range(len(vd)):
        print(vd[i].image_name, "--", vd[i].view, "--", vd[i].step)
    import time
    time.sleep(100)
