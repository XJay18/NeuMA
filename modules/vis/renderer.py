import torch
import numpy as np
from viser import ViserServer
from nerfview import CameraState
from omegaconf import OmegaConf, DictConfig
from typing_extensions import Literal

from modules.vis.viewer import DynamicViewer
from modules.vis.neuma_instance import NeuMAInstance


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class VisManager(metaclass=Singleton):
    _servers = {}


def get_server(port: int | None = None) -> ViserServer:
    manager = VisManager()
    if port is None:
        avail_ports = list(manager._servers.keys())
        port = avail_ports[0] if len(avail_ports) > 0 else 8890
    if port not in manager._servers:
        manager._servers[port] = ViserServer(port=port, verbose=False)
    return manager._servers[port]


class Renderer:
    def __init__(
        self,
        model: NeuMAInstance,
        work_dir: str,              # logging
        port: int | None = None,
        up_axis: Literal["x", "y", "z"] = "y",
    ):

        self.model = model
        self.num_frames = len(model.gaus)
        render_dt = model.cfg.sim.dt * model.cfg.skip_frames

        self.work_dir = work_dir

        self.viewer = None
        if port is not None:
            server = get_server(port=port)
            self.viewer = DynamicViewer(
                server, self.render_fn, self.simulate_fn,
                self.num_frames, render_dt, model.cfg.objects,
                work_dir, mode="rendering", up_axis=up_axis
            )

    @staticmethod
    def init_from_config_file(
        path: str, **kwargs
    ) -> "Renderer":
        print(f"Loading config file from {path} ...")
        cfg = OmegaConf.load(path)
        cfg.update({
            "eval_steps": kwargs.pop("eval_steps", 400),
            "skip_frames": kwargs.pop("skip_frames", 5),
        })
        model = NeuMAInstance(cfg)
        renderer = Renderer(model, **kwargs)
        return renderer

    @torch.inference_mode()
    def render_fn(self, camera_state: CameraState, img_wh: tuple[int, int]):
        if self.viewer is None:
            return np.full((img_wh[1], img_wh[0], 3), 255, dtype=np.uint8)

        t = int(self.viewer._playback_guis[0].value)
        img = self.model.render(t, camera_state, img_wh)
        img = (img.cpu().numpy() * 255.0).astype(np.uint8)
        return img

    @torch.inference_mode()
    def simulate_fn(self, cfg_objects: DictConfig):
        self.model.simulate(cfg_objects)
        return True
