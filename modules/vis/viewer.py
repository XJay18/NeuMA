from pathlib import Path
from omegaconf import DictConfig
from typing import Callable, Literal, Optional, Tuple, Union

import time
import viser
import numpy as np
from jaxtyping import Float32, UInt8
from nerfview import CameraState, Viewer
from viser import Icon, ViserServer

from modules.vis.render_panel import populate_render_tab
from modules.vis.playback_panel import add_gui_playback_group
from modules.vis.simulation_panel import add_gui_object_group

NeuMA2PATH = {
    "bouncy": "experiments/logs/bouncyball-v1/finetune/1000_lora.pt",
    "clay": "experiments/logs/claycat-v1/finetune/1000_lora.pt",
    "honey": "experiments/logs/honeybottle-v1/finetune/1000_lora.pt",
    "jelly": "experiments/logs/jellyduck-v1/finetune/1000_lora.pt",
    "rubber": "experiments/logs/rubberpawn-v1/finetune/1000_lora.pt",
    "sand": "experiments/logs/sandfish-v1/finetune/1000_lora.pt",
}

NeuMA2BASE = {
    "bouncy": "experiments/base_models/jelly_0300.pt",
    "clay": "experiments/base_models/plasticine_0300.pt",
    "honey": "experiments/base_models/sand_0300.pt",
    "jelly": "experiments/base_models/jelly_0300.pt",
    "rubber": "experiments/base_models/plasticine_0300.pt",
    "sand": "experiments/base_models/sand_0300.pt",
}

PATH2NeuMA = {v: k for k, v in NeuMA2PATH.items()}

class DynamicViewer(Viewer):
    def __init__(
        self,
        server: ViserServer,
        render_fn: Callable[
            [CameraState, Tuple[int, int]],
            Union[
                UInt8[np.ndarray, "H W 3"],
                Tuple[UInt8[np.ndarray, "H W 3"], Optional[Float32[np.ndarray, "H W"]]],
            ],
        ],
        simulate_fn: Callable[
            [DictConfig], bool
        ],
        num_frames: int,
        render_dt: float,
        sim_objects: Tuple[DictConfig],
        work_dir: str,
        mode: Literal["rendering", "training"] = "rendering",
        up_axis: Literal["x", "y", "z"] = "y",
    ):
        self.simulate_fn = simulate_fn
        self.num_frames = num_frames
        self.render_dt = render_dt
        self.sim_objects = sim_objects
        self.work_dir = Path(work_dir)
        self._object_guis = list()
        server.scene.set_up_direction(f"+{up_axis}")
        if up_axis == "x":
            grid_plane = "yz"
            self.up_axis = 0
        elif up_axis == "y":
            grid_plane = "xz"
            self.up_axis = 1
        else:
            grid_plane = "xy"
            self.up_axis = 2
        server.scene.add_grid('floor', 1, 1, 1, 1, grid_plane, position=(0.5, 0., 0.5))
        super().__init__(server, render_fn, mode)

    def _define_guis(self):
        super()._define_guis()
        server = self.server
        self._dynamics_folder = server.gui.add_folder("Dynamics")
        with self._dynamics_folder:
            self._playback_guis = add_gui_playback_group(
                server,
                num_frames=self.num_frames,
                render_dt=self.render_dt,
                initial_fps=30.0,
            )
            self._playback_guis[0].on_update(self.rerender)
        
        self._objects_folder = server.gui.add_folder("Objects")
        with self._objects_folder:
            for i, obj in enumerate(self.sim_objects):
                object_guis = add_gui_object_group(
                    server,
                    obj.sim_data_name.split('-')[0],
                    drop_vel=abs(obj.particle_data.vel.lin_vel[1]),
                    weight=obj.constitution.lora.alpha / obj.constitution.lora.r,
                    neuma=PATH2NeuMA[obj.constitution.load_lora],
                )
                self._object_guis.append(object_guis)
            self._simulation_ips_button = server.gui.add_button(
                "Please Wait During Simulating ...", icon=viser.Icon.CLOCK)
            self._simulation_ips_button.visible = False
            self._simulation_ips_button.disabled = True
            self._simulation_not_ready_button = server.gui.add_button(
                "Please 'Pause' the trajectory first :)", icon=viser.Icon.EXCLAMATION_CIRCLE)
            self._simulation_not_ready_button.visible = False
            self._simulation_not_ready_button.disabled = True
            self._simulation_no_objects_button = server.gui.add_button(
                "Please keep at least one object :)", icon=viser.Icon.EXCLAMATION_CIRCLE)
            self._simulation_no_objects_button.visible = False
            self._simulation_no_objects_button.disabled = True
            self._simulation_button = server.gui.add_button("Re-Simulate", icon=viser.Icon.ROBOT)
        
        @self._simulation_button.on_click
        def _(_) -> None:
            # check whether is playing
            if self._playback_guis[3].visible:  # pause visible
                self._simulation_button.visible = not self._simulation_button.visible
                self._simulation_not_ready_button.visible = not self._simulation_not_ready_button.visible
                time.sleep(5)
                self._simulation_not_ready_button.visible = not self._simulation_not_ready_button.visible
                self._simulation_button.visible = not self._simulation_button.visible
            else:
                # track the existance of objects
                something_exists = False
                
                for i, obj in enumerate(self.sim_objects):
                    drop_vel, weight, neuma_path, object_exists = self._object_guis[i]
                    obj.particle_data.vel.lin_vel[self.up_axis] = -drop_vel.value           # NOTE: drop_vel is negative
                    obj.constitution.lora.alpha = weight.value * obj.constitution.lora.r
                    obj.pretrained_ckpt = NeuMA2BASE[neuma_path.value]
                    obj.constitution.load_lora = NeuMA2PATH[neuma_path.value]
                    obj.exists = object_exists.value
                    something_exists = something_exists or object_exists.value
                
                if not something_exists:
                    self._simulation_button.visible = not self._simulation_button.visible
                    self._simulation_no_objects_button.visible = not self._simulation_no_objects_button.visible
                    time.sleep(5)
                    self._simulation_no_objects_button.visible = not self._simulation_no_objects_button.visible
                    self._simulation_button.visible = not self._simulation_button.visible
                    return
                
                self._playback_guis[0].value = 0
                self._simulation_button.visible = not self._simulation_button.visible
                self._simulation_ips_button.visible = not self._simulation_ips_button.visible
                self._playback_guis[1].disabled = not self._playback_guis[1].disabled
                self._playback_guis[2].disabled = not self._playback_guis[2].disabled
                self._playback_guis[4].disabled = not self._playback_guis[4].disabled
                
                print(
                    f'\nRe-simulating with new parameters: '
                    f'Drop Vel {[obj.particle_data.vel.lin_vel[self.up_axis] for obj in self.sim_objects]} '
                    f'Weight {[obj.constitution.lora.alpha for obj in self.sim_objects]} '
                    f'NeuMA {[PATH2NeuMA[obj.constitution.load_lora] for obj in self.sim_objects]} '
                    f'Exists {[obj.exists for obj in self.sim_objects]}'
                )

                self.simulate_fn(self.sim_objects)
                self._playback_guis[4].disabled = not self._playback_guis[4].disabled
                self._playback_guis[2].disabled = not self._playback_guis[2].disabled
                self._playback_guis[1].disabled = not self._playback_guis[1].disabled
                self._simulation_ips_button.visible = not self._simulation_ips_button.visible
                self._simulation_button.visible = not self._simulation_button.visible

        tabs = server.gui.add_tab_group()
        with tabs.add_tab("Render", Icon.CAMERA):
            self.render_tab_state = populate_render_tab(
                server, Path(self.work_dir) / "camera_paths", self._playback_guis[0]
            )
