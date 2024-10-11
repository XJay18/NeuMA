import sys
import torch
from torch import nn
from torch.optim import RAdam
from torchvision.utils import save_image
from torch.nn.utils import clip_grad_norm_
from tensorboardX import SummaryWriter
import random
import argparse
import warp as wp
import numpy as np
from pathlib import Path
from typing import Optional
from natsort import natsorted
from omegaconf import DictConfig, OmegaConf
from tqdm.autonotebook import tqdm, trange

from modules.nclaw.utils import (
    mkdir,
    denormalize_points_helper_func
)
from modules.nclaw.sim import (
    MPMModelBuilder,
    MPMCacheDiffSim,
    MPMStaticsInitializer,
    MPMInitData,
)
from modules.nclaw.material import (
    InvariantFullMetaElasticity,
    InvariantFullMetaPlasticity
)

from modules.d3gs.utils.loss_utils import l1_loss, l2_loss
from modules.d3gs.scene.gaussian_model import GaussianModel
from modules.tune.dataset.neuma_dataset import VideoDataset
from modules.tune.scheduler import fetch_scheduler
from modules.tune.utils import (
    Logger, Timer,
    get_warp_device,
    compute_bindings_xyz,
    compute_bindings_F,
    prepare_simulation_data,
    diff_rasterization
)

ASSETS_PATH = Path(__file__).parent / "assets"
EPS = 6e-7
PIXEL_LOSSES = {
    "l1": l1_loss,
    "l2": l2_loss
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", type=str, required=True,
        help="Path to the config file."
    )
    args = parser.parse_args()
    return args


def optimize_init_velocity(
    cfg: DictConfig,
    gaussians: GaussianModel,
    dataset: VideoDataset,
    elasticity: InvariantFullMetaElasticity,
    plasticity: InvariantFullMetaPlasticity,
    bindings: torch.Tensor,
    background: torch.Tensor,
    tune_root: Path,
    force_mask_data: bool,
    debug: bool,
    debug_raster_root: Optional[Path] = None
):
    already_optimized = (tune_root / 'init.pt').exists()

    if already_optimized:

        # read initial velocity from the checkpoint
        print("\n===================================")
        print(f'Loading initial velocity from checkpoint ...\n')
        init_x_and_v = torch.load(tune_root / 'init.pt', map_location="cpu")
        dataset.set_init_x_and_v(init_x=init_x_and_v['init_x'], init_v=init_x_and_v['init_v'])
    else:

        # optimize initial velocity
        print("\n===================================")
        print(f'Optimizing initial velocity ...\n')

        torch.cuda.empty_cache()
        wp_device = get_warp_device(background.device)

        nframes = cfg.velocity.num_frames
        substeps = cfg.velocity.substeps
        used_views = dataset.views if cfg.velocity.get("views", "all") == 'all' else cfg.velocity.views
        used_views = sorted(used_views)
        nsteps = nframes * substeps
        print(f"[velocity] Simulation steps: {nsteps}")
        print(f"[velocity] Using substeps: {substeps}")
        print(f"[velocity] Using views: {used_views}")

        # warp

        cfg.sim.eps = EPS                                                   # NOTE: manually setting !!!
        model = MPMModelBuilder().parse_cfg(cfg.sim).finalize(wp_device, requires_grad=True)
        sim = MPMCacheDiffSim(model, nsteps)
        statics_initializer = MPMStaticsInitializer(model)
        cfg.particle_data.span = [0, nsteps]                                # NOTE: manually setting !!!
        cfg.particle_data.shape.name = cfg.sim_data_name + "/particles"     # NOTE: manually setting !!!
        init_data = MPMInitData.get(cfg.particle_data)
        pixel_loss = PIXEL_LOSSES[cfg.velocity.get("pixel_loss", "l2")]
        print(f"[velocity] Using dt: {model.constant.dt}")
        print(f"[velocity] Using eps: {model.constant.eps}")
        print(f"[velocity] Using clip bound: {init_data.clip_bound}")
        print(f"[velocity] Using mask data: {force_mask_data}")
        print(f"[velocity] Using pixel loss: {cfg.velocity.get('pixel_loss', 'l2')}")

        #    -- copy particle data from MPMInitData to video dataset
        dataset.set_init_x_and_v(init_x=init_data.pos)
        dataset.init_velocity_optimizer(RAdam, lr=cfg.velocity.lr)
        dataset.init_velocity_scheduler(cfg.velocity.scheduler, init_lr=cfg.velocity.lr)

        #    -- assertion
        assert init_data.pos.shape[0] == dataset.get_init_x.shape[0], \
            f"Shape mismatch: init_data {init_data.pos.shape[0]} dataset {dataset.get_init_x.shape[0]}"

        #    -- good to go
        statics_initializer.add_group(init_data)
        statics = statics_initializer.finalize()

        # fine-tune the initial velocity

        for epoch in trange(1, cfg.velocity.num_epochs + 1, position=1):

            x, init_v, C, F, _ = dataset.get_init_material_data()
            assert init_v.requires_grad, "init_v should require grad"
            dataset.getVelocityOptimizer.zero_grad()

            de_x = denormalize_points_helper_func(x, init_data.size, init_data.center)

            de_x_prev = de_x.clone().detach()
            g_prev = gaussians.get_xyz.clone().detach()

            v = init_v + 0.0
            loss_rgb = 0.0

            for it in tqdm(range(nsteps), position=0, leave=False):
                stress = elasticity(F)
                x, v, C, F = sim(statics, it, x, v, C, F, stress)
                F = plasticity(F)

                # rasterization here
                if (it + 1) % substeps == 0:
                    cur_step = (it + 1) // substeps
                    cur_step = dataset.steps[cur_step]
                    de_x = denormalize_points_helper_func(x, init_data.size, init_data.center)

                    means3D = compute_bindings_xyz(de_x, de_x_prev, g_prev, bindings)
                    deform_grad = compute_bindings_F(F, bindings)

                    for view in used_views:
                        render = diff_rasterization(
                            means3D, deform_grad, gaussians,
                            dataset.getCameras(view, cur_step), background,
                            scaling_modifier=cfg.gaussian.get('scaling_modifier', 1.0),
                            force_mask_data=force_mask_data
                        )
                        gt = dataset.getCameras(view, cur_step).original_image.to(x.device)
                        # accumulate rgb loss
                        loss_rgb += pixel_loss(render, gt)

                    de_x_prev = de_x.clone().detach()
                    g_prev = means3D.clone().detach()

                    if debug and (epoch == 1 or epoch == cfg.velocity.num_epochs // 2 or epoch == cfg.velocity.num_epochs):
                        with torch.no_grad():
                            for view in dataset.views:
                                # save debug images
                                if view in cfg.get("debug_views", list()) and cur_step % cfg.velocity.get("debug_image_steps", 5) == 0:
                                    # rasterize deformed kernels
                                    db_render = diff_rasterization(
                                        means3D, deform_grad, gaussians,
                                        dataset.getCameras(view, cur_step), background,
                                        scaling_modifier=cfg.gaussian.get('scaling_modifier', 1.0),
                                        force_mask_data=force_mask_data
                                    )
                                    db_gt = dataset.getCameras(view, cur_step).original_image.to(x.device)
                                    # concat render and gt
                                    cat_img = torch.cat([db_render, db_gt], dim=2)
                                    save_image(cat_img, debug_raster_root / f'({cur_step})_{epoch}_{it}_{view}.png')

            if cfg.velocity.get("lambda_reg") is not None and epoch > int(0.1 * cfg.velocity.num_epochs):
                # warm up loss_reg for the first 10% of epochs
                if cfg.velocity.get("reg_all", False):
                    loss_reg = cfg.velocity.lambda_reg * init_v.abs().mean()
                else:
                    loss_reg = cfg.velocity.lambda_reg * (init_v[:, 0].abs().mean() + init_v[:, 2].abs().mean()) / 2. # prior on x-z velocity
            else:
                loss_reg = torch.zeros_like(loss_rgb)
            loss = loss_rgb + loss_reg

            loss.backward()
            dataset.getVelocityOptimizer.step()

            with torch.no_grad():
                # Progress bar
                msgs = [
                    f"Epoch {epoch}/{cfg.velocity.num_epochs}",
                    f"L rgb: {loss_rgb.item():.4e}, reg: {loss_reg.item():.4e}",
                    f"lr: {dataset.getVelocityOptimizer.param_groups[0]['lr']:.4f}",
                    f"init_v: {dataset._init_v.detach().cpu()}",
                ]

                msg = ' | '.join(msgs)
                tqdm.write(f"[{msg}]")
            
            dataset.getVelocityScheduler.step()

        # clean for next stage 'fine-tune the constitutive models'

        dataset.freeze_velocity()
        dataset.export_init_x_and_v(tune_root / 'init.pt')
        dataset.free_velocity_optimizer()
        dataset.free_velocity_scheduler()
        wp.synchronize()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    print(f'\nInitial velocity obtained: {dataset.get_init_v.mean(0)}.')
    print("===================================")


def finetune_constitutive(
    cfg: DictConfig,
    gaussians: GaussianModel,
    dataset: VideoDataset,
    elasticity: InvariantFullMetaElasticity,
    plasticity: InvariantFullMetaPlasticity,
    bindings: torch.Tensor,
    background: torch.Tensor,
    tune_root: Path,
    force_mask_data: bool,
    debug: bool,
    debug_raster_root: Path
):
    print("\n===================================")
    print(f'Finetuning the constitutive models ...\n')

    torch.cuda.empty_cache()
    wp_device = get_warp_device(background.device)

    # init

    writer = SummaryWriter(tune_root.parent, purge_step=0)

    timer = Timer()

    nframes = cfg.constitution.num_frames
    substeps = cfg.constitution.substeps
    used_views = dataset.views if cfg.constitution.get("views", "all") == 'all' else cfg.constitution.views
    used_views = sorted(used_views)
    nsteps = nframes * substeps
    print(f"[constitutive] Simulation steps: {nsteps}")
    print(f"[constitutive] Using substeps: {substeps}")
    print(f"[constitutive] Using views: {used_views}")

    # warp

    cfg.sim.eps = EPS                                               # NOTE: manually setting !!!
    model = MPMModelBuilder().parse_cfg(cfg.sim).finalize(wp_device, requires_grad=True)
    sim = MPMCacheDiffSim(model, nsteps)
    statics_initializer = MPMStaticsInitializer(model)
    cfg.particle_data.span = [0, nsteps]                            # NOTE: manually setting !!!
    cfg.particle_data.shape.name = cfg.sim_data_name + "/particles" # NOTE: manually setting !!!
    init_data = MPMInitData.get(cfg.particle_data)
    pixel_loss = PIXEL_LOSSES[cfg.constitution.get("pixel_loss", "l2")]
    print(f"[constitutive] Using dt: {model.constant.dt}")
    print(f"[constitutive] Using eps: {model.constant.eps}")
    print(f"[constitutive] Using clip bound: {init_data.clip_bound}")
    print(f"[constitutive] Using mask data: {force_mask_data}")
    print(f"[constitutive] Using pixel loss: {cfg.constitution.get('pixel_loss', 'l2')}")

    #    -- assertion
    assert init_data.pos.shape[0] == dataset.get_init_x.shape[0], \
        f"Shape mismatch: init_data {init_data.pos.shape[0]} dataset {dataset.get_init_x.shape[0]}"

    #    -- good to go
    statics_initializer.add_group(init_data)
    statics = statics_initializer.finalize()

    # prepare for 'fine-tune the constitutive models'

    #    -- init lora
    elasticity.init_lora_layers(r=cfg.constitution.lora.r, lora_alpha=cfg.constitution.lora.alpha)
    plasticity.init_lora_layers(r=cfg.constitution.lora.r, lora_alpha=cfg.constitution.lora.alpha)

    #    -- check whether to load fine-tuned weights
    if cfg.resume:
        previous_loras = list()
        for ckpt in tune_root.glob('*_lora.pt'):
            previous_loras.append(ckpt)
        if len(previous_loras) > 0:
            previous_loras = sorted(previous_loras)
            print(f'Find fine-tuned lora weights from {previous_loras[-1]}')
            ckpt = torch.load(previous_loras[-1], map_location=background.device)
            elasticity.load_state_dict(ckpt['elasticity'], strict=False)
            plasticity.load_state_dict(ckpt['plasticity'], strict=False)
            print(f'Loaded fine-tuned lora weights')

    #    -- only tune lora weights
    elasticity.freeze_all_except_lora()
    plasticity.freeze_all_except_lora()

    # optimizer

    e_opt = RAdam(
        filter(lambda p: p.requires_grad, elasticity.parameters()),
        lr=cfg.constitution.elasticity_lr, weight_decay=cfg.constitution.elasticity_wd
    )
    e_sch = fetch_scheduler(cfg.constitution.elasticity_scheduler).get_scheduler(e_opt, cfg.constitution.elasticity_lr)

    p_opt = RAdam(
        filter(lambda p: p.requires_grad, plasticity.parameters()),
        lr=cfg.constitution.plasticity_lr, weight_decay=cfg.constitution.plasticity_wd
    )
    p_sch = fetch_scheduler(cfg.constitution.plasticity_scheduler).get_scheduler(p_opt, cfg.constitution.plasticity_lr)

    # fine-tune the constitutive models

    for epoch in tqdm(range(1, cfg.constitution.num_epochs + 1), position=0, leave=False):

        # train from the initial step
        x, v, C, F, _ = dataset.get_init_material_data()
        assert not v.requires_grad, "init_v should not require grad for finetuning constitutive models"

        de_x = denormalize_points_helper_func(x, init_data.size, init_data.center)

        de_x_prev = de_x.clone().detach()
        g_prev = gaussians.get_xyz.clone().detach()

        loss_rgb = 0.0

        # warm-up lr
        if cfg.constitution.warmup_step != 0 and epoch <= cfg.constitution.warmup_step:
            e_lr = cfg.constitution.elasticity_lr * float(epoch) / cfg.constitution.warmup_step
            for param_group in e_opt.param_groups:
                param_group['lr'] = e_lr
            p_lr = cfg.constitution.plasticity_lr * float(epoch) / cfg.constitution.warmup_step
            for param_group in p_opt.param_groups:
                param_group['lr'] = p_lr

        if cfg.constitution.lambda_max_decay > 0:
            coeff_max_decay = 1. / cfg.constitution.lambda_max_decay
            ratio = min(coeff_max_decay * epoch / cfg.constitution.num_epochs, 1.0)
        else:
            ratio = 1.0
        decay_rate = cfg.constitution.decay_init + (cfg.constitution.decay_final - cfg.constitution.decay_init) * ratio

        for it in tqdm(range(nsteps), position=0, leave=False):

            stress = elasticity(F)
            x, v, C, F = sim(statics, it, x, v, C, F, stress)
            F = plasticity(F)

            # rasterization here
            if (it + 1) % substeps == 0:
                cur_step = (it + 1) // substeps
                cur_frame = dataset.steps[cur_step]
                # NOTE!!! modified here, do not compute loss for certain steps
                if cur_frame in cfg.constitution.get("exclude_steps", list()):
                    continue
                de_x = denormalize_points_helper_func(x, init_data.size, init_data.center)

                means3D = compute_bindings_xyz(de_x, de_x_prev, g_prev, bindings)
                deform_grad = compute_bindings_F(F, bindings)

                for view in used_views:
                    # rasterize deformed kernels
                    render = diff_rasterization(
                        means3D, deform_grad, gaussians,
                        dataset.getCameras(view, cur_frame), background,
                        scaling_modifier=cfg.gaussian.get('scaling_modifier', 1.0),
                        force_mask_data=force_mask_data
                    )
                    gt = dataset.getCameras(view, cur_frame).original_image.to(x.device)
                    # accumulate loss
                    rollout_decay_rate = decay_rate ** ((cur_step - 1) // cfg.constitution.decay_steps)
                    loss_rgb += rollout_decay_rate * pixel_loss(render, gt)

                de_x_prev = de_x.clone().detach()
                g_prev = means3D.clone().detach()

                if debug and (epoch == 1 or epoch % 100 == 0 or epoch == cfg.constitution.num_epochs):
                    with torch.no_grad():
                        for view in dataset.views:
                            # save debug images
                            if view in cfg.get("debug_views", list()) and cur_frame % cfg.constitution.get("debug_image_steps", 100) == 0:
                                # rasterize deformed kernels
                                db_render = diff_rasterization(
                                    means3D, deform_grad, gaussians,
                                    dataset.getCameras(view, cur_frame), background,
                                    scaling_modifier=cfg.gaussian.get('scaling_modifier', 1.0),
                                    force_mask_data=force_mask_data
                                )
                                db_gt = dataset.getCameras(view, cur_frame).original_image.to(x.device)
                                # concat render and gt
                                cat_img = torch.cat([db_render, db_gt], dim=2)
                                save_image(
                                    cat_img, debug_raster_root / 
                                    f'({cur_frame})_{it}_{epoch:03d}_{view}.png'
                                )
        loss = loss_rgb
        loss.backward()

        try:
            elasticity_grad_norm = clip_grad_norm_(
                elasticity.parameters(),
                max_norm=cfg.constitution.elasticity_grad_max_norm,
                error_if_nonfinite=True)
            e_opt.step()

            plasticity_grad_norm = clip_grad_norm_(
                plasticity.parameters(),
                max_norm=cfg.constitution.plasticity_grad_max_norm,
                error_if_nonfinite=True)
            p_opt.step()

        except Exception as e:
            import logging

            print("**************************")
            print(f"epoch: {epoch}, it: {it}")
            print(f"loss: {loss:.7f}, loss_rgb: {loss_rgb:.7f}")

            e_grads = [torch.norm(p.grad, p=2) for p in elasticity.parameters() if p.grad is not None]
            print(e_grads)

            p_grads = [torch.norm(p.grad, p=2) for p in plasticity.parameters() if p.grad is not None]
            print(p_grads)
            print("**************************")

            logging.exception(e)
            exit(1)

        with torch.no_grad():
            # Progress bar
            msgs = [
                f"Epoch {epoch}/{cfg.constitution.num_epochs}",
                f"L rgb: {loss_rgb.item():.4e}",
                f"e-lr: {e_opt.param_groups[0]['lr']:.2e}",
                f"e-gd: {elasticity_grad_norm:.2e}",
                f"p-lr: {p_opt.param_groups[0]['lr']:.2e}",
                f"p-gd: {plasticity_grad_norm:.2e}",
                f"decay: {decay_rate:.2f}",
                f"elp: {timer.measure()}",
                f"est: {timer.measure(epoch / cfg.constitution.num_epochs)}",
            ]

            msg = ' | '.join(msgs)
            tqdm.write(f"[{msg}]")

            writer.add_scalar('lr/elasticity', e_opt.param_groups[0]['lr'], epoch)
            writer.add_scalar('grad_norm/elasticity', elasticity_grad_norm, epoch)
            writer.add_scalar('lr/plasticity', p_opt.param_groups[0]['lr'], epoch)
            writer.add_scalar('grad_norm/plasticity', plasticity_grad_norm, epoch)
            writer.add_scalar('loss/rgb', loss_rgb.item(), epoch)
            writer.add_scalar('lr/decay', decay_rate, epoch)

            if epoch == 1 or epoch % 10 == 0 or epoch == cfg.constitution.num_epochs:
                # save lora weights
                torch.save({
                    'elasticity': elasticity.lora_state_dict(),
                    'plasticity': plasticity.lora_state_dict(),
                    'loss': loss_rgb.item(),
                }, tune_root / f'{epoch:04d}_lora.pt')

                lora_files = natsorted([f.as_posix() for f in tune_root.glob('*_lora.pt')])
                if len(lora_files) > cfg.constitution.get('num_lora_ckpts', 3):
                    Path(lora_files[0]).unlink()

        if cfg.constitution.warmup_step == 0 or epoch > cfg.constitution.warmup_step:
            e_sch.step()
            p_sch.step()

    writer.close()
    print(f'\nFinetuning ends.')
    print("===================================")


def finetune(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # init

    seed = cfg.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    debug = cfg.debug

    wp.init()
    wp.config.verify_cuda = True
    wp.ScopedTimer.enabled = False
    wp.set_module_options({'fast_math': False})

    torch_device = torch.device(f'cuda:{cfg.gpu}')
    torch.backends.cudnn.benchmark = True

    # background

    force_mask_data = cfg.video_data.data.get("read_mask_only", False)
    if force_mask_data:
        # Force to use black background when loading mask data
        cfg.video_data.data.white_background = False
        print(f"[Warning] Force to use black background when loading mask data")

    background = (
        torch.tensor([1, 1, 1], dtype=torch.float32, device=torch_device)
        if cfg.video_data.data.get("white_background", False)   # default to black background
        else torch.tensor([0, 0, 0], dtype=torch.float32, device=torch_device)
    )

    # path

    root: Path = Path(cfg.root)
    exp_root: Path = root / cfg.name
    mkdir(exp_root, resume=cfg.resume, overwrite=cfg.overwrite)
    OmegaConf.save(cfg, exp_root / 'config.yaml', resolve=True)
    sys.stdout = Logger(exp_root / 'log.txt')

    tune_root: Path = exp_root / 'finetune' # fine-tuned weights
    tune_root.mkdir(exist_ok=True)

    data_root: Path = ASSETS_PATH / cfg.sim_data_name
    data_root.mkdir(exist_ok=True)

    if debug:
        debug_root: Path = exp_root / 'debug'
        debug_root.mkdir(exist_ok=True)

        debug_velocity_root: Path = debug_root / 'raster_velocity'
        debug_velocity_root.mkdir(exist_ok=True)

        debug_finetune_root: Path = debug_root / 'raster_finetune'
        debug_finetune_root.mkdir(exist_ok=True)

    # data

    if cfg.particle_data.get('particles_path') is not None:

        prepare_simulation_data(
            kernels_path=Path(cfg.gaussian.kernels_path),
            particles_path=Path(cfg.particle_data.particles_path),
            save_dir=data_root,
            sh_degree=cfg.gaussian.sh_degree,
            opacity_thres=cfg.gaussian.opacity_thres,
            particles_downsample_factor=cfg.particle_data.downsample_factor,
            confidence=cfg.gaussian.confidence,
            max_particles=cfg.gaussian.max_particles
        )

    elif cfg.particle_data.get('mesh_path') is not None:

        prepare_simulation_data(
            kernels_path=Path(cfg.gaussian.kernels_path),
            mesh_path=Path(cfg.particle_data.mesh_path),
            mesh_sample_mode=cfg.particle_data.mesh_sample_mode,
            mesh_sample_resolution=cfg.particle_data.mesh_sample_resolution,
            save_dir=data_root,
            sh_degree=cfg.gaussian.sh_degree,
            opacity_thres=cfg.gaussian.opacity_thres,
            particles_downsample_factor=1,
            confidence=cfg.gaussian.confidence,
            max_particles=cfg.gaussian.max_particles
        )

    else:

        raise ValueError("Either 'particles_path' or 'mesh_path' must be provided in configuration.")

    #    -- video data
    cfg.video_data.device = f"cuda:{cfg.gpu}"       # NOTE: manually setting !!!
    dataset = VideoDataset(cfg.video_data)

    #    -- binding data
    bind_data = torch.load(data_root / 'bindings.pt')
    # bindings: torch.Tensor = bind_data['bindings'].to(torch_device).float()
    bindings: torch.Tensor = torch.sparse_coo_tensor(
        bind_data['bindings_ind'], bind_data['bindings_val'], bind_data['bindings_size']
    ).to(torch_device).float()
    n_particles: torch.Tensor = bind_data['n_particles'].to(torch_device).float()

    has_particles = n_particles > 0
    print(f'Exp name [{cfg.name}]')
    print(f'Using data name [{cfg.sim_data_name}]')
    print(f'#Gaussians with particle bindings: {has_particles.sum()}')
    print(f'#Avg particles: {n_particles.mean()}')
    print(f'#Max particles: {n_particles.max()}')

    # gaussians

    gaussians = GaussianModel(cfg.gaussian.sh_degree)
    gaussians.load_ply(data_root / f"kernels.ply", requires_grad=False)

    # material

    elasticity: nn.Module = InvariantFullMetaElasticity(cfg.constitution.elasticity)
    elasticity.to(torch_device)
    elasticity.requires_grad_(True)
    elasticity.train(True)

    plasticity: nn.Module = InvariantFullMetaPlasticity(cfg.constitution.plasticity)
    plasticity.to(torch_device)
    plasticity.requires_grad_(True)
    plasticity.train(True)

    #    -- load pretrained weights
    ckpt_path = cfg.pretrained_ckpt
    pretrained = torch.load(ckpt_path, map_location=torch_device)
    elasticity.load_state_dict(pretrained['elasticity'])
    plasticity.load_state_dict(pretrained['plasticity'])
    print(f'Loaded pretrained weights from {ckpt_path}')

    # fine-tune the initial velocity

    #   -- check whether the initial velocity has been optimized and stored in data_root
    if (data_root / 'init.pt').exists() and not (tune_root / 'init.pt').exists():
        (tune_root / 'init.pt').symlink_to(data_root / 'init.pt')
        print(
            f'Found initial velocity from {data_root / "init.pt"}.\n'
            f'Linked {data_root / "init.pt"} to {tune_root / "init.pt"}'
        )

    optimize_init_velocity(
        cfg,
        gaussians,
        dataset,
        elasticity,
        plasticity,
        bindings,
        background,
        tune_root,
        force_mask_data,
        debug,
        debug_velocity_root
    )

    # fine-tune the constitutive models

    finetune_constitutive(
        cfg,
        gaussians,
        dataset,
        elasticity,
        plasticity,
        bindings,
        background,
        tune_root,
        force_mask_data,
        debug,
        debug_finetune_root
    )


if __name__ == "__main__":
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    cfg = DictConfig(cfg)
    finetune(cfg)
