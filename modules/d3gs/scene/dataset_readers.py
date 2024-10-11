#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import json
import numpy as np
from PIL import Image
from typing import NamedTuple
from modules.d3gs.scene.colmap_loader import (
    read_extrinsics_text, read_intrinsics_text, qvec2rotmat,
    read_extrinsics_binary, read_intrinsics_binary,
    read_points3D_binary, read_points3D_text
)
from modules.d3gs.utils.graphics_utils import getWorld2View2, focal2fov, fov2focal

from plyfile import PlyData, PlyElement
from modules.d3gs.scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class PhysCameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    view: int
    step: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readNeuMASyntheticCameras(path, transformsfile, white_background, extension=".png", init_frame=None, exclude_steps=[-1], used_views=None):
    """Read the camera setup from the NeuMA Synthetic dataset."""
    cam_infos = list()
    subfolder = transformsfile.split(".")[0]
    print(f"Reading NeuMA Synthetic [{subfolder}] Cameras with init_frame={init_frame}...")

    # check how many views do we have automatically
    views = set()
    steps = set()
    data = os.listdir(os.path.join(path, subfolder))
    for d in data:
        view = str(d.rsplit("_", 1)[0])
        if used_views is None or view in used_views:
            views.add(view)
        step = int(d.rsplit("_", 1)[1].split(".")[0])
        if step not in exclude_steps:
            steps.add(step)
    views = sorted(list(views))
    steps = sorted(list(steps))
    print(f"Views found: {views if len(views) < 20 else views[:20]} {'' if len(views) < 20 else f'#all: {len(views)} ...'}\n"
          f"Steps found: {steps if len(steps) < 20 else steps[:20]} {'' if len(steps) < 20 else f'#all: {len(steps)} ...'}")

    idx = 0
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        meta_info = dict()
        for entry in contents:
            file_path = entry.pop("file_path")
            meta_info[file_path] = entry
        # only read the first frame if `init_frame` is set
        steps = [init_frame] if init_frame is not None else steps
        for view in views:
            for step in steps:
                file_path_to_fetch = f"./{subfolder}/{view}_{step:03d}{extension}"

                assert file_path_to_fetch in meta_info, f"File {file_path_to_fetch} not found in meta_info!"

                # NeRF 'transform_matrix' is a camera-to-world transform
                c2w = np.array(meta_info[file_path_to_fetch]["c2w"])
                if c2w.shape[0] == 3: # (3, 4)
                    c2w = np.concatenate([c2w, np.array([[0, 0, 0, 1]])], axis=0)
                # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
                c2w[:3, 1:3] *= -1

                # get the world-to-camera transform and set R, T
                w2c = np.linalg.inv(c2w)
                R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
                T = w2c[:3, 3]

                image_name = os.path.join(path, file_path_to_fetch)
                image = Image.open(image_name)

                im_data = np.array(image.convert("RGBA"))

                bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

                norm_data = im_data / 255.0
                arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
                image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

                intrinsics = meta_info[file_path_to_fetch]["intrinsic"]
                focalx = intrinsics[0][0]
                focaly = intrinsics[1][1]
                FovX = focal2fov(focalx, image.size[0])
                FovY = focal2fov(focaly, image.size[1])

                cam_infos.append(
                    PhysCameraInfo(
                        uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                        image_path=image_name, image_name=image_name,
                        width=image.size[0], height=image.size[1],
                        view=view, step=step
                    )
                )
                idx += 1

    return {'cam_infos': cam_infos, 'views': views, 'steps': steps}


def readRealCaptureCameras(
    path,
    white_background,
    extension=".jpg",
    width=1920,             # hardcoded for RealCapture
    height=1080,            # hardcoded for RealCapture
    init_frame=None,
    exclude_steps=[-1],
    used_views=None,
    read_mask_only=False    # if the images are binary masks
):
    import cv2


    """Read the camera setup from the RealCapture dataset."""
    cam_infos = list()
    print(f"Reading RealCapture Cameras...")

    cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)

    K = np.array([
        [cam_intrinsics[1].params[0] * width / 4752, 0, width / 2],
        [0, cam_intrinsics[1].params[1] * height / 2672, height / 2],
        [0, 0, 1]
    ])

    FovY = focal2fov(K[0][0], height)
    FovX = focal2fov(K[1][1], width)

    with open(os.path.join(path, 'cameras_calib.json'), 'r') as f:
        cam_calib = json.load(f)

    # check how many views do we have automatically
    views = set()
    steps = set()
    data = os.listdir(os.path.join(path, "dynamics"))
    for d in data:
        view = str(d.rsplit("_", 1)[0])
        if used_views is None or view in used_views:
            views.add(view)
        step = int(d.rsplit("_", 1)[1].split(".")[0])
        if step not in exclude_steps:
            steps.add(step)
    views = sorted(list(views))
    steps = sorted(list(steps))
    print(f"Views found: {views if len(views) < 20 else views[:20]} {'' if len(views) < 20 else f'#all: {len(views)} ...'}\n"
          f"Steps found: {steps if len(steps) < 20 else steps[:20]} {'' if len(steps) < 20 else f'#all: {len(steps)} ...'}")

    idx = 0
    steps = [init_frame] if init_frame is not None else steps
    for view in views:
        rvecs = cam_calib[view]["rvecs"]
        tvecs = cam_calib[view]["tvecs"]
        rot_mat, _ = cv2.Rodrigues(np.array(rvecs))
        R = np.transpose(rot_mat)
        T = np.array(tvecs).reshape(3)

        for step in steps:
            file_path_to_fetch = f"./dynamics/{view}_{step}{extension}"
            image_name = os.path.join(path, file_path_to_fetch)
            mask_name = image_name.replace("/dynamics/", "/dynamic_masks/").replace(extension, ".png")

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            if read_mask_only:
                mask = Image.open(mask_name)
                image = Image.fromarray(np.repeat(np.array(mask)[:, :, np.newaxis], 3, axis=-1), "RGB")
            else:
                image = Image.open(image_name)
                im_data = np.array(image)
                mask = Image.open(mask_name)
                mask = np.array(mask)[:, :, np.newaxis] / 255.0
                arr = (im_data / 255.0) * mask + bg * (1 - mask)
                image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

            cam_infos.append(
                PhysCameraInfo(
                    uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                    image_path=image_name, image_name=image_name,
                    width=width, height=height,
                    view=view, step=step
                )
            )
            idx += 1

    return {'cam_infos': cam_infos, 'views': views, 'steps': steps}
