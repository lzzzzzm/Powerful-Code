import os

import mmcv
import mmengine
import numpy as np
import open3d as o3d
import torch
import matplotlib.pyplot as plt

import cv2 as cv

from mmengine.fileio import get
from mmengine.structures import InstanceData

from visualization import Det3DLocalVisualizer

from mmdet3d.structures.points import BasePoints, get_points_type
from mmdet3d.structures import PointData

from nuscenes.nuscenes import NuScenes
from matplotlib.colors import ListedColormap
palette= [
    [0, 0, 0],          # noise                black
    [255, 120, 50],     # barrier              orange
    [255, 192, 203],    # bicycle              pink
    [255, 255, 0],      # bus                  yellow
    [0, 150, 245],      # car                  blue
    [0, 255, 255],      # construction_vehicle cyan
    [255, 127, 0],      # motorcycle           dark orange
    [255, 0, 0],        # pedestrian           red
    [255, 240, 150],    # traffic_cone         light yellow
    [135, 60, 0],       # trailer              brown
    [160, 32, 240],     # truck                purple
    [255, 0, 255],      # driveable_surface    dark pink
    [139, 137, 137],    # other_flat           dark red
    [75, 0, 75],        # sidewalk             dard purple
    [150, 240, 80],     # terrain              light green
    [230, 230, 250],    # manmade              white
    [0, 175, 0],        # vegetation           green
]
def downsample_label(label, voxel_size=(200, 200, 16), downscale=2, dataset='occ3d'):
    r"""downsample the labeled data,
    code taken from https://github.com/waterljwant/SSC/blob/master/dataloaders/dataloader.py#L262
    Shape:
        label, (240, 144, 240)
        label_downscale, if downsample==4, then (60, 36, 60)
    """
    if dataset == 'occ3d':
        empty_cls_idx = 17
    elif dataset == 'openocc':
        empty_cls_idx = 16

    if downscale == 1:
        return label
    ds = downscale
    small_size = (
        voxel_size[0] // ds,
        voxel_size[1] // ds,
        voxel_size[2] // ds,
    )  # small size
    label_downscale = np.zeros(small_size, dtype=np.uint8)
    empty_t = 0.95 * ds * ds * ds  # threshold
    s01 = small_size[0] * small_size[1]
    label_i = np.zeros((ds, ds, ds), dtype=np.int32)

    for i in range(small_size[0] * small_size[1] * small_size[2]):
        z = int(i / s01)
        y = int((i - z * s01) / small_size[0])
        x = int(i - z * s01 - y * small_size[0])

        label_i[:, :, :] = label[
            x * ds : (x + 1) * ds, y * ds : (y + 1) * ds, z * ds : (z + 1) * ds
        ]
        label_bin = label_i.flatten()

        zero_count_0 = np.array(np.where(label_bin == empty_cls_idx)).size

        zero_count = zero_count_0
        if zero_count > empty_t:
            label_downscale[x, y, z] = empty_cls_idx
        else:
            label_i_s = label_bin[
                np.where(np.logical_and(label_bin >= 0, label_bin < empty_cls_idx))
            ]
            label_downscale[x, y, z] = np.argmax(np.bincount(label_i_s))
    return label_downscale

def load_occ_data():
    occ_gt_data_path = 'demo_occ_data/gt_occ/0.npz'
    pred_occ_1_1_path = 'demo_occ_data/pred_occ_1_1/0.npz'
    pred_occ_1_2_path = 'demo_occ_data/pred_occ_1_2/0.npz'
    pred_occ_1_4_path = 'demo_occ_data/pred_occ_1_4/0.npz'
    pred_occ_1_8_path = 'demo_occ_data/pred_occ_1_8/0.npz'

    gt_occ = np.load(occ_gt_data_path)['occ']
    pred_occ_1_1 = np.load(pred_occ_1_1_path)['occ_pred'][0]
    pred_occ_1_2 = np.load(pred_occ_1_2_path)['occ_pred']
    pred_occ_1_4 = np.load(pred_occ_1_4_path)['occ_pred']
    pred_occ_1_8 = np.load(pred_occ_1_8_path)['occ_pred']

    return gt_occ, pred_occ_1_1, pred_occ_1_2, pred_occ_1_4, pred_occ_1_8

def vis_occ_on_3d(semantics_occ, voxel_size, palette, dataset='occ3d'):
    if dataset == 'occ3d':
        free_label = 17 # for occ3d
    else:
        free_label = 16 # for openocc

    vis = Det3DLocalVisualizer()
    param = o3d.io.read_pinhole_camera_parameters('view.json')

    if dataset == 'occ3d':
        vis._draw_occ_sem_seg(semantics_occ, palette, voxelSize=voxel_size, ignore_labels=[0, free_label])
    elif dataset == 'openocc':
        vis._draw_occ_sem_seg(semantics_occ, palette, voxelSize=voxel_size, ignore_labels=[free_label])
    vis.show(view_json=param)

    param = vis.o3d_vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters('view.json', param)

    vis.o3d_vis.destroy_window()

if __name__ == '__main__':
    dataset = 'occ3d'

    gt_occ, pred_occ_1_1, pred_occ_1_2, pred_occ_1_4, pred_occ_1_8 = load_occ_data()
    vis_occ_on_3d(gt_occ, voxel_size=[0.4, 0.4, 0.4], palette=palette, dataset=dataset)
    vis_occ_on_3d(pred_occ_1_1, voxel_size=[0.4, 0.4, 0.4], palette=palette, dataset=dataset)
    vis_occ_on_3d(pred_occ_1_2, voxel_size=[0.8, 0.8, 0.8], palette=palette, dataset=dataset)
    vis_occ_on_3d(pred_occ_1_4, voxel_size=[1.6, 1.6, 1.6], palette=palette, dataset=dataset)
    vis_occ_on_3d(pred_occ_1_8, voxel_size=[3.2, 3.2, 3.2], palette=palette, dataset=dataset)