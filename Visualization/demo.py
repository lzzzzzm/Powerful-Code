import mmcv
import mmengine
import numpy as np
import open3d as o3d

import math

import torch
from mmengine.fileio import get
from mmengine.structures import InstanceData

from visualization import Det3DLocalVisualizer

from mmdet3d.structures.points import BasePoints, get_points_type
from mmdet3d.structures import PointData

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

openoccv2_palette = np.array(
    [
        [0, 150, 245],      # car                  blue         √
        [160, 32, 240],     # truck                purple       √
        [135, 60, 0],       # trailer              brown        √
        [255, 255, 0],      # bus                  yellow       √
        [0, 255, 255],      # construction_vehicle cyan         √
        [255, 192, 203],    # bicycle              pink         √
        [255, 127, 0],      # motorcycle           dark orange  √
        [255, 0, 0],        # pedestrian           red          √
        [255, 240, 150],    # traffic_cone         light yellow
        [255, 120, 50],     # barrier              orange
        [255, 0, 255],      # driveable_surface    dark pink
        [139, 137, 137],    # other_flat           dark red
        [75, 0, 75],        # sidewalk             dard purple
        [150, 240, 80],     # terrain              light green
        [230, 230, 250],    # manmade              white
        [0, 175, 0],        # vegetation           green
        [0, 0, 0],          # noise                Fake
    ]
)

bboxes_palette = [
    [0, 150, 245],      # car                  blue
    [160, 32, 240],     # truck                purple
    [135, 60, 0],       # trailer              brown
    [255, 255, 0],      # bus                  yellow
    [0, 255, 255],      # construction_vehicle cyan
    [255, 192, 203],    # bicycle              pink
    [255, 127, 0],      # motorcycle           dark orange
    [255, 0, 0],        # pedestrian           red
    [255, 240, 150],    # traffic_cone         light yellow
    [255, 120, 50],     # barrier              orange
    [0, 175, 0],        # vegetation           green
    [0, 175, 0],        # vegetation           green
    [0, 175, 0],        # vegetation           green
]

# gt bboxes annotations
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

def load_pts_data():
    pts_filename = 'demo_data/lidar_data.bin'
    pts_bytes = get(pts_filename, backend_args=None)
    points = np.frombuffer(pts_bytes, dtype=np.float32)
    points = points.reshape(-1, 5)[:, :3]

    lidar2ego = [[ 0.00203327,  0.99970406 , 0.02424172 , 0.94371301],
                 [-0.99998051,  0.00217566, -0.00584864,  0.        ],
                 [-0.00589965, -0.02422936,  0.99968904,  1.84022999],
                 [ 0.  ,        0.   ,       0.  ,        1.        ]]

    lidar2ego = np.array(lidar2ego)
    lidar2ego_rot = lidar2ego[:3, :3]
    lidar2ego_trans = lidar2ego[:3, 3:4]

    ego2lidar = np.eye(4)
    ego2lidar[:3, :3] = lidar2ego_rot.T
    ego2lidar[:3, 3:4] = -1 * np.matmul(
        lidar2ego_rot.T, lidar2ego_trans.reshape(3, 1)
    )

    lidar_info = dict()
    lidar_info['lidar2ego'] = lidar2ego
    lidar_info['ego2lidar'] = ego2lidar

    return points, lidar_info

def load_pts_sem_data():
    pts_semantic_mask_path = 'demo_data/lidar_seg_data.bin'
    mask_bytes = get(
        pts_semantic_mask_path, backend_args=None)
    # add .copy() to fix read-only bug
    pts_semantic_mask = np.frombuffer(
        mask_bytes, dtype=np.uint8).copy()

    # for nuscenes dataset
    seg_label_mapping = {0: 0, 1: 0, 2: 7, 3: 7, 4: 7, 5: 0, 6: 7,
                         7: 0, 8: 0,9: 1, 10: 0, 11: 0, 12: 8, 13: 0,
                         14: 2, 15: 3, 16: 3, 17: 4, 18: 5, 19: 0, 20: 0, 21: 6,
                         22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 0,
                         30: 16, 31: 0}
    converted_pts_sem_mask = np.vectorize(
        seg_label_mapping.__getitem__, otypes=[np.uint8])(
        pts_semantic_mask)

    return converted_pts_sem_mask

def load_img_data():
    image_path = 'demo_data/CAM_FRONT.jpg'
    gt_bboxes_3d_path = 'demo_data/gt_bboxes_3d.pkl'
    gt_labels_3d_path = 'demo_data/gt_labels_3d.pkl'


    img_bytes = get(image_path, None)
    img = mmcv.imfrombytes(img_bytes)
    img = mmcv.bgr2rgb(img)
    gt_bboxes_3d = mmengine.load(gt_bboxes_3d_path)
    gt_labels_3d = mmengine.load(gt_labels_3d_path)

    cam2img = [[1.2664172e+03, 0.0000000e+00, 8.1626703e+02, 0.0000000e+00],
               [0.0000000e+00, 1.2664172e+03, 4.9150708e+02, 0.0000000e+00],
               [0.0000000e+00, 0.0000000e+00, 1.0000000e+00, 0.0000000e+00],
               [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00]]
    lidar2cam = [[0.9999702572822571, 0.003407371463254094, 0.0069207423366606236, 0.01687305048108101],
                 [0.006852706428617239, 0.019589632749557495, -0.9997846484184265, -0.32902389764785767],
                 [-0.0035422123037278652, 0.99980229139328, 0.019565701484680176, -0.4292221665382385],
                 [0.0, 0.0, 0.0, 1.0]]
    cam2img = np.array(cam2img)
    lidar2cam = np.array(lidar2cam)
    lidar2img = cam2img @ lidar2cam

    img_info = dict()
    img_info['cam2img'] = cam2img
    img_info['lidar2cam'] = lidar2cam
    img_info['lidar2img'] = lidar2img
    img_info['gt_bboxes_3d'] = gt_bboxes_3d
    img_info['gt_labels_3d'] = gt_labels_3d
    return img, img_info

def load_aug_data():
    gt_bboxes_3d_path = 'demo_data/gt_bboxes_3d.pkl'
    gt_labels_3d_path = 'demo_data/gt_labels_3d.pkl'
    gt_occupancy_path= 'demo_data/occ_labels.npz'

    gt_bboxes_3d = mmengine.load(gt_bboxes_3d_path)
    gt_labels_3d = mmengine.load(gt_labels_3d_path)
    gt_occupancy = np.load(gt_occupancy_path)
    return gt_bboxes_3d, gt_labels_3d, gt_occupancy

def load_occ_data():
    # occ_label_path = 'demo_data/occ_labels.npz'
    occ_label_path = 'pred_save/63.npz'
    semantics_occ = np.load(occ_label_path)

    return semantics_occ

def vis_lidar_on_3d(points):
    vis = Det3DLocalVisualizer()
    vis.set_points(points, vis_mode='add')
    vis.show()
    vis.o3d_vis.destroy_window()

def vis_lidar_sem_on_3d(points, pts_semantic_mask):
    pts_seg = PointData()
    pts_seg.pts_semantic_mask = pts_semantic_mask

    vis = Det3DLocalVisualizer()
    vis._draw_pts_sem_seg(points, pts_seg, palette)
    vis.show()
    vis.o3d_vis.destroy_window()

def vis_points_on_img(points, img, img_info):
    pts2img = img_info['lidar2img']

    vis = Det3DLocalVisualizer()
    vis.set_image(img)
    vis.draw_points_on_image(points, pts2img)
    vis.show()

def vis_3dboxes_on_img(img, img_info):
    gt_bboxes_3d = img_info['gt_bboxes_3d']

    vis = Det3DLocalVisualizer()
    vis.set_image(img)
    vis.draw_proj_bboxes_3d(gt_bboxes_3d, img_info)
    vis.show()

def vis_points_with_3dboxes_on_3d(points, pts_semantic_mask, img_info, lidar_info):
    gt_bboxes_3d = img_info['gt_bboxes_3d']
    gt_labels_3d = img_info['gt_labels_3d']

    pts_seg = PointData()
    pts_seg.pts_semantic_mask = pts_semantic_mask

    lidar2ego = lidar_info['lidar2ego']
    points = np.concatenate((points, np.ones_like(points[..., :1])), -1)
    points = (points @ (lidar2ego.T))[:, :3]

    vis = Det3DLocalVisualizer()
    vis._draw_pts_sem_seg(points, pts_seg, palette, pcd_mode=0)     # pcd_mode=0 means need convert to the Depth coordinate
    vis.draw_bboxes_3d(gt_bboxes_3d, gt_labels_3d=gt_labels_3d, palette=bboxes_palette, trans_matrix=lidar2ego)
    vis.show()

def vis_occ_on_3d(semantics_occ):
    free_label = 17 # for occ3d

    occ_labels = semantics_occ['semantics']
    mask_lidar = semantics_occ['mask_lidar']
    occ_labels[mask_lidar == 0] = free_label

    vis = Det3DLocalVisualizer()
    param = o3d.io.read_pinhole_camera_parameters('view.json')

    vis._draw_occ_sem_seg(occ_labels, palette)
    vis.show(view_json=param)

    param = vis.o3d_vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters('view.json', param)

    vis.o3d_vis.destroy_window()

def vis_occ_nomask_on_3d(semantics_occ):
    total_size = [80, 80, 6.4]

    # free_label = 16 # for openocc_v2
    free_label = 17 # for occ3d
    occ_labels = semantics_occ['semantics']
    voxelSize = [0.4, 0.4, 0.4]
    voxelSize[0] = total_size[0] / occ_labels.shape[0]
    voxelSize[1] = total_size[1] / occ_labels.shape[1]
    voxelSize[2] = total_size[2] / occ_labels.shape[2]

    vis = Det3DLocalVisualizer()
    param = o3d.io.read_pinhole_camera_parameters('view.json')

    # vis._draw_occ_sem_seg(occ_labels, openoccv2_palette, ignore_labels=[free_label], voxelSize=voxelSize)
    vis._draw_occ_sem_seg(occ_labels, palette, ignore_labels=[free_label], voxelSize=voxelSize)
    vis.show(view_json=param)

    param = vis.o3d_vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters('view.json', param)

    vis.o3d_vis.destroy_window()

def vis_lidar_with_occ_on_3d(semantics_occ, points, lidar_info, pts_semantic_mask):
    free_label = 17 # for occ3d

    occ_labels = semantics_occ['semantics']
    mask_lidar = semantics_occ['mask_lidar']
    occ_labels[mask_lidar == 0] = free_label

    lidar2ego = lidar_info['lidar2ego']
    points = np.concatenate((points, np.ones_like(points[..., :1])), -1)
    points = (points @ (lidar2ego.T))[:, :3]

    pts_seg = PointData()
    pts_seg.pts_semantic_mask = pts_semantic_mask

    vis = Det3DLocalVisualizer()
    vis._draw_pts_sem_seg(points, pts_seg, palette)
    vis._draw_occ_sem_seg(occ_labels, palette, show_color=False)
    vis.show()

def vis_3dboxes_with_occ_on_3d(semantics_occ, img_info, lidar_info):
    free_label = 17  # for occ3d

    occ_labels = semantics_occ['semantics']
    mask_lidar = semantics_occ['mask_lidar']
    occ_labels[mask_lidar == 0] = free_label

    gt_bboxes_3d = img_info['gt_bboxes_3d']

    lidar2ego = lidar_info['lidar2ego']

    vis = Det3DLocalVisualizer()
    vis._draw_occ_sem_seg(occ_labels, palette)
    vis.draw_bboxes_3d(gt_bboxes_3d, trans_matrix=lidar2ego, rot_axis=2)
    vis.show()

def vis_3dboxes_with_occ_on_3d_aug(gt_bboxes_3d, gt_labels_3d, gt_occupancy):
    vis = Det3DLocalVisualizer()
    vis._draw_occ_sem_seg(gt_occupancy, palette)
    # vis.draw_bboxes_3d(gt_bboxes_3d, trans_matrix=lidar2ego)
    vis.show()

def generate_lidar_rays():
    # prepare lidar ray angles
    pitch_angles = []
    for k in range(10):
        angle = math.pi / 2 - math.atan(k + 1)
        pitch_angles.append(-angle)

    # nuscenes lidar fov: [0.2107773983152201, -0.5439104895672159] (rad)
    while pitch_angles[-1] < 0.21:
        delta = pitch_angles[-1] - pitch_angles[-2]
        pitch_angles.append(pitch_angles[-1] + delta)

    lidar_rays = []
    for pitch_angle in pitch_angles:
        for azimuth_angle in np.arange(0, 360, 1):
            azimuth_angle = np.deg2rad(azimuth_angle)

            x = np.cos(pitch_angle) * np.cos(azimuth_angle)
            y = np.cos(pitch_angle) * np.sin(azimuth_angle)
            z = np.sin(pitch_angle)

            lidar_rays.append((x, y, z))

    return np.array(lidar_rays, dtype=np.float32)

if __name__ == '__main__':

    points, lidar_info = load_pts_data()
    pts_semantic_mask = load_pts_sem_data()
    img, img_info = load_img_data()
    semantics_occ = load_occ_data()
    lidar_rays = generate_lidar_rays()
    gt_bboxes_3d, gt_labels_3d, gt_occupancy = load_aug_data()
    # vis_lidar_on_3d(lidar_rays)
    # vis_lidar_sem_on_3d(points, pts_semantic_mask)
    # vis_points_on_img(points, img, img_info)
    # vis_3dboxes_on_img(img, img_info)
    # vis_points_with_3dboxes_on_3d(points, pts_semantic_mask, img_info, lidar_info)
    # vis_occ_on_3d(semantics_occ)
    # vis_occ_nomask_on_3d(semantics_occ)
    # vis_lidar_with_occ_on_3d(semantics_occ, points, lidar_info, pts_semantic_mask)
    # vis_3dboxes_with_occ_on_3d(semantics_occ, img_info, lidar_info)
    # vis_3dboxes_with_occ_on_3d_aug(gt_bboxes_3d, gt_labels_3d, gt_occupancy)