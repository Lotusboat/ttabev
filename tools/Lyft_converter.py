# Copyright (c) OpenMMLab. All rights reserved.
import pickle

import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from lyft_dataset_sdk import LyftDataset
from tools.data_converter import nuscenes_converter as nuscenes_converter

map_name_from_general_to_detection = {
    'car':'car',
    'truck':'truck',
    'bus':'bus',
    'emergency_vehicle':'ignore',
    'other_vehicle':'car',
    'motorcycle':'motorcycle',
    'bicycle':'bicycle',
    'pedestrian':'pedestrian',
    'animal':'ignore',
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
}
classes = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

def get_gt(info):
    """Generate gt labels from info.

    Args:
        info(dict): Infos needed to generate gt labels.

    Returns:
        Tensor: GT bboxes.
        Tensor: GT labels.
    """
    ego2global_rotation = info['cams']['CAM_FRONT']['ego2global_rotation']
    ego2global_translation = info['cams']['CAM_FRONT'][
        'ego2global_translation']
    trans = -np.array(ego2global_translation)
    rot = Quaternion(ego2global_rotation).inverse
    gt_boxes = list()
    gt_labels = list()

    for ann_info in info['ann_infos']:
        # Use ego coordinate.
        if (map_name_from_general_to_detection[ann_info['category_name']]
                not in classes): # or ann_info['num_lidar_pts'] + ann_info['num_radar_pts'] <= 0
            print('*********************************')
            print(ann_info['category_name'])
            continue
        box = Box(
            ann_info['translation'],
            ann_info['size'],
            Quaternion(ann_info['rotation']),
            velocity=ann_info['velocity'],
        )
        box.translate(trans)
        box.rotate(rot)
        box_xyz = np.array(box.center)
        box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
        box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
        box_velo = np.array(box.velocity[:2])
        print('box_xyz',box_xyz)
        print('box_dxdydz',box_dxdydz)
        print('box_yaw',box_yaw)
        print('box_velo',box_velo)
        gt_box = np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])
        gt_boxes.append(gt_box)

        gt_labels.append(
            classes.index(
                map_name_from_general_to_detection[ann_info['category_name']]))
    return gt_boxes, z



def add_ann_adj_info(extra_tag):
    nuscenes_version = 'v1.0-trainval'
    dataroot = './data/lyft/'
    for set in ['train', 'test']:
        dataset = pickle.load(
            open('./data/lyft/%s_infos_%s.pkl' % (extra_tag, set), 'rb'))
        nuscenes = LyftDataset(dataroot, dataroot + 'v1.01-' + set + '/v1.01-'+set)
        for id in range(len(dataset['infos'])):
            if id % 10 == 0:
                print('%d/%d' % (id, len(dataset['infos'])))
            info = dataset['infos'][id]
            # get sweep adjacent frame info
            sample = nuscenes.get('sample', info['token'])
            ann_infos = list()
            for ann in sample['anns']:
                ann_info = nuscenes.get('sample_annotation', ann)
                velocity = nuscenes.box_velocity(ann_info['token'])
                if np.any(np.isnan(velocity)):
                    velocity = np.zeros(3)
                ann_info['velocity'] = velocity
                ann_infos.append(ann_info)
            dataset['infos'][id]['ann_infos'] = ann_infos
            dataset['infos'][id]['ann_infos'] = get_gt(dataset['infos'][id])
            dataset['infos'][id]['scene_token'] = sample['scene_token']
        with open('./data/lyft/%s_infos_ann_%s.pkl' % (extra_tag, set),
                  'wb') as fid:
            pickle.dump(dataset, fid)



def add_ann_adj_info_val(extra_tag):
    dataroot = './data/lyft/'
    set = 'val'
    dataset = pickle.load(
        open('./data/lyft/%s_infos_%s.pkl' % (extra_tag, set), 'rb'))
    nuscenes = LyftDataset(dataroot, dataroot + 'v1.01-train' + '/v1.01-train')
    for id in range(len(dataset['infos'])):
        if id % 10 == 0:
            print('%d/%d' % (id, len(dataset['infos'])))
        info = dataset['infos'][id]
        # get sweep adjacent frame info
        sample = nuscenes.get('sample', info['token'])
        ann_infos = list()
        for ann in sample['anns']:
            ann_info = nuscenes.get('sample_annotation', ann)
            velocity = nuscenes.box_velocity(ann_info['token'])
            if np.any(np.isnan(velocity)):
                velocity = np.zeros(3)
            ann_info['velocity'] = velocity
            ann_infos.append(ann_info)
        dataset['infos'][id]['ann_infos'] = ann_infos
        dataset['infos'][id]['ann_infos'] = get_gt(dataset['infos'][id])
        dataset['infos'][id]['scene_token'] = sample['scene_token']
    with open('./data/lyft/%s_infos_ann_%s.pkl' % (extra_tag, set),
              'wb') as fid:
        pickle.dump(dataset, fid)


if __name__ == '__main__':

    dataset = 'lyft'
    version = 'v1.01'
    train_version = f'{version}-trainval'
    root_path = './data/nuscenes'
    extra_tag = 'lyft'
    print('add_ann_infos')
    add_ann_adj_info(extra_tag)
    add_ann_adj_info_val(extra_tag)


#
# dataroot = './data/lyft/'
# set = 'train'
# dataset = pickle.load(
#     open('./data/lyft/%s_infos_%s.pkl' % (extra_tag, set), 'rb'))
# nuscenes = LyftDataset(dataroot, dataroot + 'v1.01-' + set + '/v1.01-' + set)
# for id in range(len(dataset['infos'])):
#     if id % 10 == 0:
#         print('%d/%d' % (id, len(dataset['infos'])))
#     info = dataset['infos'][id]
#     # get sweep adjacent frame info
#     sample = nuscenes.get('sample', info['token'])
#     ann_infos = list()
#     for ann in sample['anns']:
#         ann_info = nuscenes.get('sample_annotation', ann)
#         velocity = nuscenes.box_velocity(ann_info['token'])
#         if np.any(np.isnan(velocity)):
#             velocity = np.zeros(3)
#         ann_info['velocity'] = velocity
#         ann_infos.append(ann_info)
#     dataset['infos'][id]['ann_infos'] = ann_infos
#     dataset['infos'][id]['ann_infos'] = get_gt(dataset['infos'][id])
#     print(dataset['infos'][id]['ann_infos'])
#     dataset['infos'][id]['scene_token'] = sample['scene_token']