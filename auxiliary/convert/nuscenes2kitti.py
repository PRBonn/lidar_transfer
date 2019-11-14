#!/usr/bin/python3

import sys
from nuscenes.nuscenes import NuScenes
import nuscenes.utils.geometry_utils as geoutils
from pyquaternion import Quaternion
import numpy as np
import os
import numpy.linalg as la


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: ./nuscenes2kitti.py <dataset_folder> <output_folder> [<scene_name>]")
        exit(1)

    dataroot = sys.argv[1]


    nusc = NuScenes(version = 'v1.0-mini', dataroot = dataroot, verbose = True)

    name2id = {scene["name"]:id for id, scene in enumerate(nusc.scene)}

    scenes2parse = []

    if len(sys.argv) > 3:
        if sys.argv[3] not in name2id: 
            print("No scene with name {} found.".format(sys.argv[2]))
            print("Available scenes: {}".format(" ".join(name2id.keys())))

            exit(1)
        scenes2parse.append(sys.argv[3])
    else:
        scenes2parse = name2id.keys()

    for scene_name in scenes2parse:

        print("Converting {} ...".format(scene_name))

        output_folder = os.path.join(sys.argv[2], scene_name)
        velodyne_folder = os.path.join(output_folder, "velodyne/")

        first_lidar = nusc.get('sample', nusc.scene[name2id[scene_name]]["first_sample_token"])["data"]["LIDAR_TOP"]
        last_lidar = nusc.get('sample', nusc.scene[name2id[scene_name]]["last_sample_token"])["data"]["LIDAR_TOP"]

        current_lidar = first_lidar
        lidar_filenames = []
        poses = []

        if not os.path.exists(velodyne_folder):
            print("Creating output folder: {}".format(output_folder))
            os.makedirs(velodyne_folder)

        original = []

        while current_lidar != "":
            lidar_data = nusc.get('sample_data', current_lidar)
            calib_data = nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
            egopose_data = nusc.get('ego_pose',  lidar_data["ego_pose_token"])
            
            car_to_velo = geoutils.transform_matrix(calib_data["translation"], Quaternion(calib_data["rotation"]))
            pose_car = geoutils.transform_matrix(egopose_data["translation"], Quaternion(egopose_data["rotation"]))
            
            pose = np.dot(pose_car, car_to_velo)
            
            scan = np.fromfile(os.path.join(dataroot, lidar_data["filename"]), dtype=np.float32)
            points = scan.reshape((-1, 5))[:, :4]

            # ensure that remission is in [0,1]
            max_remission = np.max(points[:, 3])
            min_remission = np.min(points[:, 3])
            points[:, 3] = (points[:, 3] - min_remission)  / (max_remission - min_remission)
            
            output_filename = os.path.join(velodyne_folder, "{:05d}.bin".format(len(lidar_filenames)))
            points.tofile(output_filename)

            original.append(("{:05d}.bin".format(len(lidar_filenames)), lidar_data["filename"]))
            
            poses.append(pose)
            lidar_filenames.append(os.path.join(dataroot, lidar_data["filename"]))
            
            current_lidar =  lidar_data["next"]

        ref = la.inv(poses[0])
        pose_file = open(os.path.join(output_folder, "poses.txt"), "w")
        for pose in poses:
            pose_str = [str(v) for v in (np.dot(ref, pose))[:3,:4].flatten()]
            pose_file.write(" ".join(pose_str))
            pose_file.write("\n")

        print("{} scans read.".format(len(lidar_filenames)))

        pose_file.close()

        # write dummy calibration.
        calib_file = open(os.path.join(output_folder, "calib.txt"), "w")
        calib_file.write("P0: 1 0 0 0 0 1 0 0 0 0 1 0\n")
        calib_file.write("P1: 1 0 0 0 0 1 0 0 0 0 1 0\n")
        calib_file.write("P2: 1 0 0 0 0 1 0 0 0 0 1 0\n")
        calib_file.write("P3: 1 0 0 0 0 1 0 0 0 0 1 0\n")
        calib_file.write("Tr: 1 0 0 0 0 1 0 0 0 0 1 0\n")
        calib_file.close()

        original_file = open(os.path.join(output_folder, "original.txt"), "w")
        for pair in original: original_file.write(pair[0] + ":" + pair[1] + "\n")
        original_file.close()
