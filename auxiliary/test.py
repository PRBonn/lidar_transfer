#!/usr/bin/env python3

import unittest as ut
import numpy as np
import yaml
from laserscan import LaserScan, SemLaserScan, MultiSemLaserScan

def project(points, fov_up, fov_down):
    # laser parameters
    fov_up = fov_up / 180.0 * np.pi      # field of view up in radians
    fov_down = fov_down / 180.0 * np.pi  # field of view down in radians
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in radians

    depth = np.linalg.norm(points, 2, axis=1)

    # get scan components
    scan_x = points[:, 0]
    scan_y = points[:, 1]
    scan_z = points[:, 2]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth) # arcsin!!
    proj_x = 0.5 * (yaw / np.pi + 1.0)          # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]

    return np.array([proj_x, proj_y]).T

def unproject(points2d, depth, fov_up, fov_down):
    # laser parameters
    fov_up = fov_up / 180.0 * np.pi      # field of view up in radians
    fov_down = fov_down / 180.0 * np.pi  # field of view down in radians
    fov = abs(fov_down) + abs(fov_up)    # get field of view total in radians

    # proj_x = np.zeros((2,1))
    proj_x = points2d[:,0]
    proj_y = points2d[:,1]

    yaw = (proj_x * 2 - 1.0) * np.pi # theta
    pitch = 1.0 * fov - proj_y * fov - abs(fov_down) # phi
    point_x = depth * np.sin(np.pi/2 - pitch) * np.cos(-yaw)
    point_y = depth * np.sin(np.pi/2 - pitch) * np.sin(-yaw)
    point_z = depth * np.cos(np.pi/2 - pitch)

    return np.array([point_x, point_y, point_z]).T

class SimpleTest(ut.TestCase):
    def test(self):
        points = np.array([[1, 0, 0], [0, 0, 1], [1,1,1], [2, 2, 2]])
        print("*" * 10, "\n",points, "\n", "*" * 10)
        depth = np.linalg.norm(points, 2, axis=1)
        fov_up = 10
        fov_down = 10
        points2d = project(points, fov_up, fov_down)
        points_test = unproject(points2d, depth, fov_up, fov_down)
        print("*" * 10, "\n", points_test, "\n", "*" * 10)
        self.assertTrue(np.allclose(points, points_test))

        # CFG = yaml.load(open("config/lidar_transfer.yaml", 'r'))
        # color_dict = CFG["color_map"]
        # nclasses = len(color_dict)
        # scans = MultiSemLaserScan(1, 1, nclasses, "cp", color_dict)
        # scans.points = points
        # scans.remissions = np.zeros(4)
        # scans.do_range_projection(fov_up, fov_down)
        # print(scans.proj_x)

if __name__ == '__main__':
    ut.main()