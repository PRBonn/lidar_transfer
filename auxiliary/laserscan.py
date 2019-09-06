#!/usr/bin/env python3
import numpy as np
import os
import struct
import time
import torch
import copy
import imageio
import auxiliary.fusion_lidar as fl
from auxiliary.np_ioueval import iouEval
from auxiliary.tools import get_mpl_colormap, convert_range


class LaserScan:
  """Class that contains LaserScan with x,y,z,r"""
  EXTENSIONS_SCAN = ['.bin']

  def __init__(self, H, W, transformation=None, beam_angles=None):
    self.proj_H = H
    self.proj_W = W
    if not transformation:
      transformation = np.eye(4)
    self.transformation = np.array(transformation).reshape(4, 4)
    self.reset()
    self.beam_angles = beam_angles  # TODO import deg transform to rad
    self.pose = np.eye(4)

  def reset(self):
    """ Reset scan members. """
    self.points = np.zeros((0, 3), dtype=np.float32)     # [m, 3]: x, y, z
    self.remissions = np.zeros((0, ), dtype=np.float32)  # [m ,1]: remission
    self.back_points = np.zeros((0, 3), dtype=np.float32)

    # a scan can also be represented as a spherical projection depth
    # image, and another channel which saves for each pixel, to which
    # in the point it corresponds
    self.proj_range = \
        np.full((self.proj_H, self.proj_W), -1,
                dtype=np.float32)           # [H,W] range (-1 is no data)
    self.proj_xyz = \
        np.full((self.proj_H, self.proj_W, 3), -1,
                dtype=np.float32)           # [H,W,3] xyz coord (-1 is no data)
    self.proj_remissions = \
        np.full((self.proj_H, self.proj_W), -1,
                dtype=np.float32)           # [H,W] intensity (-1 is no data)
    self.proj_idx = \
        np.full((self.proj_H, self.proj_W), -1,
                dtype=np.int32)             # [H,W] index (-1 is no data)
    self.proj_x = np.zeros((0, 1), dtype=np.float32)        # [m, 1]: x
    self.proj_y = np.zeros((0, 1), dtype=np.float32)        # [m, 1]: y
    self.unproj_range = np.zeros((0, 1), dtype=np.float32)  # [m, 1]: range
    self.proj_mask = np.zeros(
        (self.proj_H, self.proj_W), dtype=np.int32)  # [H,W] mask

  def size(self):
    """ Return the size of the point cloud. """
    return self.points.shape[0]

  def __len__(self):
    return self.size()

  def open_scan_append(self, filename, pose, fov_up, fov_down):
    """ Open raw scan and fill in attributes and appends it to existing scans
    """
    # check filename is string
    if not isinstance(filename, str):
      raise TypeError("Filename should be string type, "
                      "but was {type}".format(type=str(type(filename))))

    # if all goes well, open pointcloud
    scan = np.fromfile(filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    pose = pose.reshape((-1, 4))

    # transform with pose
    hom_points = np.ones((scan.shape[0], 4))
    hom_points[:, 0:3] = scan[:, 0:3]
    t_points = np.matmul(pose, hom_points.T)

    # Apply given transformation to move sensor to specific position/angle
    t_points = np.matmul(self.transformation, t_points).T

    # TODO no pose adaption on import or remove it from tsdf
    # t_points = scan[:, 0:3]

    # put in attribute
    if self.points.size == 0:
      self.points = t_points[:, 0:3]    # get xyz
      self.remissions = scan[:, 3]  # get remission
    else:
      new_points = t_points[:, 0:3]
      # print(new_points.shape, self.points.shape)
      self.points = np.concatenate((self.points, new_points))
      # print("->", self.points.shape)
      new_remissions = scan[:, 3]
      self.remissions = np.concatenate((self.remissions, new_remissions))

  def apply_transformation(self, transformation):
    """ transform with given transformation (4x4)
    """
    hom_points = np.ones((self.points.shape[0], 4))
    hom_points[:, 0:3] = self.points[:, 0:3]
    hom_points = np.matmul(transformation, hom_points.T).T
    self.points = hom_points[:, 0:3]

  def apply_pose(self):
    """ transform with pose
    """
    self.apply_transformation(self.pose)

  def apply_inv_pose(self):
    """ transform with pose
    """
    self.apply_transformation(np.linalg.inv(self.pose))

  def open_scan(self, filename, fov_up, fov_down):
    """ Open raw scan and fill in attributes
    """
    # reset just in case there was an open structure
    self.reset()

    # check filename is string
    if not isinstance(filename, str):
      raise TypeError("Filename should be string type, "
                      "but was {type}".format(type=str(type(filename))))

    # check extension is a laserscan
    if not any(filename.endswith(ext) for ext in self.EXTENSIONS_SCAN):
      raise RuntimeError("Filename extension is not valid scan file.")

    # if all goes well, open pointcloud
    scan = np.fromfile(filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))

    # put in attribute
    self.points = scan[:, 0:3]    # get xyz
    self.remissions = scan[:, 3]  # get remission

    # if projection is wanted, then do it and fill in the structure
    # self.do_range_projection(fov_up, fov_down)

  def remove_points(self, keep_index):
    self.points = self.points[keep_index]
    self.remissions = self.remissions[keep_index]
    # TODO Merge classes to avoid this??
    self.label = self.label[keep_index]
    self.label_color = self.label_color[keep_index]

    # TODO add pinhole projection?
  def do_range_projection(self, fov_up, fov_down, remove=False):
    """ Project a pointcloud into a spherical projection image.projection.
        Function takes two arguments.
    """
    # laser parameters
    fov_up = fov_up / 180.0 * np.pi      # field of view up in radians
    fov_down = fov_down / 180.0 * np.pi  # field of view down in radians
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in radians

    # get depth of all points
    depth = np.linalg.norm(self.points, 2, axis=1)
    # remove points with depth == 0 to counter division by zero
    keep_index = (depth != 0)
    if remove:
      depth = depth[keep_index]
      self.remove_points(keep_index)

    # TODO expose parameter to limit by depth
    # depth_50 = depth < 50
    # depth = depth[depth_50]

    # get scan components
    scan_x = self.points[:, 0]  # [depth_50]
    scan_y = self.points[:, 1]  # [depth_50]
    scan_z = self.points[:, 2]  # [depth_50]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # find closest matching pitch angle in hardcoded beam angles
    if self.beam_angles:
      pitch_match = np.copy(pitch)
      for i in range(len(pitch)):
        closest_beam_index = np.abs(pitch[i] - self.beam_angles).argmin()
        pitch_match[i] = self.beam_angles[closest_beam_index]
      pitch = pitch_match

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)            # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

    # remove non-valid points
    if remove:
      min_value = np.argmin(proj_y)
      # print("y", np.amin(proj_y), proj_y[min_value])
      keep_index = (proj_y >= 0) & (proj_y <= 1)
      self.remove_points(keep_index)
      depth = depth[keep_index]
      proj_y = proj_y[keep_index]
      proj_x = proj_x[keep_index]
      # print("y", np.amin(proj_y), proj_y[min_value])

    assert proj_y.any() >= 0.0
    assert proj_x.any() >= 0.0

    # scale to image size using angular resolution
    proj_x *= self.proj_W                              # in [0.0, W]
    proj_y *= self.proj_H                              # in [0.0, H]
    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(self.proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]
    self.proj_x = np.copy(proj_x)  # store a copy in original order

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(self.proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]
    self.proj_y = np.copy(proj_y)  # store a copy in original order
    self.unproj_range = np.copy(depth)  # copy of depth in original order

    # indices of original points sorted by depth
    indices = np.arange(depth.shape[0])
    # order in decreasing depth
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    indices = indices[order]
    points = self.points[order]
    remissions = self.remissions[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    # assing to images
    self.depth = depth
    self.proj_range[proj_y, proj_x] = depth
    self.proj_xyz[proj_y, proj_x] = points
    self.proj_remissions[proj_y, proj_x] = remissions
    self.proj_idx[proj_y, proj_x] = indices
    self.proj_x = proj_x
    self.proj_y = proj_y
    self.proj_mask = (self.proj_idx > 0).astype(np.float32)

  def do_range_projection_new(self, fov_up, fov_down, remove=False,
                              method="depth"):
    """ Project a pointcloud into a spherical projection image.projection.
        Function takes four arguments.
    """
    # laser parameters
    fov_up = fov_up / 180.0 * np.pi      # field of view up in radians
    fov_down = fov_down / 180.0 * np.pi  # field of view down in radians
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in radians

    # get depth of all points
    depth = np.linalg.norm(self.points, 2, axis=1)

    # remove points with depth == 0 to counter division by zero
    keep_index = (depth != 0)
    depth = depth[keep_index]
    self.remove_points(keep_index)

    # get scan components
    scan_x = self.points[:, 0]
    scan_y = self.points[:, 1]
    scan_z = self.points[:, 2]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # find closest matching pitch angle in hardcoded beam angles
    if self.beam_angles:
      pitch_match = np.copy(pitch)
      for i in range(len(pitch)):
        closest_beam_index = np.abs(pitch[i] - self.beam_angles).argmin()
        pitch_match[i] = self.beam_angles[closest_beam_index]
      pitch = pitch_match

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)            # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

    # remove non-valid points
    if remove:
      min_value = np.argmin(proj_y)
      # print("y", np.amin(proj_y), proj_y[min_value])
      keep_index = (proj_y >= 0) & (proj_y <= 1)
      self.remove_points(keep_index)
      depth = depth[keep_index]
      proj_y = proj_y[keep_index]
      proj_x = proj_x[keep_index]
      # print("y", np.amin(proj_y), proj_y[min_value])

    # scale to image size using angular resolution
    proj_x *= self.proj_W                              # in [0.0, W]
    proj_y *= self.proj_H                              # in [0.0, H]

    # self.proj_x = np.zeros((self.proj_W * self.proj_H, 1))
    # self.proj_y = np.zeros((self.proj_W * self.proj_H, 1))

    # round and clamp for use as index
    proj_x_cl = np.floor(proj_x)
    proj_x_cl = np.minimum(self.proj_W - 1, proj_x_cl)
    proj_x_cl = np.maximum(0, proj_x_cl).astype(np.int32)   # in [0,W-1]
    proj_y_cl = np.floor(proj_y)
    proj_y_cl = np.minimum(self.proj_H - 1, proj_y_cl)
    proj_y_cl = np.maximum(0, proj_y_cl).astype(np.int32)   # in [0,H-1]

    # proj_x = np.minimum(self.proj_W - 1, proj_x)
    # proj_x = np.maximum(0, proj_x)
    # proj_y = np.minimum(self.proj_H - 1, proj_y)
    # proj_y = np.maximum(0, proj_y)

    # Index pointing to self.points row index
    self.index = np.full((self.proj_H, self.proj_W), -1, dtype=np.int32)
    self.range_image = np.full((self.proj_H, self.proj_W), 0, dtype=np.float32)
    self.label_image = np.zeros((self.proj_H, self.proj_W, 1))
    self.label_color_image = np.zeros((self.proj_H, self.proj_W, 3))
    self.proj_remissions = np.full((self.proj_H, self.proj_W), -1,
                                   dtype=np.float32)

    if method == "depth":
      for i in range(len(proj_x)):  # iterate all points
        proj_y_i = proj_y_cl[i]
        proj_x_i = proj_x_cl[i]
        if (depth[i] < self.range_image[proj_y_i, proj_x_i] or
                self.index[proj_y_i, proj_x_i] == -1):
          self.range_image[proj_y_i, proj_x_i] = depth[i]
          self.index[proj_y_i, proj_x_i] = i
          self.label_image[proj_y_i, proj_x_i] = self.label[i]
          self.label_color_image[proj_y_i, proj_x_i] = self.label_color[i]
          self.proj_remissions[proj_y_i, proj_x_i] = self.remissions[i]

      self.proj_y = proj_y_cl[self.index]
      self.proj_x = proj_x_cl[self.index]

      self.proj_y_float = proj_y[self.index]
      self.proj_x_float = proj_x[self.index]

      self.proj_range = self.range_image
      self.unproj_range = np.copy(depth)

    elif method == "pdist":
      self.dist_image = np.full((self.proj_H, self.proj_W), 1000,
                                dtype=np.float32)

      for i in range(len(proj_x)):  # iterate all points
        proj_y_i = proj_y_cl[i]
        proj_x_i = proj_x_cl[i]
        # With smaller distance raster becomes more visible in the far points
        dist = np.linalg.norm(np.array([proj_y[i], proj_x[i]]) -
                              np.array([proj_y_i + 0.5, proj_x_i + 0.5]))
        if dist < self.dist_image[proj_y_i, proj_x_i]:
          self.dist_image[proj_y_i, proj_x_i] = dist
          self.range_image[proj_y_i, proj_x_i] = depth[i]
          self.index[proj_y_i, proj_x_i] = i
          self.label_image[proj_y_i, proj_x_i] = self.label[i]
          self.label_color_image[proj_y_i, proj_x_i] = self.label_color[i]

      self.proj_y = proj_y_cl[self.index]
      self.proj_x = proj_x_cl[self.index]

      self.proj_y_float = proj_y[self.index]
      self.proj_x_float = proj_x[self.index]

      # print( sum(self.range_image - self.proj_range) )
      self.proj_range = self.range_image
      # self.range_image = self.proj_range

    elif method == "depthfast":
      # TODO ! Faster range image by closest point by depth
      indices = np.arange(depth.shape[0])
      order = np.argsort(depth)[::-1]
      depth2 = depth[order]
      indices = indices[order]
      points = self.points[order]
      remissions = self.remissions[order]
      proj_y2 = proj_y_cl[order]
      proj_x2 = proj_x_cl[order]
      self.proj_range[proj_y2, proj_x2] = depth2
      self.proj_xyz[proj_y2, proj_x2] = points
      self.proj_remissions[proj_y2, proj_x2] = remissions
      self.proj_idx[proj_y2, proj_x2] = indices
      self.proj_x2 = proj_x2
      self.proj_y2 = proj_y2
      self.range_image = self.proj_range

      self.proj_y_float = proj_y[order]
      self.proj_x_float = proj_x[order]

    else:
      quit()

  def do_reverse_projection(self, fov_up, fov_down):
    # laser parameters
    fov_up = fov_up / 180.0 * np.pi      # field of view up in radians
    fov_down = fov_down / 180.0 * np.pi  # field of view down in radians
    fov = abs(fov_down) + abs(fov_up)    # get field of view total in radians

    # 2D coordinates and depth
    proj_x = self.proj_x[self.proj_idx] / self.proj_W
    proj_y = self.proj_y[self.proj_idx] / self.proj_H
    depth = self.depth[self.proj_idx]
    self.label_color_back = self.label_color[self.proj_idx]

    print("")
    # print(self.proj_x[0:10,:])
    print(self.proj_idx.shape)
    # proj_x = np.tile(np.linspace(1, 900, num=900), (64, 1)).reshape(-1)
    # proj_y = np.tile(np.linspace(1, 64, num=64), (900, 1)).reshape(-1)
    print("depth", depth.shape)
    print("proj_x", proj_x.shape)
    print("proj_y", proj_y.shape)

    # transform into 3D
    yaw = (proj_x * 2 - 1.0) * np.pi  # theta
    # pitch = 90-phi
    pitch = np.pi / 2 - (1.0 * fov - proj_y * fov - abs(fov_down))
    point_x = depth * np.sin(pitch) * np.cos(-yaw)
    point_y = depth * np.sin(pitch) * np.sin(-yaw)
    point_z = depth * np.cos(pitch)
    # TODO fix weird transposing
    self.back_points = np.array([point_x, point_y, point_z]).transpose(1, 2, 0)

  def do_reverse_projection_new(self, fov_up, fov_down, preserve_float=False):
    # laser parameters
    fov_up = fov_up / 180.0 * np.pi      # field of view up in radians
    fov_down = fov_down / 180.0 * np.pi  # field of view down in radians
    fov = abs(fov_down) + abs(fov_up)    # get field of view total in radians

    # 2D coordinates and depth
    depth = self.range_image

    if preserve_float:
      proj_x = self.proj_x_float / self.proj_W
      proj_y = self.proj_y_float / self.proj_H
    else:
      proj_x = self.proj_x / self.proj_W
      proj_y = self.proj_y / self.proj_H

    # transform into 3D
    yaw = (proj_x * 2 - 1.0) * np.pi  # theta
    # pitch = 90 - phi
    pitch = np.pi / 2 - (1.0 * fov - proj_y * fov - abs(fov_down))
    point_x = depth * np.sin(pitch) * np.cos(-yaw)
    point_y = depth * np.sin(pitch) * np.sin(-yaw)
    point_z = depth * np.cos(pitch)

    # TODO fix weird transposing
    self.back_points = np.array([point_x, point_y, point_z]) \
        .transpose(1, 2, 0).reshape(-1, 3)

  def torch(self):
    # pass scan to pytorch in [3, m] shape (channel first)
    self.points = torch.from_numpy(self.points.transpose()).float()
    self.remissions = torch.from_numpy(self.remissions.transpose()).float()

    # pass proj to pytorch in [1, H, W] shape (channel first)
    self.proj_range = torch.from_numpy(self.proj_range).float()
    self.unproj_range = torch.from_numpy(self.unproj_range).float()
    self.proj_xyz = torch.from_numpy(
        np.transpose(self.proj_xyz, (2, 0, 1))).float()
    self.proj_remissions = torch.from_numpy(self.proj_remissions).float()
    self.proj_idx = torch.from_numpy(self.proj_idx).long()
    self.proj_mask = torch.from_numpy(self.proj_mask).float()
    # for projecting pointclouds into image
    self.proj_x = torch.from_numpy(self.proj_x).long()
    self.proj_y = torch.from_numpy(self.proj_y).long()

  def numpy(self):
    # pass scan to numpy in [m, 3] shape (channel last)
    self.points = self.points.t_().cpu().numpy().astype(np.float32)
    self.remissions = self.remissions.t_().cpu().numpy().astype(np.float32)

    # pass proj to numpy in [H, W] shape
    self.proj_range = self.proj_range.cpu().numpy().astype(np.float32)
    self.unproj_range = self.unproj_range.cpu().numpy().astype(np.float32)
    self.proj_xyz = self.proj_xyz.cpu().numpy().astype(np.float32)
    self.proj_xyz = np.transpose(self.proj_xyz, (2, 0, 1))
    self.proj_remissions = self.proj_remission.cpu().numpy().astype(np.float32)
    self.proj_idx = self.proj_idx.cpu().numpy().astype(np.int32)
    self.proj_mask = self.proj_mask.cpu().numpy().astype(np.float32)
    self.proj_x = self.proj_x.cpu().numpy().astype(np.int32)
    self.proj_y = self.proj_y.cpu().numpy().astype(np.int32)


class SemLaserScan(LaserScan):
  """Class that contains LaserScan with x,y,z,r,label,color_label"""
  EXTENSIONS_LABEL = ['.label']

  def __init__(self, H, W, nclasses, color_dict=None, transformation=None,
               beam_angles=None):
    super(SemLaserScan, self).__init__(H, W, transformation, beam_angles)
    self.reset()
    self.nclasses = nclasses         # number of classes
    self.color_dict = color_dict

    # make colors
    max_key = 0
    for key, data in color_dict.items():
      if key + 1 > max_key:
        max_key = key + 1
    self.color_lut = np.zeros((max_key + 100, 3), dtype=np.float32)
    for key, value in color_dict.items():
      self.color_lut[key] = np.array(value, np.float32) / 255.0

  def reset(self):
    """ Reset scan members. """
    super(SemLaserScan, self).reset()
    self.label = np.zeros((0, ), dtype=np.uint32)               # [m, 1]: label
    self.label_image = np.zeros((0, ), dtype=np.uint32)         # [m, 1]: label
    self.label_color_image = np.zeros((0, 3), dtype=np.uint32)  # [m, 1]: label
    self.label_color = np.zeros((0, 3), dtype=np.float32)       # [m ,3]: color
    # projection color with labels
    self.proj_label = np.zeros((self.proj_H, self.proj_W),
                               dtype=np.int32)              # [H,W]  label
    self.proj_color = np.zeros((self.proj_H, self.proj_W, 3),
                               dtype=np.float)              # [H,W,3] color

  def open_label(self, filename):
    """ Open raw scan and fill in attributes
    """
    # check filename is string
    if not isinstance(filename, str):
      raise TypeError("Filename should be string type, "
                      "but was {type}".format(type=str(type(filename))))

    # check extension is a laserscan
    if not any(filename.endswith(ext) for ext in self.EXTENSIONS_LABEL):
      raise RuntimeError("Filename extension is not valid label file.")

    # if all goes well, open label
    label = np.fromfile(filename, dtype=np.uint32)
    label = label.reshape((-1))

    # only fill in attribute if the right size
    if label.shape[0] == self.points.shape[0]:
      self.label = label & 0xFFFF  # semantic label in lower half
    else:
      raise ValueError("Scan and Label don't contain same number of points")

    # self.do_label_projection()

  def open_label_append(self, filename):
    """ Open raw scan and fill in attributes and appends it
    """
    # check filename is string
    if not isinstance(filename, str):
      raise TypeError("Filename should be string type, "
                      "but was {type}".format(type=str(type(filename))))

    # check extension is a laserscan
    if not any(filename.endswith(ext) for ext in self.EXTENSIONS_LABEL):
      raise RuntimeError("Filename extension is not valid label file.")

    # if all goes well, open label
    label = np.fromfile(filename, dtype=np.uint32)
    label = label.reshape((-1))

    # only fill in attribute if the right size
    # if label.shape[0] == self.points.shape[0]:
    #   self.label = label
    # else:
    #   raise ValueError("Scan and Label don't contain same number of points")
    # print(self.label.shape, 'vs', label.shape)
    if self.label.size == 0:
      self.label = label
    else:
      self.label = np.concatenate((self.label, label))

    # if self.project:
      # self.do_label_projection()

  def set_label(self, label):
    """ Set points for label not from file but from np
    """
    # check label makes sense
    if not isinstance(label, np.ndarray):
      raise TypeError("Label should be numpy array")

    # only fill in attribute if the right size
    if label.shape[0] == self.points.shape[0]:
      self.label = label
    else:
      raise ValueError("Scan and Label don't contain same number of points")

    self.do_label_projection()

  def colorize(self):
    """ Colorize pointcloud with the color of each semantic label
    """
    self.label_color = self.color_lut[self.label]
    self.label_color = self.label_color.reshape((-1, 3))

  def do_label_projection(self):
    # only map colors to labels that exist
    mask = self.proj_idx >= 0
    self.proj_label[mask] = self.label[self.proj_idx[mask]]
    self.proj_color[mask] = self.color_lut[self.label[self.proj_idx[mask]]]

  def remove_class(self, class_index):
    keep_index = self.label != class_index
    self.points = self.points[keep_index]
    self.remissions = self.remissions[keep_index]
    self.label = self.label[keep_index]
    self.label_color = self.label_color[keep_index]

  def remove_classes(self, classes):
    remove_index = np.full((len(self.points),), False)
    for c in classes:
      remove_index += self.label == c

    # invert to get the keep index
    keep_index = np.invert(remove_index)

    # remove from all data variables
    self.points = self.points[keep_index]
    self.remissions = self.remissions[keep_index]
    self.label = self.label[keep_index]
    self.label_color = self.label_color[keep_index]

  def do_label_projection_new(self):
    # only map colors to labels that exist
    mask = self.index >= 0
    self.proj_label[mask] = self.label[self.index[mask]]
    self.proj_color[mask] = self.color_lut[self.label[self.index[mask]]]

  def get_bnds(self):
    amin = np.amin(self.points, axis=0)
    amax = np.amax(self.points, axis=0)
    return np.concatenate((amin.reshape(3, 1), amax.reshape(3, 1)), axis=1)

  def get_label_map(self):
    """ converts RGB label to single sequential label index
    """
    label_map = np.ndarray(shape=self.proj_color.shape[:2], dtype=int)
    label_map[:, :] = -1
    i = 0
    for idx, rgb in self.color_dict.items():
      label_map[((self.proj_color * 255).astype(np.uint8) == rgb).all(2)] = i
      i += 1  # encode label as sequential number
    return label_map

  def convert_color_to_label(self):
    """ converts RGB label to original label index
    """
    label_map = np.ndarray(shape=self.proj_color.shape[:2], dtype=int)
    label_map[:, :] = -1
    for idx, rgb in self.color_dict.items():
      label_map[((self.proj_color * 255).astype(np.uint8) == rgb).all(2)] = idx
    return label_map

  def torch(self):
    super(SemLaserScan, self).torch()
    # pass label to pytorch in [1, m] shape (channel first)
    self.label = torch.from_numpy(self.label).long()

    # pass proj label to pytorch in [1, H, W] shape (channel first)
    self.proj_label = torch.from_numpy(self.proj_label).long()

  def numpy(self):
    super(SemLaserScan, self).numpy()
    # pass scan to numpy in [m] shape
    self.label = self.label.cpu().numpy().astype(np.float)

    # pass proj to pytorch in [H, W] shape
    self.proj_label = self.proj_label.cpu().numpy().astype(np.int32)


class MultiSemLaserScan():
  """Class that contains multiple LaserScans with x,y,z,r,label,color_label"""

  def __init__(self, H, W, nscans, nclasses, ignore_classes, moving_classes,
               fov_up, fov_down, color_dict=None, transformation=None,
               beam_angles=None, preserve_float=False,
               voxel_size=0.1, vol_bnds=None):
    self.H = H
    self.W = W
    self.nscans = nscans
    self.nclasses = nclasses
    self.ignore_classes = ignore_classes
    self.moving_classes = moving_classes
    self.fov_up = fov_up
    self.fov_down = fov_down
    self.color_dict = color_dict
    self.transformation = transformation
    self.beam_angles = beam_angles
    self.preserve_float = preserve_float
    self.voxel_size = voxel_size
    self.poses = np.zeros((nscans, 4, 4), dtype=np.float32)
    self.vol_bnds = vol_bnds
    self.scans = []
    for n in range(self.nscans):
      self.scans.append(SemLaserScan(H, W, nclasses, color_dict,
                        transformation, beam_angles))

    self.reset()

  def reset(self):
    """ Reset scan members. """
    for scan in self.scans:
      scan.reset()

  def get_scan(self, idx):
    return self.scans[idx]

  def open_multiple_scans(self, scan_names, label_names, poses, idx):
    """ Open multiple raw scan and fill in attributes
    """
    self.reset()

    if self.nscans > 1:
      # use also previous scans
      number_of_prev_scans = self.nscans // 2
      number_of_next_scans = self.nscans - number_of_prev_scans
      relative_idx = np.arange(-number_of_prev_scans, number_of_next_scans)

      # move 0 to last in index to clean moving classes
      relative_idx = np.delete(relative_idx, np.where(relative_idx == 0), 0)
      # relative_idx = np.append(relative_idx, 0)
      relative_idx = np.insert(relative_idx, 0, 0)

      for i, scan in enumerate(self.scans):
        scan_idx = idx + relative_idx[i]
        print("Open scan %d/%d %d:%d" % (i + 1, self.nscans, relative_idx[i],
                                         scan_idx + 1))
        self.poses[i] = scan.pose = poses[scan_idx]
        scan.open_scan(scan_names[scan_idx], self.fov_up, self.fov_down)
        scan.open_label(label_names[scan_idx])
        scan.colorize()
        scan.apply_pose()

        # remove moving classes from all but primary scan
        if i != 0:
          scan.remove_classes(self.moving_classes)

        # remove class unlabeled (0), outlier (1)
        scan.remove_classes(self.ignore_classes)

    else:  # use a single scan
      self.scans[0].open_scan(scan_names[idx], self.fov_up, self.fov_down)
      self.scans[0].open_label(label_names[idx])
      self.scans[0].colorize()
      self.poses[0] = self.scans[0].pose = poses[idx]
      self.scans[0].apply_pose()

      # remove class unlabeled (0), outlier (1)
      self.scans[0].remove_classes(self.ignore_classes)

  def deform(self, adaption, poses, idx):
    """ Deforms laserscan with specified adaption method and transformation
    """
    # assert(self.scan_idx.shape[0] == self.points.shape[0])
    # assert(self.label.shape[0] == self.points.shape[0])
    self.adaption = adaption
    # print(self.transformation)
    # different approaches for point cloud adaption
    if adaption == 'cp':  # closest point
      self.merged = SemLaserScan(self.H, self.W, self.nclasses,
                                 self.color_dict, self.transformation,
                                 self.beam_angles)
      self.merged.reset()
      self.merged.pose = poses[idx]

      # Merge scans into a single scan
      for scan in self.scans:
        self.merged.points = np.concatenate((self.merged.points, scan.points))
        self.merged.remissions = np.concatenate((self.merged.remissions,
                                                 scan.remissions))
        self.merged.label = np.concatenate((self.merged.label, scan.label))
        self.merged.label_color = np.concatenate((self.merged.label_color,
                                                  scan.label_color))

        # After accumulating all scans undo the pose and do range projection
      self.merged.apply_inv_pose()
      self.merged.do_range_projection_new(self.fov_up, self.fov_down,
                                          remove=True)
      self.merged.do_label_projection_new()
      self.merged.do_reverse_projection_new(self.fov_up, self.fov_down,
                                            preserve_float=self.preserve_float)

      # Map from merged to self
      self.back_points = self.merged.back_points
      self.proj_range = self.merged.proj_range
      self.proj_remissions = self.merged.proj_remissions
      self.proj_color = self.merged.proj_color
      self.label_image = self.merged.label_image
      self.index = self.merged.index
      self.label_color = self.merged.label_color_image.reshape(-1, 3)

      return [], [], []

    elif adaption == 'mesh':
      vol_bnds = self.vol_bnds
      # Automatically set voxel bounds by examining the complete point cloud
      if vol_bnds.all() is None:
        vol_bnds = np.zeros((3, 2), dtype=np.int32)
        for scan in self.scans:
          bnds = scan.get_bnds()
          vol_bnds = np.concatenate(
              (np.minimum(bnds[:, 0], vol_bnds[:, 0]).reshape(3, 1),
               np.maximum(bnds[:, 1], vol_bnds[:, 1]).reshape(3, 1)), axis=1)
        print(vol_bnds)
      t0_elapse = time.time()
      for i, scan in enumerate(self.scans):
        print("Create range image %d/%d" % (i + 1, self.nscans))
        # Undo the primary pose and then do range projection
        scan.apply_transformation(np.linalg.inv(poses[idx]))
        scan.do_range_projection_new(self.fov_up, self.fov_down, remove=True)
        scan.do_label_projection_new()
        # TODO Limit range of scans instead of limit the voxel grid volume
      fps = self.nscans / (time.time() - t0_elapse)
      print("Average FPS: %.2f" % (fps))

      print("Initializing voxel volume...")
      tsdf_vol = fl.TSDFVolume(vol_bnds, voxel_size=self.voxel_size,
                               fov_up=self.fov_up, fov_down=self.fov_down)

      t0_elapse = time.time()
      for i, scan in enumerate(self.scans):
        print("Fusing scan %d/%d" % (i + 1, self.nscans))
        # Pose transformation already applied
        proj_label3 = np.zeros(scan.proj_color.shape)
        proj_label3[:, :, 0] = scan.proj_label
        tsdf_vol.integrate(proj_label3, scan.proj_range,
                           scan.proj_remissions, np.eye(3),
                           obs_weight=1.)
      fps = self.nscans / (time.time() - t0_elapse)
      print("Average FPS: %.2f" % (fps))

      rays = self.create_rays()
      origin = np.array([0, 0, 0]).astype(np.float32)
      t0_elapse = time.time()
      self.back_points, label_color, \
          verts, colors, faces, self.proj_range, rem_image = \
          tsdf_vol.throw_rays_at_mesh(rays, origin, self.H, self.W,
                                      self.scans[0].color_lut)

      self.proj_color = label_color.reshape(self.H, self.W, 3)
      self.label_color = self.scans[0].color_lut[label_color[:, 2]]
      self.label = np.copy(self.proj_color[:, :, 2])
      self.proj_color = self.scans[0].color_lut[self.label]
      colors = self.scans[0].color_lut[colors[:, 2]]

      rps = len(rays) / (time.time() - t0_elapse)
      print("Average Rays per sec: %.2f" % (rps))
      return verts, colors, faces

    # Adapt mesh from merged scans
    elif adaption == 'mergemesh':
      # TODO ? use source parameter for range image creation, but for raytracing the target one
      self.merged = SemLaserScan(self.H, self.W, self.nclasses,
                                 self.color_dict, self.transformation,
                                 self.beam_angles)
      self.merged.reset()
      self.merged.pose = poses[idx]

      # Merge scans into a single scan
      for scan in self.scans:
        self.merged.points = np.concatenate((self.merged.points, scan.points))
        self.merged.remissions = np.concatenate((self.merged.remissions,
                                                 scan.remissions))
        self.merged.label = np.concatenate((self.merged.label, scan.label))
        self.merged.label_color = np.concatenate((self.merged.label_color,
                                                  scan.label_color))
      print(np.amax(self.merged.points, axis=0))
      print(np.amin(self.merged.points, axis=0))
      # After accumulating all scans undo the pose and do range projection
      self.merged.apply_inv_pose()

      # TODO maybe apply here transformation or inverse trafo for different sensor position
      print(self.fov_down, self.fov_up)

      self.merged.do_range_projection_new(self.fov_up, self.fov_down,
                                          remove=True)
      self.merged.do_label_projection_new()

      print("Merged proj H", self.merged.proj_H)
      print("self H", self.H)

      # Get voxel bounds by examining the complete point cloud
      merged_bnds = np.rint(self.merged.get_bnds()).astype(int)

      # Limit voxel bounds by given max bounds
      vol_bnds = self.vol_bnds
      vol_bnds[:, 0] = np.maximum(vol_bnds[:, 0], merged_bnds[:, 0])
      vol_bnds[:, 1] = np.minimum(vol_bnds[:, 1], merged_bnds[:, 1])

      print("Initializing voxel volume...")
      tsdf_vol = fl.TSDFVolume(vol_bnds, voxel_size=self.voxel_size,
                               fov_up=self.fov_up, fov_down=self.fov_down)

      t0_elapse = time.time()
      proj_label3 = np.zeros(self.merged.proj_color.shape)
      proj_label3[:, :, 0] = self.merged.proj_label
      tsdf_vol.integrate(proj_label3, self.merged.proj_range,
                         self.merged.proj_remissions, np.eye(3), obs_weight=1.)
      fps = 1.0 / (time.time() - t0_elapse)
      # print("Average FPS: %.2f" % (fps))

      # TODO use now target settings?
      t_H = int(self.H)
      t_W = int(self.W)
      print("target dim:", t_H, t_W)
      rays = self.create_rays(self.fov_up, self.fov_down, self.H, self.W)
      # rays, origin = self.create_rays_moving(self.fov_up, self.fov_down, self.H, self.W)
      # TODO expose origin or use given translation to use as origin
      origin = np.array([0, 0, 0]).astype(np.float32)
      # origin = np.array([0, 0, 0.5]).astype(np.float32)
      print("Origin:", origin)
      t0_elapse = time.time()
      self.back_points, label_color, \
          verts, colors, faces, self.proj_range, self.proj_remissions = \
          tsdf_vol.throw_rays_at_mesh(rays, origin, t_H, t_W,
                                      self.scans[0].color_lut)

      # proj_color    H x W x 3   [0,1]   -> colors for projected label image
      # label_color   back x 3    [0,1]   -> same as above, but same dim as points
      # label         H x W x 1   [0,259] -> label index for export
      #     label_image   H x W x 1   [0,259] -> same as above but image dims
      # colors        verts x 3   [0,1]   -> colors for mesh visualisation

      self.proj_color = label_color.reshape(t_H, t_W, 3)
      self.label_color = self.merged.color_lut[label_color[:, 2]]
      self.label_image = np.copy(self.proj_color[:, :, 2])
      self.proj_color = self.merged.color_lut[self.label_image]  # [0,1]
      colors = self.merged.color_lut[colors[:, 2]]
      rps = len(rays) / (time.time() - t0_elapse)
      # print("Average Rays per sec: %.2f" % (rps))
      return verts, colors, faces

    elif adaption == 'catmesh':
      # TODO Category Mesh
      quit()

    else:
      # Error
      print("\nAdaption method not recognized or not defined")
      quit()

  def get_label_map(self):
    """ converts RGB label to single sequential label index
    """
    proj_color = self.label_color.reshape(self.H, self.W, 3)
    label_map = np.ndarray(shape=proj_color.shape[:2], dtype=int)
    label_map[:, :] = -1
    i = 0
    for idx, rgb in self.color_dict.items():
      label_map[(proj_color.astype(np.uint8) == rgb).all(2)] = i
      i += 1  # encode label as sequential number
    return label_map

  def create_rays_moving(self, fov_up, fov_down, H, W):

    # TODO pass parameter for rays

    # TODO pass beam_angles

    # TODO Option to add noise to angle values

    # NOT WORKING Adapt speed from poses to reflect rolling shutter in rays

    beams = np.zeros((H, W, 3)).astype(np.float32)
    origin = np.zeros((H, W, 3)).astype(np.float32)
    origin_final = np.array([0, 1, 0]).astype(np.float32)
    # correct initial rotation of sensor
    initial = 180
    yaw_angles = (np.linspace(0, 360, W) + initial)
    larger = yaw_angles > 360
    yaw_angles[larger] -= 360
    yaw_angles = yaw_angles / 180. * np.pi
    pitch = np.linspace(fov_up, fov_down, H) / 180. * np.pi
    pitch = np.pi / 2 - pitch
    # point_z = np.cos(pitch)  #* np.ones(yaw.shape)
    # point_z = point_z.reshape(H, 1)
    distortion = np.eye(4)
    delay = 0
    for w, yaw in enumerate(yaw_angles):
      for h, p in enumerate(pitch):
        # point_x = np.sin(p) * np.cos(-yaw)
        # point_y = np.sin(p) * np.sin(-yaw)
        # point_z = np.cos(pitch)
        # point_x = point_x.reshape(H, 1)
        # point_y = point_y.reshape(H, 1)
        # single_column = np.concatenate((point_x, point_y, point_z,
        #                                 np.ones((H, 1))), axis=1)
        point = np.array([np.sin(p) * np.cos(-yaw),
                          np.sin(p) * np.sin(-yaw),
                          np.cos(p), 1.0]).astype(np.float32)
        point_m = np.matmul(distortion, point.T).T
      # Apply distortion
        # distortion[0, 0] = np.cos(delay)
        # distortion[0, 1] = - np.sin(delay)
        # distortion[1, 0] = np.sin(delay)
        # distortion[1, 1] = np.cos(delay)
      # distortion[0, 2] += .05
      # print(single_column[0, 0:3])
      # single_column = np.matmul(distortion, single_column.T)
      # print(single_column[0, 0:3])
      # beams.append(single_column[:, 0:3])
      # beams.append(single_column[0:3, :].T)
        beams[h, w, :] = point_m[0:3]
        origin[h, w, :] = origin_final * (h + 1) / (360)
      # delay += .1 / 180. * np.pi
    beams = np.array(beams).reshape(H * W, -1)
    return np.ascontiguousarray(beams.astype(np.float32)), np.ascontiguousarray(origin.astype(np.float32))

  def create_rays(self, fov_up, fov_down, H, W):

    # TODO pass beam_angles

    # TODO Option to add noise to angle values

    beams = []

    # correct initial rotation of sensor
    initial = 180
    yaw_angles = (np.linspace(0, 360, W) + initial)
    larger = yaw_angles > 360
    yaw_angles[larger] -= 360
    yaw_angles = yaw_angles / 180. * np.pi
    pitch = np.linspace(fov_up, fov_down, H) / 180. * np.pi
    pitch = np.pi / 2 - pitch
    yaw = yaw_angles
    for p in pitch:
      point_x = np.sin(p) * np.cos(-yaw)
      point_y = np.sin(p) * np.sin(-yaw)
      point_z = np.cos(p) * np.ones(yaw.shape)
      point_x = point_x.reshape(W, 1)
      point_y = point_y.reshape(W, 1)
      point_z = point_z.reshape(W, 1)
      single_row = np.concatenate((point_x, point_y, point_z), axis=1)
      beams.append(single_row)
    beams = np.array(beams).reshape(W * H, -1)
    return np.ascontiguousarray(beams.astype(np.float32))

  def write(self, out_dir, idx, range_image=False):
    # write range_image and label_image as pngs
    # imageio.imwrite(os.path.join(out_dir, "velodyne_png", str(idx).zfill(6) + ".png"),
    #                 (viridis_colors[..., ::-1] * 255).astype(np.uint8))
    # imageio.imwrite(os.path.join(out_dir, "labels_png", str(idx).zfill(6) + ".png"),
    #                 (self.proj_color[..., ::-1] * 255).astype(np.uint8))

    if self.adaption == 'cp':  # closest point
      back_points = self.merged.back_points.reshape(-1, 3)
      label_image = self.merged.label_image.reshape(-1)
      remissions = self.merged.proj_remissions.reshape(-1)

      # TODO necessary?
      # Only write back_points which are valid (not black/unlabeled)
      index = self.merged.index.reshape(-1,) > 0
      back_points = back_points[index]
      remissions = remissions[index]
      label_image = label_image[index].astype(np.int32)
    else:
      back_points = self.back_points.reshape(-1, 3)
      label_image = self.label_image.reshape(-1)
      remissions = self.proj_remissions.reshape(-1)

    index = label_image >= 0  # TODO there should be no -1
    back_points = back_points[index]
    remissions = remissions[index]
    label_image = label_image[index].astype(np.int32)
    keep = np.sum(back_points, axis=1) != 0  # remove points with (0, 0, 0)
    back_points = back_points[keep]
    remissions = remissions[keep]
    label_image = label_image[keep]

    if self.adaption == 'cp':
      assert(back_points.shape[0] == label_image.shape[0] == remissions.shape[0])
    else:
      assert(back_points.shape[0] == label_image.shape[0])

    scan_file = open(
        os.path.join(out_dir, "velodyne", str(idx).zfill(6) + ".bin"), "wb")
    for i, point in enumerate(back_points):
        # print(point[0], point[1], point[2])
        byte_values = struct.pack("ffff", point[0], point[1], point[2],
                                  remissions[i])
        scan_file.write(byte_values)
    scan_file.close()

    # write labels
    label_file = open(
        os.path.join(out_dir, "labels", str(idx).zfill(6) + ".label"), "wb")
    for label in label_image:
      # print(label.dtype, label.shape, label.astype(np.int32))
      byte_values = struct.pack("I", label)
      label_file.write(byte_values)
    label_file.close()


def compare(scan_source, scan_target):
  """ Compare two scans by examine labels, range and remissions
  """
  # Label intersection image
  source_color = scan_source.proj_color
  # source_label_map = scan_source.get_label_map()
  source_label = scan_source.proj_label

  if scan_target.adaption == 'cp':
    target_label = scan_target.merged.proj_label
    target_color = scan_target.merged.proj_color
  else:
    target_color = scan_target.proj_color
    target_label = np.copy(scan_target.label_image)

  assert(source_color.size == target_color.size)
  assert(source_label.size == target_label.size)

  # Mask out no data (= black) in target scan
  black_values = np.sum(source_color, axis=2) == 0
  source_label[black_values] = 0
  target_label[black_values] = 0
  black_values_3 = np.repeat(black_values[:, :, np.newaxis], 3, axis=2)
  target_color[black_values_3] = 0

  bg_label = source_label == 0
  bg_label_3 = np.repeat(bg_label[:, :, np.newaxis], 3, axis=2)
  target_label[bg_label] = 0
  target_color[bg_label_3] = 0

  label_diff = abs(source_color - target_color)

  # Ignore empty classes
  unique_values = np.union1d(np.unique(source_label), np.unique(target_label))

  for i, value in enumerate(unique_values):
    mask_source = source_label == value
    mask_target = target_label == value
    source_label[mask_source] = i
    target_label[mask_target] = i

  unique_values = np.union1d(np.unique(source_label), np.unique(target_label))
  empty = np.isin(np.arange(scan_source.nclasses), unique_values,
                  invert=True)

  # Evaluate by label
  eval = iouEval(scan_source.nclasses,
                 np.arange(scan_source.nclasses)[empty])
  eval.addBatch(target_label, source_label)
  m_iou, iou = eval.getIoU()
  print("IoU class: ", (iou * 100).astype(int))
  m_acc = eval.getacc()
  print("IoU: ", m_iou)
  print("Acc: ", m_acc)

  # Range diff image
  source_range = scan_source.proj_range
  target_range = scan_target.proj_range
  # Mask out no data (= black) in target scan
  black = source_range == 0
  # target_range[black] = 0

  # Mask out too far data in target scan
  # too_far = source_range >= 80
  # source_range[too_far] = 0
  # target_range[too_far] = 0

  range_diff = (source_range - target_range) ** 2
  # data = self.convert_range(range_diff)
  MSE = range_diff.sum() / range_diff.size
  print("MSE: ", MSE)

  source_remissions = scan_source.proj_remissions
  target_remissions = scan_target.proj_remissions
  remissions_diff = (source_remissions - target_remissions) ** 2

  return label_diff, range_diff, remissions_diff, m_iou, m_acc, MSE
