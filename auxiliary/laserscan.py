#!/usr/bin/env python3
import numpy as np
import torch


class LaserScan:
  """Class that contains LaserScan with x,y,z,r"""
  EXTENSIONS_SCAN = ['.bin']

  def __init__(self, H, W):
    self.proj_H = H
    self.proj_W = W
    self.reset()

  def reset(self):
    """ Reset scan members. """
    self.points = np.zeros((0, 3), dtype=np.float32)        # [m, 3]: x, y, z
    self.remissions = np.zeros((0, 1), dtype=np.float32)    # [m ,1]: remission

    # a scan can also be represented as a spherical projection depth
    # image, and another channel which saves for each pixel, to which
    # in the point it corresponds
    self.proj_range = np.full((self.proj_H, self.proj_W), -1,
                              dtype=np.float32)    # [H,W] range (-1 is no data)
    self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1,
                            dtype=np.float32)    # [H,W,3] xyz coord (-1 is no data)
    self.proj_remission = np.full((self.proj_H, self.proj_W), -1,
                                  dtype=np.float32)    # [H,W] intensity (-1 is no data)
    self.proj_idx = np.full((self.proj_H, self.proj_W), -1,
                            dtype=np.int32)       # [H,W] index (-1 is no data)
    self.proj_x = np.zeros((0, 1), dtype=np.float32)        # [m, 1]: x
    self.proj_y = np.zeros((0, 1), dtype=np.float32)        # [m, 1]: y
    self.unproj_range = np.zeros(
        (0, 1), dtype=np.float32)        # [m, 1]: range
    self.proj_mask = np.zeros(
        (self.proj_H, self.proj_W), dtype=np.int32)       # [H,W] mask

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
    # print(pose.shape, "x", hom_points.T.shape)
    # print(pose.shape, "x", np.matmul(hom_points, pose).shape)
    t_points = np.matmul(pose, hom_points.T).T
    # t_points = np.matmul(np.linalg.inv(pose), np.matmul(pose, hom_points.T)).T
    # t_points = np.matmul(np.matmul(hom_points, pose), np.linalg.inv(pose)).T
    # print(t_points.shape)

    # TODO Apply given transformation to move sensor to specific position/angle

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

    # scan[:, 1] += 4 # DEMO TRANSFORMATION!!!!
    # scan[:, 0] -= 4 # DEMO TRANSFORMATION!!!!

    # put in attribute
    self.points = scan[:, 0:3]    # get xyz
    self.remissions = scan[:, 3]  # get remission

    # if projection is wanted, then do it and fill in the structure
    self.do_range_projection(fov_up, fov_down)

  def remove_points(self, keep_index):
    self.points = self.points[keep_index]
    self.remissions = self.remissions[keep_index]
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
    depth = depth[keep_index]
    self.points = self.points[keep_index]

    # get scan components
    scan_x = self.points[:, 0]
    scan_y = self.points[:, 1]
    scan_z = self.points[:, 2]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)
    # TODO use array of hardcoded angles
    # np.nan_to_num(pitch, copy=False) # in case its divided by zero not needed

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)          # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]
    
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
    # copy of depth in original order
    self.unproj_range = np.copy(depth)

    # order in decreasing depth
    indices = np.arange(depth.shape[0]) # indices of original points sorted by range
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    indices = indices[order]
    points = self.points[order]
    remission = self.remissions[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    # assing to images
    self.proj_range[proj_y, proj_x] = depth
    self.proj_xyz[proj_y, proj_x] = points
    self.proj_remission[proj_y, proj_x] = remission
    self.proj_idx[proj_y, proj_x] = indices
    self.proj_mask = (self.proj_idx > 0).astype(np.float32)

  def torch(self):
    # pass scan to pytorch in [3, m] shape (channel first)
    self.points = torch.from_numpy(self.points.transpose()).float()
    self.remissions = torch.from_numpy(self.remissions.transpose()).float()

    # pass proj to pytorch in [1, H, W] shape (channel first)
    self.proj_range = torch.from_numpy(self.proj_range).float()
    self.unproj_range = torch.from_numpy(self.unproj_range).float()
    self.proj_xyz = torch.from_numpy(
        np.transpose(self.proj_xyz, (2, 0, 1))).float()
    self.proj_remission = torch.from_numpy(self.proj_remission).float()
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
    self.proj_remission = self.proj_remission.cpu().numpy().astype(np.float32)
    self.proj_idx = self.proj_idx.cpu().numpy().astype(np.int32)
    self.proj_mask = self.proj_mask.cpu().numpy().astype(np.float32)
    self.proj_x = self.proj_x.cpu().numpy().astype(np.int32)
    self.proj_y = self.proj_y.cpu().numpy().astype(np.int32)


class SemLaserScan(LaserScan):
  """Class that contains LaserScan with x,y,z,r,label,color_label"""
  EXTENSIONS_LABEL = ['.label']

  def __init__(self, H, W, nclasses, color_dict=None):
    super(SemLaserScan, self).__init__(H, W)
    self.reset()
    self.nclasses = nclasses         # number of classes

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
    self.label = np.zeros((0, 1), dtype=np.float32)         # [m, 1]: label
    self.label_color = np.zeros((0, 3), dtype=np.float32)   # [m ,3]: color
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
      self.label = label
    else:
      raise ValueError("Scan and Label don't contain same number of points")

    self.do_label_projection()

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


class MultiSemLaserScan(SemLaserScan):
  """Class that contains multiple LaserScans with x,y,z,r,label,color_label",scan_index"""

  def __init__(self, H, W, nclasses, adaption, color_dict=None):
    super(MultiSemLaserScan, self).__init__(H, W, nclasses, color_dict)
    self.adaption = adaption
    self.reset()
    # self.scans

  def reset(self):
    """ Reset scan members. """
    super(MultiSemLaserScan, self).reset()

  def open_multiple_scans(self, scan_names, label_names, poses, idx, number_of_scans, fov_up, fov_down):
      """ Open multiple raw scan and fill in attributes
      """
      self.reset()
      
      # TODO use also previous scans
      for i in range(number_of_scans):
        scan_idx = idx + i
        # scan_idx = (idx + i) % len(scan_names)
        # print(i, idx, scan_idx, len(scan_names))
        super(MultiSemLaserScan, self).open_scan_append(scan_names[scan_idx], poses[scan_idx], fov_up, fov_down)
        super(MultiSemLaserScan, self).open_label_append(label_names[scan_idx])
        super(MultiSemLaserScan, self).colorize()
      # print(len(self.points), "points")

      # After accumulating all scans then undo the pose and then do range projection
      hom_points = np.ones((self.points.shape[0], 4))
      hom_points[:, 0:3] = self.points[:, 0:3]
      t_points = np.matmul(np.linalg.inv(poses[idx]), hom_points.T).T
      self.points[:, 0:3] = t_points[:, 0:3]

      # different approaches for point cloud adaption
      if self.adaption == 'cp': # closest point
        self.do_range_projection(fov_up, fov_down, remove=True)
        self.do_label_projection()
      elif self.adaption == 'mesh':
        # TODO
        quit()
      elif self.adaption == 'catmesh':
        # TODO
        quit()
      else:
        # Error
        print("\nAdaption method not recognized or not defined")
        quit()

      # TODO export backprojected point cloud (+ range image)
