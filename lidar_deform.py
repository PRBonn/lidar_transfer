#!/usr/bin/env python3

import argparse
import os
import time
import yaml
import numpy as np
from shutil import copy2
from auxiliary.laserscan import *
from auxiliary.laserscanvis import LaserScanVis


def parse_calibration(filename):
  """ read calibration file with given filename

      Returns
      -------
      dict
          Calibration matrices as 4x4 numpy arrays.
  """
  calib = {}

  calib_file = open(filename)
  for line in calib_file:
    key, content = line.strip().split(":")
    values = [float(v) for v in content.strip().split()]

    pose = np.zeros((4, 4))
    pose[0, 0:4] = values[0:4]
    pose[1, 0:4] = values[4:8]
    pose[2, 0:4] = values[8:12]
    pose[3, 3] = 1.0

    calib[key] = pose

  calib_file.close()

  return calib


def parse_poses(filename, calibration):
  """ read poses file with per-scan poses from given filename

      Returns
      -------
      list
          list of poses as 4x4 numpy arrays.
  """
  file = open(filename)

  poses = []

  Tr = calibration["Tr"]
  Tr_inv = np.linalg.inv(Tr)

  pose = np.eye(4)

  i = 0
  for line in file:
    if len(line.strip()) == 0: continue
    print(line)
    values = [float(v) for v in line.strip().split()]

    cur_pose = np.zeros((4, 4))
    cur_pose[0, 0:4] = values[0:4]
    cur_pose[1, 0:4] = values[4:8]
    cur_pose[2, 0:4] = values[8:12]
    cur_pose[3, 3] = 1.0

    pose = cur_pose
    poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))
    i += 1

  return poses


if __name__ == '__main__':
  parser = argparse.ArgumentParser("./lidar_deform.py")
  parser.add_argument(
      '--dataset', '-d',
      type=str,
      required=True,
      help='Dataset to adapt. No Default',
  )
  parser.add_argument(
      '--config', '-c',
      type=str,
      required=False,
      default="config/lidar_transfer.yaml",
      help='Dataset config file. Defaults to %(default)s',
  )
  parser.add_argument(
      '--sequence', '-s',
      type=str,
      default="00",
      required=False,
      help='Sequence to visualize. Defaults to %(default)s',
  )
  parser.add_argument(
      '--target', '-t',
      type=str,
      default='',
      help='Target dataset config file. Defaults to dataset config',
  )
  parser.add_argument(
      '--offset', '-o',
      type=int,
      default=0,
      required=False,
      help='Sequence to start. Defaults to %(default)s',
  )
  parser.add_argument(
      '--output', '-p',
      type=str,
      required=False,
      default="output/",
      help='Output folder to write bin files to. Defaults to %(default)s',
  )
  parser.add_argument(
      '--batch', '-b',
      action='store_true',
      required=False,
      help='Run in batch mode.',
  )
  parser.add_argument(
      '--write', '-w',
      action='store_true',
      required=False,
      help='Write new dataset to file.',
  )
  parser.add_argument(
      '--one_scan',
      action='store_true',
      required=False,
      help='Run only once.',
  )
  FLAGS, unparsed = parser.parse_known_args()

  # print summary of what we will do
  print("*" * 80)
  print("INTERFACE:")
  print("Dataset", FLAGS.dataset)
  print("Source config", FLAGS.config)
  print("Sequence", FLAGS.sequence)
  print("Target config", FLAGS.target)
  print("Offset", FLAGS.offset)
  print("Output", FLAGS.output)
  print("Batch mode", FLAGS.batch)
  print("Write mode", FLAGS.write)
  print("One_scan mode", FLAGS.one_scan)
  print("*" * 80)

  # open config file
  try:
    print("Opening config file %s" % FLAGS.config)
    CFG = yaml.safe_load(open(FLAGS.config, 'r'))
  except Exception as e:
    print(e)
    print("Error opening yaml file.")
    quit()

  # does output folder and subfolder exists?
  if FLAGS.write:
    print("Output folder is %s" % FLAGS.output)
    if os.path.isdir(FLAGS.output):
      out_path = os.path.join(FLAGS.output, "sequences", FLAGS.sequence)
      out_scan_paths = os.path.join(out_path, "velodyne")
      out_label_paths = os.path.join(out_path, "labels")
      if os.path.isdir(out_scan_paths) and os.path.isdir(out_label_paths):
        print("Output folder with subfolder exists! %s" % FLAGS.output)
        if os.listdir(out_scan_paths):
          print("Output folder velodyne is not empty! " +
                "Data will be overwritten!")
        if os.listdir(out_label_paths):
          print("Output folder label is not empty! Data will be overwritten!")
      else:
        os.makedirs(out_scan_paths)
        os.makedirs(out_label_paths)
        print("Created subfolder in output folder %s!" % FLAGS.output)

      png_scan_paths = os.path.join(out_path, "velodyne_png")
      png_label_paths = os.path.join(out_path, "labels_png")
      if os.path.isdir(png_scan_paths) and os.path.isdir(png_label_paths):
        print("Output folder with subfolder exists! %s" % FLAGS.output)
        if os.listdir(png_scan_paths):
          print("Output folder velodyne is not empty! " +
                "Data will be overwritten!")
        if os.listdir(png_label_paths):
          print("Output folder label is not empty! Data will be overwritten!")
      else:
        os.makedirs(png_scan_paths)
        os.makedirs(png_label_paths)
        print("Created subfolder in output folder %s!" % FLAGS.output)
    else:
      print("Output folder doesn't exist! Exiting...")
      quit()

  # does sequence folder exist?
  scan_paths = os.path.join(FLAGS.dataset, "sequences",
                            FLAGS.sequence, "velodyne")
  if os.path.isdir(scan_paths):
    print("Sequence folder exists! Using sequence from %s" % scan_paths)
  else:
    print("Sequence folder doesn't exist! Exiting...")
    quit()

  # get pointclouds filenames
  scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
      os.path.expanduser(scan_paths)) for f in fn]
  scan_names.sort()

  # does label folder exist?
  label_paths = os.path.join(FLAGS.dataset, "sequences",
                             FLAGS.sequence, "labels")
  if os.path.isdir(label_paths):
    print("Labels folder exists! Using labels from %s" % label_paths)
  else:
    print("Labels folder doesn't exist! Exiting...")
    quit()

  # get label filenames
  label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
      os.path.expanduser(label_paths)) for f in fn]
  label_names.sort()

  # check that there are same amount of labels and scans
  assert(len(label_names) == len(scan_names))

  print("*" * 80)

  # read config.yaml of dataset
  try:
    source_config_path = os.path.join(FLAGS.dataset, "config.yaml")
    print("Opening source config file", source_config_path)
    source_config = yaml.safe_load(open(source_config_path, 'r'))
  except Exception as e:
    print(e)
    print("Error opening source config.yaml file %s." % source_config_path)
    quit()

  # read calib.txt of dataset
  try:
    calib_file = os.path.join(FLAGS.dataset, "sequences",
                              FLAGS.sequence, "calib.txt")
    print("Opening calibration file", calib_file)
  except Exception as e:
    print(e)
    print("Error opening poses file.")
    quit()
  calib = parse_calibration(calib_file)

  # read poses.txt of dataset
  try:
    poses_file = os.path.join(FLAGS.dataset, "sequences",
                              FLAGS.sequence, "poses.txt")
    print("Opening poses file", poses_file)
  except Exception as e:
    print(e)
    print("Error opening poses file.")
    quit()
  poses = parse_poses(poses_file, calib)

  # additional parameter
  name = source_config['name']
  # projection = source_config['projection']
  fov_up = source_config['fov_up']
  fov_down = source_config['fov_down']
  # TODO change to more general description height?
  beams = source_config['beams']
  angle_res_hor = source_config['angle_res_hor']
  fov_hor = source_config['fov_hor']
  try:
    beam_angles = source_config['beam_angles']
    beam_angles.sort()
  except Exception as e:
    print("No beam angles in scan config: calculate equidistant angles")
  W = int(fov_hor / angle_res_hor)

  print("*" * 80)
  print("SCANNER:")
  print("Name", name)
  # print("Projection", projection)
  print("Resolution", beams, "x", W)
  print("FOV up", fov_up)
  print("FOV down", fov_down)
  print("Beam angles", beam_angles)
  print("*" * 80)

  # read config.yaml of target dataset
  try:
    if FLAGS.target == '':
      FLAGS.target = source_config_path
      print("Use source as target!")
    print("Opening target config file", FLAGS.target)
    target_config = yaml.safe_load(open(FLAGS.target, 'r'))
  except Exception as e:
    print(e)
    print("Error opening target yaml file %." & FLAGS.target)
    quit()

  # target parameter to deform to
  t_name = target_config['name']
  # t_projection = target_config['projection']
  t_fov_up = target_config['fov_up']
  t_fov_down = target_config['fov_down']
  t_beams = target_config['beams']  # TODO see above
  t_angle_res_hor = target_config['angle_res_hor']
  t_fov_hor = target_config['fov_hor']
  t_W = int(t_fov_hor / t_angle_res_hor)
  try:
    t_beam_angles = target_config['beam_angles']
    t_beam_angles.sort()
  except Exception as e:
    t_beam_angles = None
    print("No beam angles in target config: calculate equidistant angles")

  # Approach parameter
  adaption = CFG["adaption"]
  preserve_float = CFG["preserve_float"]
  voxel_size = CFG["voxel_size"]
  voxel_bounds = np.array(CFG["voxel_bounds"])
  nscans = CFG["number_of_scans"]
  ignore_classes = CFG["ignore"]
  moving_classes = CFG["moving"]
  transformation = CFG["transformation"]

  print("*" * 80)
  print("TARGET:")
  print("Name", t_name)
  # print("Projection", projection)
  print("Resolution", t_beams, "x", t_W)
  print("FOV up", t_fov_up)
  print("FOV down", t_fov_down)
  print("Beam angles", t_beam_angles)
  print("*" * 80)
  print("CONFIG:")
  print("Aggregate", nscans, "scans")
  print("Transformation", transformation)
  print("Adaption", adaption)
  print("Preserve Float", preserve_float)
  print("Voxel size", voxel_size)
  print("Voxel bounds", voxel_bounds)
  print("Ignore classes", ignore_classes)
  print("Moving classes", moving_classes)
  print("*" * 80)

  try:
    voxel_bounds = voxel_bounds.reshape(3, 2)
  except Exception as e:
    print("No voxel boundaries set")

  try:
    increment = CFG["batch_interval"]
  except Exception as e:
    increment = 1

  # create a scan
  color_dict = CFG["color_map"]
  nclasses = len(color_dict)

  config_saved = False

  # create a visualizer
  if FLAGS.batch is False:
    show_diff = False
    show_mesh = False
    show_range = True
    show_remissions = True
    if t_beams == beams:
      show_diff = True  # show diff only if comparison is possible
    if "mesh" in adaption:
      show_mesh = True
    vis = LaserScanVis([W, t_W], [beams, t_beams], show_diff=show_diff,
                       show_range=show_range, show_mesh=show_mesh,
                       show_remissions=show_remissions)
    vis.nframes = len(scan_names)

  # print instructions
  if FLAGS.batch is False:
    print("To navigate:")
    print("\tb: back (previous scan)")
    print("\tn: next (next scan)")
    print("\tq: quit (exit program)")

  idx = FLAGS.offset
  number_of_prev_scans = nscans // 2
  number_of_next_scans = nscans - number_of_prev_scans
  if number_of_prev_scans > idx:
    idx += number_of_prev_scans - idx
    print("Automatic offset %d" % (number_of_prev_scans))

  choice = "no"
  while True:
    if choice != "change":
      t0_elapse = time.time()
      scan = SemLaserScan(beams, W, nclasses, color_dict)
      scans = MultiSemLaserScan(source_config, target_config, nscans, nclasses,
                                ignore_classes, moving_classes, color_dict,
                                transformation=transformation,
                                preserve_float=preserve_float,
                                voxel_size=voxel_size, vol_bnds=voxel_bounds)
      # open single source scan for reference
      scan.open_scan(scan_names[idx], fov_up, fov_down)
      scan.open_label(label_names[idx])
      # scan.create_restricted_dataset(3, -25, idx, "", label=False)
      scan.colorize()
      scan.remove_classes(ignore_classes)
      scan.do_range_projection(fov_up, fov_down, remove=True)
      scan.do_label_projection()

      # open multiple source scans
      scans.open_multiple_scans(scan_names, label_names, poses, idx)

      # run approach
      verts, verts_colors, faces = scans.deform(adaption, poses, idx)
      if t_beams == beams and t_W == W:
        label_diff, range_diff, rem_diff, m_iou, m_acc, MSE = \
            compare(scan, scans)
        if FLAGS.batch is False:
          vis.set_diff(label_diff, range_diff, rem_diff, m_iou, m_acc, MSE)
      s = time.time() - t0_elapse
      print("Took: %.2fs" % (s))

    if FLAGS.batch is False:
      # pass to visualizer
      vis.frame = idx
      vis.set_source_3d(scan)

      if adaption == 'cp':  # closest point
        vis.set_data(scan, scans)
      elif adaption == 'mesh':
        vis.set_data(scan, scans, verts=verts, verts_colors=verts_colors,
                     faces=faces, W=t_W, H=t_beams)
      elif adaption == 'mergemesh':
        vis.set_data(scan, scans, verts=verts, verts_colors=verts_colors,
                     faces=faces, W=t_W, H=t_beams)
      else:
        # Error
        print("\nAdaption method not recognized or not defined")
        quit()

    # Export backprojected point cloud
    if FLAGS.write:
      scans.write(out_path, idx)

      # write config to export path
      if not config_saved:
        copy2(FLAGS.target, out_path)
        print("Target config", FLAGS.target, "copied to", out_path)
        copy2(FLAGS.config, out_path)
        print("Config", FLAGS.config, "copied to", out_path)
        config_saved = True

    if FLAGS.one_scan:
      quit()

    if FLAGS.batch:
      idx = idx + increment
      if idx >= (len(scan_names) - (nscans - 1)):
        quit()
      print("#" * 30, FLAGS.sequence, "-", idx, "/", len(scan_names),
            "#" * 30)
    else:
      # get user choice
      while True:
        choice = vis.get_action(0.01)
        if choice != "no":
          break
      if choice == "next":
        # take into account that we look further than one scan
        idx = (idx + 1) % (len(scan_names) - (nscans - 1))
        continue
      if choice == "back":
        idx -= 1
        if idx < 0:
          idx = len(scan_names) - 1
        continue
      elif choice == "change":
        continue
      elif choice == "quit":
        print()
      break
