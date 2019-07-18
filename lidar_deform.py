#!/usr/bin/env python3

import argparse
import os
import yaml
import numpy as np
from auxiliary.laserscan import LaserScan, SemLaserScan, MultiSemLaserScan
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
  FLAGS, unparsed = parser.parse_known_args()

  # print summary of what we will do
  print("*" * 80)
  print("INTERFACE:")
  print("Dataset", FLAGS.dataset)
  print("Config", FLAGS.config)
  print("Sequence", FLAGS.sequence)
  print("Target", FLAGS.target)
  print("offset", FLAGS.offset)
  print("*" * 80)

  # open config file
  try:
    print("Opening config file %s" % FLAGS.config)
    CFG = yaml.load(open(FLAGS.config, 'r'))
  except Exception as e:
    print(e)
    print("Error opening yaml file.")
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

  # read config.yaml of dataset
  try:
    scan_config_path = os.path.join(FLAGS.dataset, "sequences",
                                    FLAGS.sequence, "config.yaml")
    print("Opening config file", scan_config_path)
    scan_config = yaml.load(open(scan_config_path, 'r'))
  except Exception as e:
    print(e)
    print("Error opening yaml file.")
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
  name = scan_config['name']
  # projection = scan_config['projection']
  fov_up = scan_config['fov_up']
  fov_down = scan_config['fov_down']
  beams = scan_config['beams'] # TODO change to more general description height?
  angle_res_hor = scan_config['angle_res_hor']
  fov_hor = scan_config['fov_hor']
  try:
    beam_angles = scan_config['beam_angles']
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
      FLAGS.target = scan_config_path
    print("Opening target config file", FLAGS.target)
    target_config = yaml.load(open(FLAGS.target, 'r'))
  except Exception as e:
    print(e)
    print("Error opening yaml file.")
    quit()

  # target parameter to deform to
  t_name = target_config['name']
  # t_projection = target_config['projection']
  t_fov_up = target_config['fov_up']
  t_fov_down = target_config['fov_down']
  t_beams = target_config['beams'] # TODO change to more general description height?
  t_angle_res_hor = target_config['angle_res_hor']
  t_fov_hor = target_config['fov_hor']
  t_W = int(t_fov_hor / t_angle_res_hor)
  try:
    t_beam_angles = target_config['beam_angles']
    t_beam_angles.sort()
  except Exception as e:
    print("No beam angles in target config: calculate equidistant angles")

  # Approach parameter
  adaption = CFG["adaption"]  #['mesh', 'catmesh', 'cp']
  number_of_scans = CFG["number_of_scans"]
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
  print("Aggregate", number_of_scans, "scans")
  print("Transformation", transformation)
  print("Adaption", adaption)
  print("Ignore classes", ignore_classes)
  print("Moving classes", moving_classes)
  print("*" * 80)

  # create a scan
  color_dict = CFG["color_map"]
  nclasses = len(color_dict)
  scan = SemLaserScan(beams, W, nclasses, color_dict)
  scans = MultiSemLaserScan(t_beams, t_W, nclasses, adaption,
                            ignore_classes, moving_classes, color_dict,
                            transformation=transformation)

  # create a visualizer
  vis = LaserScanVis([W, t_W], [beams, t_beams])

  # print instructions
  print("To navigate:")
  print("\tb: back (previous scan)")
  print("\tn: next (next scan)")
  print("\tq: quit (exit program)")

  idx = FLAGS.offset
  while True:
    # open pointcloud
    scan.open_scan(scan_names[idx], fov_up, fov_down)
    scan.open_label(label_names[idx])
    scan.colorize()

    # open multiple scans
    scans.open_multiple_scans(scan_names, label_names, poses, idx,
                              number_of_scans, t_fov_up, t_fov_down)

    # Export backprojected point cloud (+ range image)
    # TODO check folder existence earlier
    scans.write(FLAGS.output, idx)

    # pass to visualizer
    vis.set_laserscan(scan)
    # vis.set_laserscan2(scan2)
    vis.set_laserscan2(scans, poses[idx])

    # get user choice
    while True:
      choice = vis.get_action(0.01)
      if choice != "no":
        break
    if choice == "next":
      # take into account that we look further than one scan
      idx = (idx + 1) % (len(scan_names) - number_of_scans)
      continue
    if choice == "back":
      idx -= 1
      if idx < 0:
        idx = len(scan_names) - 1
      continue
    elif choice == "quit":
      print()
      break
