#!/usr/bin/python3

import io
import sys
import os
from pathlib import Path
import struct
import scipy.io
import math
import numpy as np
from numpy.linalg import inv

def write_calibration(output_directory):
  f = open("{}/calib.txt".format(output_directory), "w")

  f.write("Tr: 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0")

  f.close()

def rotxyz(r, p, h):
  """ Compute rotation matrix about the XYZ-axes.

      R = rotxyz(rph) returns a 3x3 rotation matrix R where (r,p,h) is a 3-vector
      of Euler angles (roll, pitch, heading) measured in radians.
  """

  cr = math.cos(r); sr = math.sin(r)
  cp = math.cos(p); sp = math.sin(p)
  ch = math.cos(h); sh = math.sin(h)

  R = np.array([[ch*cp, (-sh*cr + ch*sp*sr), ( sh*sr + ch*sp*cr)], \
                  [sh*cp, ( ch*cr + sh*sp*sr), (-ch*sr + sh*sp*cr)], \
                  [-sp,         cp*sr,                cp*cr       ]])
  return R

def rotZ(alpha, N = 3):
  """ Get rotation matrix (of dim N x N) about z-axis with angle alpha in randians. """
  R = np.identity(N)
  R[0,0] = math.cos(alpha)
  R[0,1] = -math.sin(alpha)
  R[1,0] = math.sin(alpha)
  R[1,1] = math.cos(alpha)

  return R

def initialize_firstpose(filename):
  mat = scipy.io.loadmat(filename)
  xyzrph = mat["SCAN"]["X_wv"][0,0]
  # print(xyzrph)
  T = np.identity(4)
  T[0:3, 0:3] = rotxyz(*xyzrph[3:6])
  T[0:3, 3:4] = xyzrph[0:3]

  first_pose = np.linalg.inv(T)

  return first_pose

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Convert Michigan Velodyne files to KITTI compatible binary format and folder structure.", file = sys.stderr)
    print("Usage: convert <source> <sequence-id>", file = sys.stderr)
    print("Where <source> corresponds to the root folder of the dataset.", file = sys.stderr)

    sys.exit(1)

  input_dir = Path(sys.argv[1]) / "SCANS"
  sequence_dir = Path("dataset/sequences/{}/".format(sys.argv[2]))
  poses_file = Path("dataset/poses/{}.txt".format(sys.argv[2]))

  skip_existing = False

  if sequence_dir.exists() or poses_file.exists():
    print("Warning: Squence directory already exists.")
    answer = input("Do you want to overwrite all files? [y/s/N] ")
    if answer != "y" and answer != "s":
      print("Stopping conversion.")
      sys.exit(0)
    if answer == "y": print("Overwritting files.")
    if answer == "s":
      print("Skipping existing files.")
      skip_existing = True

  else:
    print("-- Creating directories.")
    sequence_dir.mkdir(parents = True)
    (sequence_dir / "velodyne").mkdir(parents = True)
    poses_dir = Path("dataset/poses/")
    if not poses_dir.exists(): poses_dir.mkdir(parents = True)

  write_calibration(sequence_dir.as_posix())

  scan_files = sorted([f for f in input_dir.iterdir() if f.is_file()])
  print("Converting files: ", end="")
  sys.stdout.flush()
  progress = 0

  out_poses = None
  if skip_existing:
    out_poses = poses_file.open("a")
  else:
    out_poses = poses_file.open("w")

  first_pose = initialize_firstpose(scan_files[0].as_posix())

  # Convert in KITTI Velodyne frame. (x pointing forward)
  C = rotZ(math.radians(-90.0))

  for i, f in enumerate(scan_files):
      if 100 * i / len(scan_files) > progress:
        progress += 10
        print(".", end="")
        sys.stdout.flush()

      out_filename = "{}/velodyne/{:06d}.bin".format(sequence_dir.as_posix(), i)
      if Path(out_filename).exists() and skip_existing: continue

      try:

        mat = scipy.io.loadmat(f.as_posix())

        xyzrph = mat["SCAN"]["X_wv"][0,0]
        # print(xyzrph)
        T = np.identity(4)
        T[0:3, 0:3] = rotxyz(*xyzrph[3:6])
        T[0:3, 3:4] = xyzrph[0:3]

        T = np.dot(first_pose, T)

        pose_string = "{} {} {} {}".format(*T[0, :])
        pose_string += " {} {} {} {}".format(*T[1, :])
        pose_string += " {} {} {} {}\n".format(*T[2, :])
        # print(pose_string)
        out_poses.write(pose_string)

        out_file = open(out_filename, "wb")

        M, N = mat["SCAN"]["XYZ"][0,0].shape
        for k in range(N):
          values = np.dot(C, mat["SCAN"]["XYZ"][0,0][:,k])

          byte_values = struct.pack("ffff", values[0], values[1], values[2], 0.0)
          out_file.write(byte_values)

        out_file.close()

      except Exception as e:
        print("\n\nException while processing: {}:".format(f.as_posix()), file = sys.stderr)
        print("Skipping...")
        

      else:
        pass

  while progress < 100:
    progress += 10
    print(".", end="")
  print(" finished.")

  #print(mat)
