#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys
import subprocess


def change_config(filename, key, value):
  with open(filename) as f:
    config = yaml.safe_load(f)
    config[key] = value

  with open(filename, "w") as f:
    yaml.dump(config, f)


def plot(fn, data, title, pre, fname, xdata, ydata, cmap):
  plt.figure(fn, figsize=(8, 6))
  plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
  plt.imshow(data, interpolation='nearest', cmap=cmap)
  plt.xlabel('voxel size')
  plt.ylabel('frames')
  plt.colorbar()
  plt.xticks(np.arange(len(xdata)), xdata)
  plt.yticks(np.arange(len(ydata)), ydata)
  plt.title(title)
  plt.savefig(pre + fname + ".svg")
  plt.savefig(pre + fname + ".pdf")


if __name__ == '__main__':
  p = "/automount_home_students/flanger"
  cfg_file = p + "/workspace/msc/lidar_transfer/experiments/" + \
      "grid_search_nframes_voxelsize.yaml"
  dataset = "/media/flanger/SAMS1TB_0/kitti-odometry/dataset/"
  sequence = "00"
  offset = 70
  adaption = "mergemesh"
  frames = [1, 2, 3, 4, 5, 10, 20]
  voxel_size = [.5, 0.25, .1, .075, .05]

  # test settings
  # adaption = "cp"
  # voxel_size = [.5]
  # frames = [1, 2]

  IoU = np.zeros((len(frames), len(voxel_size)))
  Acc = np.zeros((len(frames), len(voxel_size)))
  MSE = np.zeros((len(frames), len(voxel_size)))
  fname = "_" + adaption + "_f" + str(frames[0]) + "-" + str(frames[-1]) \
      + "_v" + str(voxel_size[0]) + "-" + str(voxel_size[-1])

  # run approach
  change_config(cfg_file, "adaption", adaption)
  for f, ff in enumerate(frames):
    change_config(cfg_file, "number_of_scans", ff)
    for v, vv in enumerate(voxel_size):
      print("Run %s with %d frames, voxel size of %f" % (adaption, ff, vv))
      change_config(cfg_file, "voxel_size", vv)
      out = subprocess.check_output([sys.executable,
                                     "lidar_deform.py",
                                     "-c", cfg_file,
                                     "-d", dataset,
                                     "-s", sequence,
                                     "-o", str(offset),
                                     "--one_scan",
                                     "-b"])
      out_decoded = out.decode().strip()
      # print(out_decoded)
      IoU[f, v] = float(out.splitlines()[-4 - 3].decode().strip()[4:])
      Acc[f, v] = float(out.splitlines()[-3 - 3].decode().strip()[4:])
      MSE[f, v] = float(out.splitlines()[-2 - 3].decode().strip()[4:])
      print("-> IoU %f, Acc %f, MSE %f" % (IoU[f, v], Acc[f, v], MSE[f, v]))
      print("->", out.splitlines()[-1 - 3].decode().strip())

  # IoU plot
  plot(1, IoU, "Grid Search IoU Score", "Score_IoU", fname, voxel_size, frames,
       plt.cm.summer)

  # Acc plot
  plot(2, Acc, "Grid Search Acc Score", "Score_Acc", fname, voxel_size, frames,
       plt.cm.summer)

  # MSE plot
  plot(3, MSE, "Grid Search MSE Score", "Score_MSE", fname, voxel_size, frames,
       plt.cm.summer_r)

  plt.show()
