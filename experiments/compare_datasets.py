#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import yaml


def dict2list(dict):
  dlist = []
  for key, value in dict.items():
    # temp = [key, value]
    temp = value
    dlist.append(temp)
  return dlist


if __name__ == '__main__':
  with open("datasets.yaml") as f:
    config = yaml.safe_load(f)

  labels = config["labels"]
  labels_list = dict2list(labels)
  labels = np.array(labels_list)

  kitti = config["kitti"]
  kitti_list = dict2list(kitti)

  nuscenes = config["nuscenes"]
  nuscenes_list = dict2list(nuscenes)

  nuscenes = np.array(nuscenes_list)
  kitti = np.array(kitti_list)

  fig = plt.figure(figsize=(16, 6))
  ax = fig.add_subplot(111)
  ax.set_axisbelow(True)
  plt.grid(linestyle=':')

  width = 2
  spacing = 6
  ind = np.arange(0, len(kitti) * spacing, spacing)
  histo_kitti = ax.bar(ind, kitti, width)
  histo_nuscenes = ax.bar(ind + width, nuscenes, width)
  plt.xticks(ind + width / 2, labels, rotation='vertical')
  fig.subplots_adjust(bottom=0.3)

  plt.xlabel('classes')
  plt.ylabel('percentage')
  # plt.title("title")
  ax.set_xlim([spacing / 2, len(kitti) * spacing - spacing / 2])
  plt.tight_layout()

  plt.savefig("datasets.svg")
  plt.savefig("datasets.pdf")
  ax.set_yscale('log')
  plt.savefig("datasets_log.svg")
  plt.savefig("datasets_log.pdf")
  plt.show()
