#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt


def get_mpl_colormap(cmap_name):
  cmap = plt.get_cmap(cmap_name)

  # Initialize the matplotlib color map
  sm = plt.cm.ScalarMappable(cmap=cmap)

  # Obtain linear color range
  color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

  return color_range.reshape(256, 3).astype(np.float32) / 255.0


def convert_range(range_image, power=16):
  data = np.copy(range_image)
  # print(data[data > 0].max(), data[data > 0].min())
  data[data > 0] = data[data > 0]**(1 / power)
  data[data < 0] = data[data > 0].min()
  # print(data.max(), data.min())
  data = (data - data[data > 0].min()) / \
      (data.max() - data[data > 0].min())
  return data
