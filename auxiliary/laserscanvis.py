#!/usr/bin/env python3

import vispy
from vispy.scene import visuals, SceneCanvas
import numpy as np
from matplotlib import pyplot as plt


class LaserScanVis():
  """Class that creates and handles a visualizer for a pointcloud"""

  def __init__(self, W, H):
    self.reset(W, H)

  def reset(self, W, H):
    """ Reset. """
    # last key press (it should have a mutex, but visualization is not
    # safety critical, so let's do things wrong)
    self.action = "no"  # no, next, back, quit are the possibilities

    # new canvas prepared for visualizing data
    # self.scan_canvas = SceneCanvas(keys='interactive', show=True, title='Original Point Cloud')
    self.scan_canvas = SceneCanvas(keys='interactive', show=True, title='Original Point Cloud | Generated Point Cloud', size=(1600,600))
    # interface (n next, b back, q quit, very simple)
    self.scan_canvas.events.key_press.connect(self.key_press)
    self.grid_view = self.scan_canvas.central_widget.add_grid()
    # laserscan part
    self.scan_view = vispy.scene.widgets.ViewBox(border_color='white', parent=self.scan_canvas.scene)
    # self.scan_view = self.scan_canvas.central_widget.add_view()
    self.scan_vis = visuals.Markers()
    self.scan_view.camera = 'turntable'
    self.scan_view.add(self.scan_vis)
    visuals.XYZAxis(parent=self.scan_view.scene)

    # new canvas for img
    self.img_canvas = SceneCanvas(
        keys='interactive', show=True, size=(W[0], H[0]), title='Original Range Image')
    # interface (n next, b back, q quit, very simple)
    self.img_canvas.events.key_press.connect(self.key_press)
    # add a view for the depth
    self.img_view = self.img_canvas.central_widget.add_view()
    self.img_vis = visuals.Image(cmap='viridis')
    self.img_view.add(self.img_vis)

    # new test canvas prepared for visualizing backprojected data
    # self.back_canvas = SceneCanvas(keys='interactive', show=True, title='Backprojected Point Cloud')
    # interface (n next, b back, q quit, very simple)
    # self.back_canvas.events.key_press.connect(self.key_press)
    # laserscan part
    self.back_view = vispy.scene.widgets.ViewBox(border_color='white', parent=self.scan_canvas.scene)
    # self.back_view = self.back_canvas.central_widget.add_view()
    self.back_vis = visuals.Markers()
    self.back_view.camera = 'turntable'
    self.back_view.camera.link(self.scan_view.camera)
    self.back_view.add(self.back_vis)

    # l = 20
    # b = 64
    # fov_up = 3
    # fov_down = -25
    # beam_angles = np.sin(np.linspace(fov_down, fov_up, b, dtype=np.float32)/180*np.pi)
    # pos = [0, 0, 0]
    # # self.beams_vis = []
    # for i in beam_angles:
    #   visuals.Line(pos= np.array([pos, [0, l, i*l]]), color=[1, 0, 0], parent=self.back_view.scene)
      # self.beams_vis.append(line)
      # self.bac_vis.add(line)

    # self.back_view.add(self.beams_vis)
    visuals.XYZAxis(parent=self.back_view.scene)

    # new test canvas
    self.test_canvas = SceneCanvas(keys='interactive', show=True, size=(W[1], H[1]), title='Test Range Image')
    # interface (n next, b back, q quit, very simple)
    self.test_canvas.events.key_press.connect(self.key_press)
    # add a view for the depth
    self.test_view = self.test_canvas.central_widget.add_view()
    self.test_vis = visuals.Image(cmap='viridis')
    self.test_view.add(self.test_vis)

    # self.grid_view.padding = 6
    self.grid_view.add_widget(self.scan_view, 0, 0)
    self.grid_view.add_widget(self.back_view, 0, 1)

  def get_mpl_colormap(self, cmap_name):
    cmap = plt.get_cmap(cmap_name)

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

    return color_range.reshape(256, 3).astype(np.float32) / 255.0

  def set_laserscan(self, scan):
    # plot range
    if hasattr(scan, 'label_color'):
      # print(scan.label_color.shape)
      self.scan_vis.set_data(scan.points,
                             face_color=scan.label_color[..., ::-1],
                             edge_color=scan.label_color[..., ::-1],
                             size=3)
    else:
      power = 16
      # print()
      range_data = np.copy(scan.unproj_range)
      # print(range_data.max(), range_data.min())
      range_data = range_data**(1 / power)
      # print(range_data.max(), range_data.min())
      viridis_range = ((range_data - range_data.min()) /
                       (range_data.max() - range_data.min()) *
                       255).astype(np.uint8)
      viridis_map = self.get_mpl_colormap("viridis")
      viridis_colors = viridis_map[viridis_range]
      self.scan_vis.set_data(scan.points,
                             face_color=viridis_colors[..., ::-1],
                             edge_color=viridis_colors[..., ::-1],
                             size=3)
    self.scan_vis.update()

    # plot range image
    if hasattr(scan, 'proj_color'):
      self.img_vis.set_data(scan.proj_color[..., ::-1])
    else:
      # print()
      data = np.copy(scan.proj_range)
      # print(data[data > 0].max(), data[data > 0].min())
      data[data > 0] = data[data > 0]**(1 / power)
      data[data < 0] = data[data > 0].min()
      # print(data.max(), data.min())
      data = (data - data[data > 0].min()) / \
          (data.max() - data[data > 0].min())
      # print(data.max(), data.min())
      self.img_vis.set_data(data)
    self.img_vis.update()

  def set_laserscan2(self, scan, pose):
    # plot range
    if hasattr(scan, 'label_color'):
      # print(scan.label_color.shape)
      # back_points = hom_points[:, 0:3]
      # back_points[:,0] -=4
      label_color = scan.label_color_image.reshape(-1,3)
      points = scan.back_points.reshape(-1,3)
      self.back_vis.set_data(points,
                             face_color=label_color[..., ::-1],
                             edge_color=label_color[..., ::-1],
                             size=3)
    else:
      power = 16
      # print()
      range_data = np.copy(scan.unproj_range)
      # print(range_data.max(), range_data.min())
      range_data = range_data**(1 / power)
      # print(range_data.max(), range_data.min())
      viridis_range = ((range_data - range_data.min()) /
                       (range_data.max() - range_data.min()) *
                       255).astype(np.uint8)
      viridis_map = self.get_mpl_colormap("viridis")
      viridis_colors = viridis_map[viridis_range]
      self.back_vis.set_data(scan.points,
                             face_color=viridis_colors[..., ::-1],
                             edge_color=viridis_colors[..., ::-1],
                             size=3)
    self.back_vis.update()

    # plot range image test
    if hasattr(scan, 'proj_color'):
      self.test_vis.set_data(scan.proj_color[..., ::-1])
    else:
      # print()
      data = np.copy(scan.proj_range)
      # print(data[data > 0].max(), data[data > 0].min())
      data[data > 0] = data[data > 0]**(1 / power)
      data[data < 0] = data[data > 0].min()
      # print(data.max(), data.min())
      data = (data - data[data > 0].min()) / \
          (data.max() - data[data > 0].min())
      # print(data.max(), data.min())
      self.test_vis.set_data(data)
    self.test_vis.update()

  def set_points(self, points, colors):
    # plot range
    colors = colors/255
    self.back_vis.set_data(points,
                            face_color=colors,
                            edge_color=colors,
                            size=3)
    self.back_vis.update()

    # plot range image test
    # if hasattr(scan.get_scan(2), 'proj_color'):
    #   self.test_vis.set_data(scan.get_scan(2).proj_color[..., ::-1])
    # else:
    #   # print()
    #   data = np.copy(scan.get_scan(2).proj_range)
    #   # print(data[data > 0].max(), data[data > 0].min())
    #   data[data > 0] = data[data > 0]**(1 / power)
    #   data[data < 0] = data[data > 0].min()
    #   # print(data.max(), data.min())
    #   data = (data - data[data > 0].min()) / \
    #       (data.max() - data[data > 0].min())
    #   # print(data.max(), data.min())
    #   self.test_vis.set_data(data)
    # self.test_vis.update()

  # interface
  def key_press(self, event):
    if event.key == 'N':
      self.action = 'next'
    elif event.key == 'B':
      self.action = 'back'
    elif event.key == 'Q' or event.key == 'Escape':
      self.destroy()
      self.action = 'quit'

  def get_action(self, timeout=0):
    # return action and void it to avoid reentry
    vispy.app.use_app().sleep(timeout)
    ret = self.action
    self.action = 'no'
    return ret

  def destroy(self):
    # destroy the visualization
    self.scan_canvas.events.key_press.disconnect()
    self.scan_canvas.close()
    self.img_canvas.events.key_press.disconnect()
    self.img_canvas.close()
    self.test_canvas.events.key_press.disconnect()
    self.test_canvas.close()
    vispy.app.quit()
