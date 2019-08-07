#!/usr/bin/env python3

import vispy
from vispy.scene import visuals, SceneCanvas
import numpy as np
from matplotlib import pyplot as plt
from auxiliary.np_ioueval import iouEval


class LaserScanVis():
  """Class that creates and handles a visualizer for a pointcloud"""

  def __init__(self, W, H, mesh=False):
    self.W = W
    self.H = H
    self.mesh = mesh
    self.frame = 0
    self.nframes = 0
    self.reset()

  def reset(self):
    """ Reset. """
    # last key press (it should have a mutex, but visualization is not
    # safety critical, so let's do things wrong)
    self.action = "no"  # no, next, back, quit are the possibilities

    # NEW canvas prepared for visualizing laserscan data
    self.scan_canvas = SceneCanvas(
        keys='interactive', show=True, title='', size=(1600, 600))
    self.scan_canvas.events.key_press.connect(self.key_press)
    self.grid_view = self.scan_canvas.central_widget.add_grid()
    
    # source laserscan
    self.scan_view = vispy.scene.widgets.ViewBox(
      border_color='white', parent=self.scan_canvas.scene)
    self.scan_vis = visuals.Markers()
    self.scan_view.camera = 'turntable'
    self.scan_view.add(self.scan_vis)
    visuals.XYZAxis(parent=self.scan_view.scene)
    self.grid_view.add_widget(self.scan_view, 0, 0)

    # target laserscan
    self.back_view = vispy.scene.widgets.ViewBox(
      border_color='white', parent=self.scan_canvas.scene)
    self.back_vis = visuals.Markers()
    self.back_view.camera = 'turntable'
    self.back_view.camera.link(self.scan_view.camera)
    self.back_view.add(self.back_vis)
    visuals.XYZAxis(parent=self.back_view.scene)
    self.grid_view.add_widget(self.back_view, 0, 1)

    # self.grid_view.padding = 6

    # NEW canvas for range img data
    self.img_canvas = SceneCanvas(keys='interactive', show=True,
                                  title='Original Range Image',
                                  size=(self.W[0], self.H[0]))
    self.img_canvas.events.key_press.connect(self.key_press)
    self.img_view = self.img_canvas.central_widget.add_view()
    self.img_vis = visuals.Image(cmap='viridis')
    self.img_view.add(self.img_vis)

    # NEW test canvas
    self.test_canvas = SceneCanvas(keys='interactive', show=True,
                                   title='Test Range Image',
                                   size=(self.W[1], self.H[1]))
    self.test_canvas.events.key_press.connect(self.key_press)
    self.test_view = self.test_canvas.central_widget.add_view()
    self.test_vis = visuals.Image(cmap='viridis')
    self.test_view.add(self.test_vis)

    # NEW canvas for showing difference in range and labels
    self.diff_canvas = SceneCanvas(keys='interactive', show=True,
                                   title='Difference Range Image',
                                   size=(self.W[1], self.H[1]*2))
    self.diff_canvas.events.key_press.connect(self.key_press)
    self.diff_view = self.diff_canvas.central_widget.add_grid()

    self.diff_view_depth = vispy.scene.widgets.ViewBox(
      border_color='white', parent=self.diff_canvas.scene)
    # self.diff_image_depth = visuals.Image(cmap='viridis')
    self.diff_image_depth = visuals.Image()
    self.diff_view_depth.add(self.diff_image_depth)
    self.diff_view.add_widget(self.diff_view_depth, 0, 0)

    self.diff_view_label = vispy.scene.widgets.ViewBox(
      border_color='white', parent=self.diff_canvas.scene)
    self.diff_image_label = visuals.Image(cmap='viridis')
    self.diff_view_label.add(self.diff_image_label)
    self.diff_view.add_widget(self.diff_view_label, 1, 0)

  def get_mpl_colormap(self, cmap_name):
    cmap = plt.get_cmap(cmap_name)

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

    return color_range.reshape(256, 3).astype(np.float32) / 255.0

  def show_mesh(self, show_mesh_):
    self.mesh = show_mesh_
    self.mesh_view = vispy.scene.widgets.ViewBox(
      border_color='white', parent=self.scan_canvas.scene)
    self.mesh_vis = visuals.Mesh(shading=None)
    self.mesh_view.camera = 'turntable'
    self.mesh_view.camera.link(self.scan_view.camera)
    self.mesh_view.add(self.mesh_vis)
    visuals.XYZAxis(parent=self.mesh_view.scene)
    self.grid_view.add_widget(self.mesh_view, 0, 2)

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

  def set_laserscans(self, scan):
    self.scan_canvas.title = 'Frame %d of %d' % (self.frame + 1, self.nframes)
    # plot range
    if hasattr(scan.get_scan(0), 'label_color'):
      label_color = scan.merged.label_color_image.reshape(-1,3)
      points = scan.merged.back_points.reshape(-1,3)
      self.back_vis.set_data(points,
                             face_color=label_color[..., ::-1],
                             edge_color=label_color[..., ::-1],
                             size=3)
    else:
      power = 16
      range_data = np.copy(scan.get_scan(0).unproj_range)
      # print(range_data.max(), range_data.min())
      range_data = range_data**(1 / power)
      # print(range_data.max(), range_data.min())
      viridis_range = ((range_data - range_data.min()) /
                       (range_data.max() - range_data.min()) *
                       255).astype(np.uint8)
      viridis_map = self.get_mpl_colormap("viridis")
      viridis_colors = viridis_map[viridis_range]
      self.back_vis.set_data(scan.get_scan(0).points,
                             face_color=viridis_colors[..., ::-1],
                             edge_color=viridis_colors[..., ::-1],
                             size=3)
    self.back_vis.update()

    # plot range image test
    if hasattr(scan.get_scan(0), 'proj_color'):
      self.test_vis.set_data(scan.get_scan(0).proj_color[..., ::-1])
    else:
      # print()
      data = np.copy(scan.get_scan(0).proj_range)
      # print(data[data > 0].max(), data[data > 0].min())
      data[data > 0] = data[data > 0]**(1 / power)
      data[data < 0] = data[data > 0].min()
      # print(data.max(), data.min())
      data = (data - data[data > 0].min()) / \
          (data.max() - data[data > 0].min())
      # print(data.max(), data.min())
      self.test_vis.set_data(data)
    self.test_vis.update()

  def set_diff(self, scan_source, scan_target):
    # Label intersection image
    source_label = scan_source.proj_color[..., ::-1]
    source_label_map = scan_source.get_label_map()

    if scan_target.adaption == 'cp':
      target_label_map = scan_target.merged.get_label_map()
      target_label = scan_target.merged.proj_color[..., ::-1]
    else:
      target_label = scan_target.ray_colors.reshape(64, -1, 3)/255
      target_label_map = scan_target.get_label_map()

    # Mask out no data (= black) in target scan
    black = np.sum(source_label, axis=2) == 0
    black = np.repeat(black[:, :, np.newaxis], 3, axis=2)
    target_label[black] = 0
    black = source_label_map == 0
    target_label_map[black] = 0

    # Ignore empty classes
    unique_values = np.unique(source_label_map)
    empty = np.isin(np.arange(scan_source.nclasses), unique_values) is False
    
    # Evaluate by label
    eval = iouEval(scan_source.nclasses, np.arange(scan_source.nclasses)[empty])
    eval.addBatch(target_label_map, source_label_map)
    m_iou, iou = eval.getIoU()
    print("IoU: ", m_iou)
    print("IoU class: ", (iou * 100).astype(int))
    m_acc = eval.getacc()
    print("Acc: ", m_acc)

    label_diff = abs(source_label - target_label)
    self.diff_image_label.set_data(label_diff)
    self.diff_image_label.update()

    # Range diff image
    source_range = scan_source.proj_range
    if scan_target.adaption == 'cp':
      target_range = scan_target.merged.proj_range
    else:
      target_range = scan_target.range_image
    # Mask out no data (= black) in target scan
    black = source_range == 0
    # target_range[black] = 0

    # Mask out too far data in target scan
    too_far = source_range >= 50
    source_range[too_far] = 0
    target_range[too_far] = 0

    print(np.amax(source_range))

    range_diff = (source_range - target_range) ** 2

    self.diff_image_depth.set_data(range_diff)
    self.diff_image_depth.update()

    MSE = range_diff.sum() / range_diff.size

    self.diff_canvas.title = \
        'IoU %5.2f%%, Acc %5.2f%%, MSE %f' % (m_iou * 100.0, m_acc * 100, MSE)

  def set_mesh(self, verts, verts_colors, faces):
    if self.mesh:
      self.mesh_vis.set_data(vertices=verts,
                             vertex_colors=verts_colors,
                             faces=faces)
      self.mesh_vis.update()

  def set_points(self, points, colors, W, H):
    # plot range
    colors = colors / 255
    self.back_vis.set_data(points,
                            face_color=colors,
                            edge_color=colors,
                            size=3)
    self.back_vis.update()

    # plot range image test
    self.test_vis.set_data(colors.reshape(H,W,3))
    self.test_vis.update()

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
