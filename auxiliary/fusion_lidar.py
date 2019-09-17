#!/usr/bin/env python

# Copyright (c) 2018 Andy Zeng
# modified by Ferdinand Langer

import numpy as np
from skimage import measure
import auxiliary.raytracing as rt
import auxiliary.raytracer.RayTracerCython as rtc
try:
  import pycuda.driver as cuda
  import pycuda.autoinit
  from pycuda.compiler import SourceModule
  FUSION_GPU_MODE = 1
except Exception as err:
  print('Warning: %s' % (str(err)))
  print('Failed to import PyCUDA. Running fusion in CPU mode.')
  FUSION_GPU_MODE = 0


class TSDFVolume(object):

  def __init__(self, vol_bnds, voxel_size, fov_up, fov_down):
    # Define projection parameters
    self.fov_up = fov_up
    self.fov_down = fov_down

    # Define voxel volume parameters
    self._vol_bnds = vol_bnds  # rows: x,y,z columns: min,max in world coordinates in meters
    self._voxel_size = voxel_size  # in meters (determines volume discretization and resolution)
    self._trunc_margin = self._voxel_size * 5  # truncation on SDF

    # Adjust volume bounds
    self._vol_dim = np.ceil((self._vol_bnds[:, 1] - self._vol_bnds[:, 0]) / self._voxel_size).copy(order='C').astype(int)  # ensure C-order contigous
    self._vol_bnds[:, 1] = self._vol_bnds[:, 0] + self._vol_dim * self._voxel_size
    self._vol_origin = self._vol_bnds[:, 0].copy(order='C').astype(np.float32)  # ensure C-order contigous
    print("Voxel volume size: %d x %d x %d" % (self._vol_dim[0], self._vol_dim[1], self._vol_dim[2]))
    print("Voxel volume [m]: %d x %d x %d" % (self._vol_dim[0] * voxel_size,
                                              self._vol_dim[1] * voxel_size,
                                              self._vol_dim[2] * voxel_size))
    print("Voxel count: %d mio" % (self._vol_dim[0] * self._vol_dim[1] * self._vol_dim[2] / 1E6))
    print("Voxel size: %f" % (self._voxel_size))

    # TODO Use larger voxel volume / higher voxel resolution by spliting and passing half/quarter of the voxel to the gpu memory

    # Initialize pointers to voxel volume in CPU memory
    self._tsdf_vol_cpu = np.ones(self._vol_dim).astype(np.float32)
    # for computing the cumulative moving average of observations per voxel
    self._weight_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)
    self._color_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)
    self._rem_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)

    # Copy voxel volumes to GPU
    if FUSION_GPU_MODE:
      self._tsdf_vol_gpu = cuda.mem_alloc(self._tsdf_vol_cpu.nbytes)
      cuda.memcpy_htod(self._tsdf_vol_gpu, self._tsdf_vol_cpu)
      self._weight_vol_gpu = cuda.mem_alloc(self._weight_vol_cpu.nbytes)
      cuda.memcpy_htod(self._weight_vol_gpu, self._weight_vol_cpu)
      self._color_vol_gpu = cuda.mem_alloc(self._color_vol_cpu.nbytes)
      cuda.memcpy_htod(self._color_vol_gpu, self._color_vol_cpu)
      self._rem_vol_gpu = cuda.mem_alloc(self._rem_vol_cpu.nbytes)
      cuda.memcpy_htod(self._rem_vol_gpu, self._rem_vol_cpu)

      # Cuda kernel function (C++)
      self._cuda_src_mod = SourceModule("""
        //typedef double Real;
        #define PI  3.14159265358979323846

        __global__ void integrate(float * tsdf_vol,
                                  float * weight_vol,
                                  float * color_vol,
                                  float * rem_vol,
                                  float * vol_dim,
                                  float * vol_origin,
                                  float * cam_pose,
                                  float * other_params,
                                  float * color_im,
                                  float * depth_im,
                                  float * rem_im) {

          // Get voxel index
          int gpu_loop_idx = (int) other_params[0];
          int max_threads_per_block = blockDim.x;
          int block_idx = blockIdx.z*gridDim.y*gridDim.x+blockIdx.y*gridDim.x+blockIdx.x;
          int voxel_idx = gpu_loop_idx*gridDim.x*gridDim.y*gridDim.z*max_threads_per_block+block_idx*max_threads_per_block+threadIdx.x;

          int vol_dim_x = (int) vol_dim[0];
          int vol_dim_y = (int) vol_dim[1];
          int vol_dim_z = (int) vol_dim[2];

          if (voxel_idx > vol_dim_x*vol_dim_y*vol_dim_z)
              return;

          // Get voxel grid coordinates (note: be careful when casting)
          float voxel_x = floorf(((float)voxel_idx)/((float)(vol_dim_y*vol_dim_z)));
          float voxel_y = floorf(((float)(voxel_idx-((int)voxel_x)*vol_dim_y*vol_dim_z))/((float)vol_dim_z));
          float voxel_z = (float)(voxel_idx-((int)voxel_x)*vol_dim_y*vol_dim_z-((int)voxel_y)*vol_dim_z);

          // Voxel grid coordinates to world coordinates
          float voxel_size = other_params[1];
          float pt_x = vol_origin[0]+voxel_x*voxel_size;
          float pt_y = vol_origin[1]+voxel_y*voxel_size;
          float pt_z = vol_origin[2]+voxel_z*voxel_size;

          // World coordinates to camera coordinates
          //float tmp_pt_x = pt_x-cam_pose[0*4+3];
          //float tmp_pt_y = pt_y-cam_pose[1*4+3];
          //float tmp_pt_z = pt_z-cam_pose[2*4+3];
          //float cam_pt_x = cam_pose[0*4+0]*tmp_pt_x+cam_pose[1*4+0]*tmp_pt_y+cam_pose[2*4+0]*tmp_pt_z;
          //float cam_pt_y = cam_pose[0*4+1]*tmp_pt_x+cam_pose[1*4+1]*tmp_pt_y+cam_pose[2*4+1]*tmp_pt_z;
          //float cam_pt_z = cam_pose[0*4+2]*tmp_pt_x+cam_pose[1*4+2]*tmp_pt_y+cam_pose[2*4+2]*tmp_pt_z;

          float cam_pt_x = pt_x;
          float cam_pt_z = pt_z;
          float cam_pt_y = pt_y;

          // spherical projection
          int im_h = (int) other_params[2];
          int im_w = (int) other_params[3];
          float fov_up = other_params[6] * PI / 180.0;
          float fov_down = other_params[7] * PI / 180.0;
          float fov = fabs(fov_up) + fabs(fov_down);
          float depth = norm3df(cam_pt_x, cam_pt_y, cam_pt_z);

          float yaw = -atan2(cam_pt_y, cam_pt_x);
          float pitch = asinf(cam_pt_z / depth);

          // Skip if outside of vertical fov
          if (pitch > fov_up || pitch < fov_down)
              return;

          float proj_x = 0.5 * (yaw / PI + 1.0);                // in [0.0, 1.0]
          float proj_y = 1.0 - (pitch + fabs(fov_down)) / fov;  // in [0.0, 1.0]
          proj_x *= im_w;                                       // in [0.0, W]
          proj_y *= im_h;                                       // in [0.0, H]

          int proj_x_cl = floor(proj_x);
          proj_x_cl = min(im_w - 1, proj_x_cl);
          proj_x_cl = max(0, proj_x_cl);                        // in [0,W-1]
          int proj_y_cl = floor(proj_y);
          proj_y_cl = min(im_h - 1, proj_y_cl);
          proj_y_cl = max(0, proj_y_cl);                        // in [0,H-1]

          int pixel_x = (int) roundf(proj_x_cl);
          int pixel_y = (int) roundf(proj_y_cl);

          // Skip if outside view frustum
          if (pixel_x < 0 || pixel_x >= im_w || pixel_y < 0 || pixel_y >= im_h)
              return;

          // Skip invalid depth
          float depth_value = depth_im[pixel_y*im_w+pixel_x];
          if (depth_value == 0)
              return;

          // Integrate TSDF
          float trunc_margin = other_params[4];
          float depth_diff = depth_value-depth;
          if (depth_diff < -trunc_margin)
              return;
          float dist = fmin(1.0f,depth_diff/trunc_margin);
          float w_old = weight_vol[voxel_idx];
          float obs_weight = other_params[5];
          float w_new = w_old + obs_weight;
          weight_vol[voxel_idx] = w_new;
          tsdf_vol[voxel_idx] = (tsdf_vol[voxel_idx]*w_old+dist)/w_new;

          // Integrate color
          float old_color = color_vol[voxel_idx];
          float old_b = floorf(old_color/(256*256));
          float old_g = floorf((old_color-old_b*256*256)/256);
          float old_r = old_color-old_b*256*256-old_g*256;
          float new_color = color_im[pixel_y*im_w+pixel_x];
          float new_b = floorf(new_color/(256*256));
          float new_g = floorf((new_color-new_b*256*256)/256);
          float new_r = new_color-new_b*256*256-new_g*256;

          // Color interpolation
          new_b = fmin(roundf((old_b*w_old+new_b)/w_new),255.0f);
          new_g = fmin(roundf((old_g*w_old+new_g)/w_new),255.0f);
          new_r = fmin(roundf((old_r*w_old+new_r)/w_new),255.0f);
          color_vol[voxel_idx] = new_b*256*256+new_g*256+new_r;

          // Integrate remissions
          float old_rem = rem_vol[voxel_idx];
          float new_rem = rem_im[pixel_y*im_w+pixel_x];
          rem_vol[voxel_idx] = (old_rem*w_old+new_rem)/w_new;
        }""")

      self._cuda_integrate = self._cuda_src_mod.get_function("integrate")

      # Determine block/grid size on GPU
      gpu_dev = cuda.Device(0)
      self._max_gpu_threads_per_block = gpu_dev.MAX_THREADS_PER_BLOCK
      n_blocks = int(np.ceil(float(np.prod(self._vol_dim)) /
                     float(self._max_gpu_threads_per_block)))
      grid_dim_x = min(gpu_dev.MAX_GRID_DIM_X,
                       int(np.floor(np.cbrt(n_blocks))))
      grid_dim_y = min(gpu_dev.MAX_GRID_DIM_Y,
                       int(np.floor(np.sqrt(n_blocks / grid_dim_x))))
      grid_dim_z = min(gpu_dev.MAX_GRID_DIM_Z,
                       int(np.ceil(float(n_blocks) /
                           float(grid_dim_x * grid_dim_y))))
      self._max_gpu_grid_dim = np.array([grid_dim_x,
                                         grid_dim_y,
                                         grid_dim_z]).astype(int)
      self._n_gpu_loops = int(np.ceil(float(np.prod(self._vol_dim)) /
                              float(np.prod(self._max_gpu_grid_dim) *
                                    self._max_gpu_threads_per_block)))

  def integrate(self, color_im, depth_im, rem_im, cam_pose, obs_weight=1.):
    """ Data should be in world frame with pose transformation applied
        Not using the cam_pose input!
    """
    im_h = depth_im.shape[0]
    im_w = depth_im.shape[1]

    # Fold RGB color image into a single channel image
    color_im = color_im.astype(np.float32)
    # color_im = np.floor(color_im[:,:,2]*256*256+color_im[:,:,1]*256+color_im[:,:,0])
    color_im = np.floor(color_im[:, :, 0] * 256 * 256 +
                        color_im[:, :, 1] * 256 + color_im[:, :, 2])

    # GPU mode: integrate voxel volume (calls CUDA kernel)
    if FUSION_GPU_MODE:
      for gpu_loop_idx in range(self._n_gpu_loops):
        self._cuda_integrate(
            self._tsdf_vol_gpu,
            self._weight_vol_gpu,
            self._color_vol_gpu,
            self._rem_vol_gpu,
            cuda.InOut(self._vol_dim.astype(np.float32)),
            cuda.InOut(self._vol_origin.astype(np.float32)),
            cuda.InOut(cam_pose.reshape(-1).astype(np.float32)),
            cuda.InOut(np.asarray(
                [gpu_loop_idx, self._voxel_size, im_h, im_w,
                 self._trunc_margin, obs_weight, self.fov_up, self.fov_down],
                np.float32)),
            cuda.InOut(color_im.reshape(-1).astype(np.float32)),
            cuda.InOut(depth_im.reshape(-1).astype(np.float32)),
            cuda.InOut(rem_im.reshape(-1).astype(np.float32)),
            block=(self._max_gpu_threads_per_block, 1, 1),
            grid=(int(self._max_gpu_grid_dim[0]),
                  int(self._max_gpu_grid_dim[1]),
                  int(self._max_gpu_grid_dim[2])))

    # CPU mode: integrate voxel volume (vectorized implementation)
    else:
      # Get voxel grid coordinates
      xv, yv, zv = np.meshgrid(range(self._vol_dim[0]),
                               range(self._vol_dim[1]),
                               range(self._vol_dim[2]), indexing='ij')
      vox_coords = np.concatenate((xv.reshape(1, -1), yv.reshape(1, -1),
                                   zv.reshape(1, -1)), axis=0).astype(int)

      # Voxel coordinates to world coordinates
      world_pts = self._vol_origin.reshape(-1, 1) \
          + vox_coords.astype(float) * self._voxel_size

      # World coordinates to camera coordinates
      world2cam = np.linalg.inv(cam_pose)
      cam_pts = np.dot(world2cam[:3, :3], world_pts) \
          + np.tile(world2cam[:3, 3].reshape(3, 1), (1, world_pts.shape[1]))

      # Sphere camera coordinates to image pixel
      fov_up = self.fov_up / 180.0 * np.pi      # field of view up in radians
      fov_down = self.fov_down / 180.0 * np.pi  # field of view down in radians
      fov = abs(fov_down) + abs(fov_up)    # get field of view total in radians
      proj_W = im_w
      proj_H = im_h
      scan_x = cam_pts[0, :]
      scan_y = cam_pts[1, :]
      scan_z = cam_pts[2, :]
      depth = np.linalg.norm(cam_pts, 2, axis=0)

      # print("scan_xyz", scan_x.shape, scan_y.shape, scan_z.shape)
      yaw = -np.arctan2(scan_y, scan_x)
      pitch = np.arcsin(scan_z / depth)
      proj_x = 0.5 * (yaw / np.pi + 1.0)            # in [0.0, 1.0]
      proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]
      proj_x *= proj_W                              # in [0.0, W]
      proj_y *= proj_H                              # in [0.0, H]

      proj_x_cl = np.floor(proj_x)
      proj_x_cl = np.minimum(proj_W - 1, proj_x_cl)
      proj_x_cl = np.maximum(0, proj_x_cl).astype(np.int32)   # in [0,W-1]
      proj_y_cl = np.floor(proj_y)
      proj_y_cl = np.minimum(proj_H - 1, proj_y_cl)
      proj_y_cl = np.maximum(0, proj_y_cl).astype(np.int32)   # in [0,H-1]

      pix_x = proj_x_cl
      pix_y = proj_y_cl

      # Skip if outside view frustum or outside of vertical FOV
      valid_pix = np.logical_and(pix_x >= 0,
                  np.logical_and(pix_x < im_w,
                  np.logical_and(pix_y >= 0,
                  np.logical_and(pix_y < im_h,
                  np.logical_and(pitch < fov_up,
                                 pitch > fov_down)))))

      depth_val = np.zeros(pix_x.shape)
      depth_val[valid_pix] = depth_im[pix_y[valid_pix],pix_x[valid_pix]]

      # Integrate TSDF
      depth_diff = depth_val - depth
      # depth_diff = depth_val-cam_pts[2,:]
      valid_pts = np.logical_and(depth_val > 0,
                                 depth_diff >= -self._trunc_margin)
      dist = np.minimum(1., np.divide(depth_diff, self._trunc_margin))
      w_old = self._weight_vol_cpu[vox_coords[0, valid_pts],
                                   vox_coords[1, valid_pts],
                                   vox_coords[2, valid_pts]]
      w_new = w_old + obs_weight
      self._weight_vol_cpu[vox_coords[0, valid_pts],
                           vox_coords[1, valid_pts],
                           vox_coords[2, valid_pts]] = w_new
      tsdf_vals = self._tsdf_vol_cpu[vox_coords[0, valid_pts],
                                     vox_coords[1, valid_pts],
                                     vox_coords[2, valid_pts]]
      self._tsdf_vol_cpu[vox_coords[0, valid_pts],
                         vox_coords[1, valid_pts],
                         vox_coords[2, valid_pts]] = np.divide(np.multiply(tsdf_vals, w_old) + dist[valid_pts], w_new)

      # Integrate color
      old_color = self._color_vol_cpu[vox_coords[0, valid_pts],
                                      vox_coords[1, valid_pts],
                                      vox_coords[2, valid_pts]]
      old_b = np.floor(old_color / (256. * 256.))
      old_g = np.floor((old_color - old_b * 256. * 256.) / 256.)
      old_r = old_color - old_b * 256. * 256. - old_g * 256.
      new_color = color_im[pix_y[valid_pts], pix_x[valid_pts]]
      new_b = np.floor(new_color / (256. * 256.))
      new_g = np.floor((new_color - new_b * 256. * 256.) / 256.)
      new_r = new_color - new_b * 256. * 256. - new_g * 256.
      new_b = np.minimum(
          np.round(np.divide(np.multiply(old_b, w_old) + new_b, w_new)), 255.)
      new_g = np.minimum(
          np.round(np.divide(np.multiply(old_g, w_old) + new_g, w_new)), 255.)
      new_r = np.minimum(
          np.round(np.divide(np.multiply(old_r, w_old) + new_r, w_new)), 255.)
      self._color_vol_cpu[vox_coords[0, valid_pts],
                          vox_coords[1, valid_pts],
                          vox_coords[2, valid_pts]] = new_b * 256. * 256. + new_g * 256. + new_r

      # TODO Integrate remissions for CPU
      # new_rem = np.minimum(np.round(np.divide(np.multiply(old_rem, w_old) + new_rem, w_new)), 255.)
      # self._rem_vol_cpu[vox_coords[0,valid_pts],vox_coords[1,valid_pts],vox_coords[2,valid_pts]] = new_rem

  # Copy voxel volume to CPU
  def get_volume(self):
    if FUSION_GPU_MODE:
      cuda.memcpy_dtoh(self._tsdf_vol_cpu, self._tsdf_vol_gpu)
      cuda.memcpy_dtoh(self._color_vol_cpu, self._color_vol_gpu)
      cuda.memcpy_dtoh(self._rem_vol_cpu, self._rem_vol_gpu)
    return self._tsdf_vol_cpu, self._color_vol_cpu, self._rem_vol_cpu

  # Get mesh of voxel volume via marching cubes
  def get_mesh(self, color_lut):
    tsdf_vol, color_vol, rem_vol = self.get_volume()

    # Marching cubes
    verts, faces, norms, vals = measure.marching_cubes_lewiner(tsdf_vol, level=0)
    verts_ind = np.round(verts).astype(int)

    # voxel grid coordinates to world coordinates
    verts = verts * self._voxel_size + self._vol_origin

    # Get vertex colors
    rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]

    # Get vertex remissioins
    rem = rem_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
    colors_b = np.floor(rgb_vals / (256 * 256))
    colors_g = np.floor((rgb_vals - colors_b * 256 * 256) / 256)
    colors_r = rgb_vals - colors_b * 256 * 256 - colors_g * 256
    colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T
    colors = colors.astype(np.uint8)
    return verts, faces, norms, colors, rem

  def throw_rays_at_mesh(self, rays, origin, H, W, color_lut):
    print("Get mesh by marching cubes...")
    verts, faces, norms, colors, rem = self.get_mesh(color_lut)

    # TODO Expose parameter to write mesh
    # meshwrite("test.ply",verts,faces,norms,colors)

    # Arrays must be contiguous and 1D
    verts_c = np.ascontiguousarray(verts.reshape(-1))
    faces_c = np.ascontiguousarray(faces.reshape(-1))
    colors_c = np.ascontiguousarray(colors.reshape(-1).astype(np.int32))
    rem_c = np.ascontiguousarray(rem.reshape(-1))
    rays = rays.reshape(-1)

    ray_endpoints = np.ascontiguousarray(np.zeros((H, W, 3)).reshape(-1)
                                         .astype(np.float32))
    ray_colors = np.ascontiguousarray(np.zeros((H, W, 3)).reshape(-1)
                                      .astype(np.int32))
    range_image = np.ascontiguousarray(np.zeros((H, W)).reshape(-1)
                                       .astype(np.float32))
    rem_image = np.ascontiguousarray(np.zeros((H, W)).reshape(-1)
                                     .astype(np.float32))

    print("Raytracing...")
    rtc.C_Trace(rays, origin, verts_c, faces_c, colors_c, rem_c, ray_endpoints,
                ray_colors, range_image, rem_image, H, W)

    return ray_endpoints.reshape(-1, 3), ray_colors.reshape(-1, 3), \
        verts, colors, faces, range_image.reshape(-1, W), rem_image.reshape(-1, W)


# ------------------------------------------------------------------------------
# Additional helper functions

# Save 3D mesh to a polygon .ply file
def meshwrite(filename, verts, faces, norms, colors):

  # Write header
  ply_file = open(filename, 'w')
  ply_file.write("ply\n")
  ply_file.write("format ascii 1.0\n")
  ply_file.write("element vertex %d\n"%(verts.shape[0]))
  ply_file.write("property float x\n")
  ply_file.write("property float y\n")
  ply_file.write("property float z\n")
  ply_file.write("property float nx\n")
  ply_file.write("property float ny\n")
  ply_file.write("property float nz\n")
  ply_file.write("property uchar red\n")
  ply_file.write("property uchar green\n")
  ply_file.write("property uchar blue\n")
  ply_file.write("element face %d\n"%(faces.shape[0]))
  ply_file.write("property list uchar int vertex_index\n")
  ply_file.write("end_header\n")

  # Write vertex list
  for i in range(verts.shape[0]):
    ply_file.write("%f %f %f %f %f %f %d %d %d\n" % (verts[i, 0], verts[i, 1],
                                                     verts[i, 2], norms[i, 0],
                                                     norms[i, 1], norms[i, 2],
                                                     colors[i, 0],
                                                     colors[i, 1],
                                                     colors[i, 2]))

  # Write face list
  for i in range(faces.shape[0]):
    ply_file.write("3 %d %d %d\n" % (faces[i, 0], faces[i, 1], faces[i, 2]))

  ply_file.close()
