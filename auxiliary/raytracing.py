import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

try:
    from pycuda.compiler import SourceModule
    from pycuda import tools
    import pycuda.gpuarray as gpuarray
    import pycuda.driver as cuda
    import pycuda.autoinit
    GPU_MODE = 1
except Exception as err:
    print('Warning: %s'%(str(err)))
    print('Failed to import PyCUDA. Running raytracing in CPU mode.')
    GPU_MODE = 0

def ray_mesh_intersection(rays, origin, vertices, vertices_colors, faces, H, W):
    if GPU_MODE:
        ray_mesh_intersection_CUDA(rays, origin, vertices, vertices_colors, faces, H=H, W=W)
    else:
        ray_mesh_intersection_CPU(rays, origin, vertices, vertices_colors, faces, H, W)

def ray_mesh_intersection_CPU(rays, origin, vertices, vertices_colors, faces, H, W):
    ray_endpoints = np.zeros((H * W, 3))
    ray_image = np.zeros((H * W, 3))
    for i, ray in enumerate(rays):
        for triangle_index in faces:
            triangle = vertices[triangle_index]
            valid, point = ray_triangle_intersection(ray, origin, triangle)

            if valid:
                # save point corresponding to current ray
                ray_endpoints[i,:] = point

                # TODO Assign color to image
                ray_image[i,:] = np.ones((1,3))*255
    return ray_endpoints, ray_image

def ray_mesh_intersection_CUDA(rays, origin, vertices, vertices_colors, faces, H=64, W=1024):
    endpoints = np.zeros(rays.shape).astype(np.float32)
    colors = np.zeros(rays.shape).astype(np.float32)
    
    rays = rays.astype(np.float32)
    rays_gpu = cuda.mem_alloc(rays.nbytes)
    cuda.memcpy_htod(rays_gpu,rays)

    vertices = vertices.astype(np.float32)
    vertices_gpu = cuda.mem_alloc(vertices.nbytes)
    cuda.memcpy_htod(vertices_gpu,vertices)

    vertices_colors = vertices_colors.astype(np.float32)
    vertices_colors_gpu = cuda.mem_alloc(vertices_colors.nbytes)
    cuda.memcpy_htod(vertices_colors_gpu,vertices_colors)

    faces = faces.astype(np.int32)
    faces_gpu = cuda.mem_alloc(faces.nbytes)
    cuda.memcpy_htod(faces_gpu,faces)

    endpoints_gpu = cuda.mem_alloc(endpoints.nbytes)
    cuda.memcpy_htod(endpoints_gpu,endpoints)

    colors_gpu = cuda.mem_alloc(colors.nbytes)
    cuda.memcpy_htod(colors_gpu,colors)

    cuda_src_mod = SourceModule("""
    #include <stdio.h>
    __global__ void intersection(float * rays,
                                 float * origin,
                                 float * verts,
                                 float * verts_colors,
                                 int * faces,
                                 float * param,
                                 float * endpoints,
                                 float * colors) {
        int no_of_faces = param[2];
        int gpu_loop_idx = param[3];
        int x = gpu_loop_idx * blockDim.x + threadIdx.x;
        //int y = threadIdx.y + blockIdx.y * blockDim.y;
        //int linearoffset = x + y + blockDim.x * blockDim.y;
        float eps = 0.000001f;

        float ray[3] = {rays[x*3+0], rays[x*3+1], rays[x*3+2]};
        
        for (int f = 0; f < no_of_faces; f++) {
            float v0[3] = {verts[faces[f*3+0]*3+0], verts[faces[f*3+0]*3+1], verts[faces[f*3+0]*3+2]};
            float v1[3] = {verts[faces[f*3+1]*3+0], verts[faces[f*3+1]*3+1], verts[faces[f*3+1]*3+2]};
            float v2[3] = {verts[faces[f*3+2]*3+0], verts[faces[f*3+2]*3+1], verts[faces[f*3+2]*3+2]};

            // edge e1 between v0 and v1
            float e1[3];
            e1[0] = v1[0] - v0[0];
            e1[1] = v1[1] - v0[1];
            e1[2] = v1[2] - v0[2];

            // edge e2 between v0 and v2
            float e2[3];
            e2[0] = v2[0] - v0[0];
            e2[1] = v2[1] - v0[1];
            e2[2] = v2[2] - v0[2];

            // h = ray X e2
            float h[3];
            h[0] = ray[1] * e2[2] - ray[2] * e2[1];
            h[1] = ray[2] * e2[0] - ray[0] * e2[2];
            h[2] = ray[0] * e2[1] - ray[1] * e2[0];

            float a = e1[0] * h[0] + e1[1] * h[1] + e1[2] * h[2];
            if (a < eps && a > -eps) {
                // Ray is parallel to this face.
                continue;
            }

            float inv_a = 1.0f/ a;
            // vector s between v0 and origin
            float s[3];
            s[0] = origin[0] - v0[0];
            s[1] = origin[1] - v0[1];
            s[2] = origin[2] - v0[2];
            float u = (s[0] * h[0] + s[1] * h[1] + s[2] * h[2]) * inv_a;
            if (u < 0 || u > 1) {
                // Not culling
                continue;
            }

            // q = s X e1
            float q[3];
            q[0] = s[1] * e1[2] - s[2] * e1[1];
            q[1] = s[2] * e1[0] - s[0] * e1[2];
            q[2] = s[0] * e1[1] - s[1] * e1[0];
            float v = (ray[0] * q[0] + ray[1] * q[1] + ray[2] * q[2]) * inv_a;
            if (v < 0 || u + v > 1) {
              continue;
            }

            float t = (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) * inv_a;
            if (t > eps){
                // TODO color by closest vertex
                colors[x*3+0] = verts_colors[faces[f*3+0]*3+2]; // R = B
                colors[x*3+1] = verts_colors[faces[f*3+0]*3+1]; // G = G
                colors[x*3+2] = verts_colors[faces[f*3+0]*3+0]; // B = R

                endpoints[x*3+0] = origin[0] + ray[0] * t;
                endpoints[x*3+1] = origin[1] + ray[1] * t;
                endpoints[x*3+2] = origin[2] + ray[2] * t;
                //printf("ep %f, %f, %f", endpoints[x*3+0], endpoints[x*3+1], endpoints[x*3+2]);
                break;
            }
        }
    }
    
    """)

    cuda_intersection = cuda_src_mod.get_function("intersection")

    # Determine block/grid size on GPU
    gpu_dev = cuda.Device(0)
    max_gpu_threads_per_block = gpu_dev.MAX_THREADS_PER_BLOCK
    # n_blocks = int(np.ceil(float(len(rays))/float(max_gpu_threads_per_block)))
    # grid_dim_x = min(gpu_dev.MAX_GRID_DIM_X,int(np.floor(np.cbrt(n_blocks))))
    # grid_dim_y = min(gpu_dev.MAX_GRID_DIM_Y,int(np.floor(np.sqrt(n_blocks/grid_dim_x))))
    # grid_dim_z = min(gpu_dev.MAX_GRID_DIM_Z,int(np.ceil(float(n_blocks)/float(grid_dim_x*grid_dim_y))))
    # max_gpu_grid_dim = np.array([grid_dim_x,grid_dim_y,grid_dim_z]).astype(int)
    # n_gpu_loops = int(np.ceil(float(len(rays))/float(np.prod(max_gpu_grid_dim)*max_gpu_threads_per_block)))
    # print("#rays", len(rays))
    # print("max_gpu_threads_per_block",max_gpu_threads_per_block)
    # print("n_blocks", n_blocks)
    # print("max_gpu_grid_dim",max_gpu_grid_dim)
    n_gpu_loops = int(len(rays) / max_gpu_threads_per_block)
    # max_gpu_threads_per_block = 1
    # n_gpu_loops = max(n_gpu_loops, 1)

    no_of_faces = faces.shape[0]
    for gpu_loop_idx in range(n_gpu_loops):
        cuda_intersection(# inputs
                          rays_gpu, cuda.InOut(origin), vertices_gpu, vertices_colors_gpu, faces_gpu,
                          cuda.InOut(np.asarray([H, W, no_of_faces, gpu_loop_idx],np.float32)),
                          # outputs  
                          endpoints_gpu,
                          colors_gpu,
                          block=(max_gpu_threads_per_block,1,1),
                        #   grid=(int(max_gpu_grid_dim[0]),int(max_gpu_grid_dim[1]),int(max_gpu_grid_dim[2]))
                          )
    cuda.memcpy_dtoh(endpoints, endpoints_gpu)
    cuda.memcpy_dtoh(colors, colors_gpu)
    return endpoints, colors

# from https://en.wikipedia.org/wiki/Möller–Trumbore_intersection_algorithm
def ray_triangle_intersection(ray, origin, triangle):
    assert len(ray) == 3
    assert len(origin) == 3
    assert triangle.shape == (3,3)

    # Run Möller-Trumbore algorithm
    eps = 0.000001
    v0 = triangle[0,:]
    v1 = triangle[1,:]
    v2 = triangle[2,:]
    edge1 = v1 - v0
    edge2 = v2 - v0

    h = np.cross(ray, edge2)
    a = edge1.dot(h)
    if abs(a) < eps:
        return False, []  # This ray is parallel to this triangle.

    inv_a = 1. /a
    s = origin - v0
    u = inv_a * s.dot(h)
    if (u < 0. or u > 1.):
        return False, []

    q = np.cross(s, edge1)
    v = inv_a * ray.dot(q)
    if (v < 0. or u+v > 1.):
        return False, []

    # Now we can compute t to find out where the intersection point is on the line.
    t = inv_a * edge2.dot(q)
    if (t > eps):
        intersection_point = origin + ray * t
        return True, intersection_point
    else:  # There is a line intersection but not a ray intersection.
        return False, []

if __name__ == "__main__":
    rays = np.array([[-39.5, -25.5, -1.7],
                     [-39.5, -25.5, -1.7]])
    depth = np.linalg.norm(rays, 2, axis=1)
    depth = depth[0]
    ray = np.array([-39.5, -25.5, -1.7])
    scan_x = ray[0]
    scan_y = ray[1]
    scan_z = ray[2]
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    print("pitch", pitch*180/np.pi)
    print("yaw",yaw*180/np.pi)
    print("depth",depth)

    pitch = np.pi/2 - pitch

    point_x = np.sin(pitch) * np.cos(-yaw)
    point_y = np.sin(pitch) * np.sin(-yaw)
    point_z = np.cos(pitch)

    ray_new = np.array([point_x, point_y, point_z])

    print("ray_new", ray_new)

    origin = np.array([0, 0, 0])

    vertices =  np.array([[-39.5      , -25.5      ,  -1.7492892],
                         [-39.5      , -25.749289 ,  -1.5      ],
                          [-39.74929  , -25.5      ,  -1.5      ],
                          [39.74929   , 25.5       ,  -1.5      ]])
    vertices_colors =  np.ones(vertices.shape)

    faces = np.array([[0, 1, 2],[0, 1, 3]])
    # GPU_MODE = False
    if GPU_MODE:
        endpoints, colors = ray_mesh_intersection_CUDA(rays, origin, vertices, vertices_colors, faces, H=1, W=1)
        print("\n\nendpoints\n", endpoints)
        print("Done")
    else:
        triangle = np.array([[-39.5      , -25.5      ,  -1.7492892],
                             [-39.5      , -25.749289 ,  -1.5      ],
                             [-39.74929  , -25.5      ,  -1.5      ]])
        valid, point = ray_triangle_intersection(ray, origin, triangle)
        print(valid, point)
        # valid, point = ray_triangle_intersection(ray_new, origin, triangle)
        # print(valid, point)