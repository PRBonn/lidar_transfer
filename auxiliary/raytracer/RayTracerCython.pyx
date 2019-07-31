
import cython
import numpy as np
cimport numpy as np

cdef extern from "RayTracer.cpp":
    void demo()

cdef extern from "RayTracer.cpp":
    void ctrace(float*, float*, float*, int*, int*, int, int, int, int, float*, int*)

cdef extern from "BBox.cpp":
    pass

cdef extern from "BVH.cpp":
    pass

def C_Trace(np.ndarray[float, ndim=1, mode='c'] rays,
            np.ndarray[float, ndim=1, mode='c'] origin,
            np.ndarray[float, ndim=1, mode='c'] verts,
            np.ndarray[int, ndim=1, mode='c'] faces,
            np.ndarray[int, ndim=1, mode='c'] colors,
            H, W):
    n_rays = len(rays)//3
    n_verts = len(verts)//3  # = n_colors
    n_faces = len(faces)//3
    s = (len(rays),)
    cdef np.ndarray[float, ndim=1, mode='c'] ray_endpoints = np.zeros(s).astype(np.float32)
    cdef np.ndarray[int, ndim=1, mode='c'] ray_colors = np.zeros(s).astype(np.int32)
    ctrace(&rays[0], &origin[0], &verts[0], &faces[0], &colors[0],
           n_rays, n_verts, n_faces, H,
           &ray_endpoints[0], &ray_colors[0])

    return ray_endpoints.reshape(-1,3), ray_colors.reshape(-1,3)