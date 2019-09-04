import cython
import numpy as np
cimport numpy as np

cdef extern from "RayTracer.cpp":
    void ctrace(float*, float*, float*, int*, int*, float *, int, int, int,
                int, float*, int*, float*, float*)

cdef extern from "BBox.cpp":
    pass

cdef extern from "BVH.cpp":
    pass

def C_Trace(float[::1] rays,
            float[::1] origin,
            float[::1] verts,
            int[::1] faces,
            int[::1] colors,
            float[::1] rem,
            float[::1] ray_endpoints,
            int[::1] ray_colors,
            float[::1] range_image,
            float[::1] rem_image,
            H, W):
    n_rays = len(rays)//3
    n_verts = len(verts)//3  # = n_colors
    n_faces = len(faces)//3
    s = (len(rays),)

    ctrace(&rays[0], &origin[0], &verts[0], &faces[0], &colors[0],
           &rem[0], n_rays, n_verts, n_faces, H,
           &ray_endpoints[0], &ray_colors[0], &range_image[0], &rem_image[0])
