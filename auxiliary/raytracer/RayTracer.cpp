#include <cstdio>
#include <vector>
#include <cstdlib>
#include "BVH.h"
#include "Sphere.h"
#include "Triangle.h"
using std::vector;

// Return a random number in [0,1]
float rand01() {
  return rand() * (1.f / RAND_MAX);
}

// Return a random vector with each component in the range [-1,1]
Vector3 randVector3() {
  return Vector3(rand01(), rand01(), rand01())*2.f - Vector3(1,1,1);
}

void trace(float* rays, float* origin_in, float* verts, int* faces, int* colors,
           float * rem, int n_rays, int n_verts, int n_faces, int height,
           float* endpoints, int* endcolors, float* range, float * endrem) {
  vector<Object*> objects;

  // for(int i=0; i<10; ++i) {
  //   printf("r %f %f %f\n", rays[i*3+0], rays[i*3+1], rays[i*3+2]);
  //   printf("v %f %f %f\n", verts[i*3+0], verts[i*3+1], verts[i*3+2]);
  //   printf("f %d %d %d\n", faces[i*3+0], faces[i*3+1], faces[i*3+2]);
  //   printf("c %f %f %f\n", colors[i*3+0], colors[i*3+1], colors[i*3+2]);
  // }

  printf("Constructing %d triangles...\n", n_faces);
  for(int i=0; i<n_faces; ++i) {
    Vector3 r(0.0, 0.0, 0.0);
    int idx = faces[i*3+0]*3;
    Vector3 v0(verts[idx+0], verts[idx+1], verts[idx+2]);
    Vector3 c0(colors[idx+0], colors[idx+1], colors[idx+2]);
    r[0] = rem[idx/3];

    idx = faces[i*3+1]*3;
    Vector3 v1(verts[idx+0], verts[idx+1], verts[idx+2]);
    Vector3 c1(colors[idx+0], colors[idx+1], colors[idx+2]);
    r[1] = rem[idx/3];

    idx = faces[i*3+2]*3;
    Vector3 v2(verts[idx+0], verts[idx+1], verts[idx+2]);
    Vector3 c2(colors[idx+0], colors[idx+1], colors[idx+2]);
    r[2] = rem[idx/3];

    // printf("v %f %f %f\n", v0[0], v0[1], v0[2]);
    objects.push_back(new Triangle(v0, v1, v2, c0, c1, c2, r));
  }

  // Compute a BVH for this object set
  BVH bvh(&objects);

  const unsigned int width=n_rays/height ;

  Vector3 origin(origin_in[0], origin_in[1], origin_in[2]);

  printf("Rendering image (%dx%d)...\n", height, width);
  // Raytrace over every pixel
#pragma omp parallel for
  for(size_t i=0; i<width; ++i) {
    for(int j=0; j<height; ++j) {
      size_t index = 3*(width * j + i);

      Vector3 single_ray(rays[index+0], rays[index+1], rays[index+2]);
      Ray ray(origin, normalize(single_ray));

      IntersectionInfo I;
      bool hit = bvh.getIntersection(ray, &I, false);

      if(hit) {
        // No interpolation return always color of first index
        Vector3 colors = I.object->getColor(0);
        float remission = I.object->getRemissions(0);
        Vector3 point = I.hit;
        endpoints[index+0] = point[0];
        endpoints[index+1] = point[1];
        endpoints[index+2] = point[2];

        endcolors[index+0] = colors.x;
        endcolors[index+1] = colors.y;
        endcolors[index+2] = colors.z;

        endrem[index/3] = remission;

        // Update range image
        range[width * j + i] = I.t;
      }
    }
  }

  // Output image file (PPM Format)
  // printf("Writing out image file: \"render.ppm\"\n");
  // FILE *image = fopen("render.ppm", "w");
  // fprintf(image, "P6\n%d %d\n255\n", width, height);
  // for(size_t j=0; j<height; ++j) {
  //   for(size_t i=0; i<width; ++i) {
  //     size_t index = 3*(width * j + i);
  //     unsigned char r = std::max(std::min(pixels[index+0]*1.f, 255.f), 0.f);
  //     unsigned char g = std::max(std::min(pixels[index+1]*1.f, 255.f), 0.f);
  //     unsigned char b = std::max(std::min(pixels[index+2]*1.f, 255.f), 0.f);
  //     fprintf(image, "%c%c%c", r,g,b);
  //   }
  // }
  // fclose(image);

  // loop over object vector delete 
  for (Object* p : objects) {
     delete p;
}
   objects.clear();
}

extern "C"
{
  void ctrace(float* rays, float* origin, float* verts, int* faces, int* colors,
              float * rem, int n_rays, int n_verts, int n_faces, int height,
              float* endpoints, int* endcolors, float* range, float * endrem) {
    trace(rays, origin, verts, faces, colors, rem, n_rays, n_verts, n_faces, height,
          endpoints, endcolors, range, endrem);
  }
}
