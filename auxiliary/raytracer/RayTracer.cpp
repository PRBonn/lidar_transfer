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
           int n_rays, int n_verts, int n_faces, int height, float* endpoints, int* endcolors) {
  vector<Object*> objects;
  // for(int i=0; i<10; ++i) {
  //   printf("r %f %f %f\n", rays[i*3+0], rays[i*3+1], rays[i*3+2]);
  //   printf("v %f %f %f\n", verts[i*3+0], verts[i*3+1], verts[i*3+2]);
  //   printf("f %d %d %d\n", faces[i*3+0], faces[i*3+1], faces[i*3+2]);
  //   printf("c %f %f %f\n", colors[i*3+0], colors[i*3+1], colors[i*3+2]);
  // }

  printf("Constructing %d triangles...\n", n_faces);
  for(int i=0; i<n_faces; ++i) {
    int idx = faces[i*3+0]*3;
    Vector3 v0(verts[idx+0], verts[idx+1], verts[idx+2]);
    Vector3 c0(colors[idx+0], colors[idx+1], colors[idx+2]);
    idx = faces[i*3+1]*3;
    Vector3 v1(verts[idx+0], verts[idx+1], verts[idx+2]);
    Vector3 c1(colors[idx+0], colors[idx+1], colors[idx+2]);
    idx = faces[i*3+2]*3;
    Vector3 v2(verts[idx+0], verts[idx+1], verts[idx+2]);
    Vector3 c2(colors[idx+0], colors[idx+1], colors[idx+2]);
    // printf("v %f %f %f\n", v0[0], v0[1], v0[2]);
    objects.push_back(new Triangle(v0, v1, v2, c0, c1, c2));
  }

  // Compute a BVH for this object set
  BVH bvh(&objects);

  // Allocate space for some image pixels
  const unsigned int width=n_rays/height ;
  // float* pixels = new float[width*height*3];

  Vector3 origin(origin_in[0], origin_in[1], origin_in[2]);

  printf("Rendering image (%dx%d)...\n", width, height);
  // Raytrace over every pixel
#pragma omp parallel for
  for(size_t i=0; i<width; ++i) {
    for(size_t j=0; j<height; ++j) {
      size_t index = 3*(width * j + i);

      Vector3 single_ray(rays[index+0], rays[index+1], rays[index+2]);
      Ray ray(origin, normalize(single_ray));

      IntersectionInfo I;
      bool hit = bvh.getIntersection(ray, &I, false);

      if(!hit) {
        // pixels[index] = pixels[index+1] = pixels[index+2] = 0.f;
      } else {
        // const Vector3 color(fabs(c0.x), fabs(c0.y), fabs(c0.z));
        // const Vector3 normal = I.object->getNormal(I);
        Vector3 colors = I.object->getColor(0);
        Vector3 point = I.hit;
        // const Vector3 color(fabs(normal.x), fabs(normal.y), fabs(normal.z));
        endpoints[index+0] = point[0];
        endpoints[index+1] = point[1];
        endpoints[index+2] = point[2];

        // pixels[index+0] = colors.x;
        endcolors[index+0] = colors.x;
        // pixels[index+1] = colors.y;
        endcolors[index+1] = colors.y;
        // pixels[index+2] = colors.z;
        endcolors[index+2] = colors.z;
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

  // Cleanup
  // delete[] pixels;
}

void demo() {
   // Create a million triangles packed in the space of a cube
  const unsigned int N = 10000;
  vector<Object*> objects;
  printf("Constructing %d triangles...\n", N);
  for(size_t i=0; i<N; ++i) {
    objects.push_back(new Triangle(randVector3(), randVector3(), randVector3(), randVector3(), randVector3(), randVector3()));
  // objects.push_back(new Triangle(Vector3(0,0,1), Vector3(0,0,0), Vector3(1,0,0), Vector3(1,0,0), Vector3(1,0,0), Vector3(1,0,0)));
  // objects.push_back(new Triangle(Vector3(0,0,-1), Vector3(0,0,0), Vector3(-1,0,0), Vector3(0,1,0), Vector3(0,1,0), Vector3(0,1,0)));
  }

  // Compute a BVH for this object set
  BVH bvh(&objects);

  // Allocate space for some image pixels
  const unsigned int width=800, height=64;
  float* pixels = new float[width*height*3];

  // Create a camera from position and focus point
  Vector3 origin(1.6, 1.3, 1.6);
  Vector3 camera_focus(0,0,0);
  Vector3 camera_up(0,1,0);

  // Camera tangent space
  Vector3 camera_dir = normalize(camera_focus - origin);
  Vector3 camera_u = normalize(camera_dir ^ camera_up);
  Vector3 camera_v = normalize(camera_u ^ camera_dir);

  printf("Rendering image (%dx%d)...\n", width, height);
  // Raytrace over every pixel
#pragma omp parallel for
  for(size_t i=0; i<width; ++i) {
    for(size_t j=0; j<height; ++j) {
      size_t index = 3*(width * j + i);

      float u = (i+.5f) / (float)(width-1) - .5f;
      float v = (height-1-j+.5f) / (float)(height-1) - .5f;
      float fov = .5f / tanf( 70.f * 3.14159265*.5f / 180.f);

      // This is only valid for square aspect ratio images
      Ray ray(origin, normalize(u*camera_u + v*camera_v + fov*camera_dir));

      IntersectionInfo I;
      bool hit = bvh.getIntersection(ray, &I, false);

      if(!hit) {
        pixels[index] = pixels[index+1] = pixels[index+2] = 0.f;
      } else {
        // const Vector3 color(fabs(c0.x), fabs(c0.y), fabs(c0.z));
        // const Vector3 normal = I.object->getNormal(I);
        Vector3 colors = I.object->getColor(0);
        // const Vector3 color(fabs(normal.x), fabs(normal.y), fabs(normal.z));

        pixels[index+0] = colors.x;
        // pixels[index+0] = 1;
        pixels[index+1] = colors.y;
        // pixels[index+1] = 1;
        pixels[index+2] = colors.z;
        // pixels[index+2] = 1;
      }
    }
  }

  // Output image file (PPM Format)
  printf("Writing out image file: \"render.ppm\"\n");
  FILE *image = fopen("render.ppm", "w");
  fprintf(image, "P6\n%d %d\n255\n", width, height);
  for(size_t j=0; j<height; ++j) {
    for(size_t i=0; i<width; ++i) {
      size_t index = 3*(width * j + i);
      unsigned char r = std::max(std::min(pixels[index+0]*255.f, 255.f), 0.f);
      unsigned char g = std::max(std::min(pixels[index+1]*255.f, 255.f), 0.f);
      unsigned char b = std::max(std::min(pixels[index+2]*255.f, 255.f), 0.f);
      fprintf(image, "%c%c%c", r,g,b);
    }
  }
  fclose(image);

  // Cleanup
  delete[] pixels;
}

extern "C"
{
  void cdemo() {demo(); printf("Demo done\n");}
  void ctrace(float* rays, float* origin, float* verts, int* faces, int* colors,
              int n_rays, int n_verts, int n_faces, int height,
              float* endpoints, int* endcolors) {
    trace(rays, origin, verts, faces, colors, n_rays, n_verts, n_faces, height,
          endpoints, endcolors);
    printf("Trace done\n");
  }
}