#ifndef Triangle_h_
#define Triangle_h_

#include <cmath>
#include "Object.h"

//! For the purposes of demonstrating the BVH, a simple Triangle
struct Triangle : public Object {
  // Vertices of triangle
  Vector3 v0;
  Vector3 v1;
  Vector3 v2;

  // Colors of vertices
  Vector3 c0;
  Vector3 c1;
  Vector3 c2;

  // Remissions of vertices
  Vector3 r;

  Triangle(const Vector3& v0, const Vector3& v1, const Vector3& v2,
           const Vector3& c0, const Vector3& c1, const Vector3& c2,
           const Vector3& r)
    : v0(v0), v1(v1), v2(v2), c0(c0), c1(c1), c2(c2), r(r) { }

  bool getIntersection(const Ray& ray, IntersectionInfo* I) const {
    Vector3 e1 = v1 - v0;
    Vector3 e2 = v2 - v0;
    Vector3 h = ray.d ^ e2;
    float a = e1 * h;
    float eps = 0.000001f;
    if (a < eps && a > -eps) return false;

    float inv_a = 1.0f/a;
    Vector3 s = ray.o - v0;
    float u = (s * h) * inv_a;
    if (u < 0 || u > 1) return false;

    Vector3 q = s ^ e1;
    float v = (ray.d * q) * inv_a;
    if (v < 0 || u + v > 1) return false;

    float t = (e2 * q) * inv_a;
    if (t < eps) return false;

    I->object = this;
    I->t = t; // set distance
    return true;
  }

  Vector3 getNormal(const IntersectionInfo& I) const {
    return normalize(v0 ^ v1);
  }

  Vector3 getColor(const int idx) const {
    if (idx == 0) return c0;
    if (idx == 1) return c1;
    if (idx == 2) return c2;
    return c0;
  }

  float getRemissions(const int idx) const {
    // if (idx == 0) return r[0];
    // if (idx == 1) return r[1];
    // if (idx == 2) return r[2];
    // return r[0];
    // Interpolate remissions
    return (r[0]+r[1]+r[2])/3;
  }

  BBox getBBox() const {
    Vector3 min_vals = min(v0, min(v1, v2));
    Vector3 max_vals = max(v0, max(v1, v2));
    return BBox(min_vals, max_vals);
  }

  Vector3 getCentroid() const {
    return (v0 + v1 + v2) / 3;
  }

};

#endif
