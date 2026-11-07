#pragma once

#include <array>

#include <opencv2/core.hpp>

namespace whoosh {

// Four corners of a plate box. For a read plate: text top-left, top-right,
// bottom-right, bottom-left. From the detector: any start, any direction.
using Quad = std::array<cv::Point2f, 4>;

// Corners of a rotated box as YOLO OBB gives it: centre, size, angle in radians.
Quad corners(float cx, float cy, float w, float h, float angle);

// Intersection over union of two convex quads.
double iou(const Quad& a, const Quad& b);

// Same box, starting k corners later (a quarter turn per step).
Quad roll(const Quad& q, int k);

// Same box with the long side first, so a plate lies horizontally.
Quad long_side_first(const Quad& q);

}  // namespace whoosh
