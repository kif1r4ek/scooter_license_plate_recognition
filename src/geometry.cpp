#include "whoosh_lpr/geometry.hpp"

#include <cmath>
#include <vector>

#include "whoosh_lpr/cv_compat.hpp"

namespace whoosh {

Quad corners(float cx, float cy, float w, float h, float angle) {
    const float c = std::cos(angle), s = std::sin(angle);
    const cv::Point2f dx(c * w / 2, s * w / 2);
    const cv::Point2f dy(-s * h / 2, c * h / 2);
    const cv::Point2f p(cx, cy);
    return {p - dx - dy, p + dx - dy, p + dx + dy, p - dx + dy};
}

double iou(const Quad& a, const Quad& b) {
    std::vector<cv::Point2f> ha, hb, inter;
    cv::convexHull(std::vector<cv::Point2f>(a.begin(), a.end()), ha);
    cv::convexHull(std::vector<cv::Point2f>(b.begin(), b.end()), hb);
    const double common = cv::intersectConvexConvex(ha, hb, inter);
    const double total = cv::contourArea(ha) + cv::contourArea(hb) - common;
    return total > 0 ? common / total : 0.0;
}

Quad roll(const Quad& q, int k) {
    Quad r;
    for (int i = 0; i < 4; ++i) r[i] = q[((i + k) % 4 + 4) % 4];
    return r;
}

Quad long_side_first(const Quad& q) {
    return cv::norm(q[1] - q[0]) < cv::norm(q[3] - q[0]) ? roll(q, 1) : q;
}

}  // namespace whoosh
