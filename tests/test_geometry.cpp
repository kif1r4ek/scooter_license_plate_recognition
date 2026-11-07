#include <gtest/gtest.h>

#include <numbers>

#include "whoosh_lpr/geometry.hpp"

using namespace whoosh;

TEST(Geometry, CornersOfAnAxisAlignedBox) {
    const Quad q = corners(10, 20, 8, 4, 0);
    EXPECT_FLOAT_EQ(q[0].x, 6);
    EXPECT_FLOAT_EQ(q[0].y, 18);
    EXPECT_FLOAT_EQ(q[2].x, 14);
    EXPECT_FLOAT_EQ(q[2].y, 22);
}

TEST(Geometry, QuarterTurnSwapsSides) {
    const Quad a = corners(0, 0, 8, 4, 0);
    const Quad b = corners(0, 0, 4, 8, std::numbers::pi_v<float> / 2);
    EXPECT_NEAR(iou(a, b), 1.0, 1e-5);
}

TEST(Geometry, Iou) {
    EXPECT_NEAR(iou(corners(0, 0, 2, 2, 0), corners(1, 0, 2, 2, 0)), 1.0 / 3, 1e-6);
    EXPECT_EQ(iou(corners(0, 0, 2, 2, 0), corners(5, 5, 2, 2, 0)), 0.0);
}

TEST(Geometry, LongSideFirst) {
    const Quad tall = corners(0, 0, 4, 8, 0);
    const Quad q = long_side_first(tall);
    EXPECT_GT(cv::norm(q[1] - q[0]), cv::norm(q[3] - q[0]));
    EXPECT_EQ(long_side_first(q)[0], q[0]);
}

TEST(Geometry, RollWrapsAround) {
    const Quad q = corners(0, 0, 4, 2, 0);
    EXPECT_EQ(roll(q, 2)[0], q[2]);
    EXPECT_EQ(roll(q, 3)[1], q[0]);
    EXPECT_EQ(roll(q, -1)[0], q[3]);
}
