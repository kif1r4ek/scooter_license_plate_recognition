#pragma once

// OpenCV 5 moved the 2D geometry functions out of imgproc.
#include <opencv2/core/version.hpp>
#include <opencv2/imgproc.hpp>
#if CV_VERSION_MAJOR >= 5
#include <opencv2/geometry.hpp>
#endif
