#pragma once

#include <filesystem>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include "whoosh_lpr/geometry.hpp"

namespace whoosh {

enum class PlateKind { Front = 0, Rear = 1 };

struct Detection {
    Quad box;  // in pixels of the original photo
    PlateKind kind;
    float confidence;
};

// YOLO11 OBB, input 1280x1280 RGB letterboxed with grey 114.
class Detector {
public:
    static constexpr int kSize = 1280;

    Detector(Ort::Env& env, const std::filesystem::path& model, const Ort::SessionOptions& options);

    std::vector<Detection> detect(const cv::Mat& bgr) const;

    float min_confidence = 0.25f;
    float nms_iou = 0.3f;

private:
    mutable Ort::Session session_;
    std::string input_;
    std::string output_;
};

}  // namespace whoosh
