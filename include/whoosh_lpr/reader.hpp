#pragma once

#include <filesystem>
#include <string>

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include "whoosh_lpr/detector.hpp"

namespace whoosh {

struct Reading {
    std::string text;  // always 2 Latin letters + 4 digits
    float confidence;  // weakest symbol x probability the crop is upright
    Quad box;          // turned so that corner 0 is the text's top-left
};

// Reads a plate inside a detector box. The box does not know which way the
// text runs, so every possible direction is read and the most likely one wins.
class Reader {
public:
    static constexpr int kHeight = 64;
    static constexpr int kWidth = 256;
    static constexpr float kMaxPlateSide = 480.0f;  // training crops were never bigger

    Reader(Ort::Env& env, const std::filesystem::path& model, const Ort::SessionOptions& options);

    Reading read(const cv::Mat& gray, const Detection& det) const;

private:
    mutable Ort::Session session_;
};

// Rectified model input, exactly as in training: H x W, a rear plate's two
// lines laid side by side. Exposed for tests and debugging.
cv::Mat plate_input(const cv::Mat& gray, const Quad& box, PlateKind kind);

}  // namespace whoosh
