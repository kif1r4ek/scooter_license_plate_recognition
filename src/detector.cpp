#include "whoosh_lpr/detector.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "whoosh_lpr/cv_compat.hpp"

namespace whoosh {

Detector::Detector(Ort::Env& env, const std::filesystem::path& model, const Ort::SessionOptions& options)
    : session_(env, model.c_str(), options) {
    Ort::AllocatorWithDefaultOptions alloc;
    input_ = session_.GetInputNameAllocated(0, alloc).get();
    output_ = session_.GetOutputNameAllocated(0, alloc).get();
}

std::vector<Detection> Detector::detect(const cv::Mat& bgr) const {
    // letterbox: keep proportions, centre, pad with grey
    const double k = double(kSize) / std::max(bgr.cols, bgr.rows);
    const int nw = int(std::lround(bgr.cols * k)), nh = int(std::lround(bgr.rows * k));
    const int dx = (kSize - nw) / 2, dy = (kSize - nh) / 2;
    cv::Mat canvas(kSize, kSize, CV_8UC3, cv::Scalar::all(114));
    cv::resize(bgr, canvas(cv::Rect(dx, dy, nw, nh)), {nw, nh}, 0, 0, cv::INTER_LINEAR);

    // HWC BGR uint8 -> CHW RGB float 0..1
    std::vector<float> input(3 * kSize * kSize);
    std::vector<cv::Mat> planes(3);
    for (int c = 0; c < 3; ++c) planes[c] = cv::Mat(kSize, kSize, CV_32F, input.data() + c * kSize * kSize);
    cv::Mat rgb;
    cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3, 1.0 / 255);
    cv::split(rgb, planes);

    const std::array<int64_t, 4> shape{1, 3, kSize, kSize};
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value tensor = Ort::Value::CreateTensor<float>(memory, input.data(), input.size(), shape.data(), shape.size());
    const char* in_names[] = {input_.c_str()};
    const char* out_names[] = {output_.c_str()};
    auto outputs = session_.Run(Ort::RunOptions{nullptr}, in_names, &tensor, 1, out_names, 1);

    // [1, 4 + classes + 1, anchors]: cx cy w h, class scores, angle
    const auto dims = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    const int rows = int(dims[1]), anchors = int(dims[2]);
    const int classes = rows - 5;
    if (classes < 1) throw std::runtime_error("unexpected detector output shape");
    const float* out = outputs[0].GetTensorData<float>();
    auto at = [&](int row, int i) { return out[row * anchors + i]; };

    struct Candidate { int index; int cls; float score; };
    std::vector<Candidate> found;
    for (int i = 0; i < anchors; ++i) {
        int best = 0;
        for (int c = 1; c < classes; ++c)
            if (at(4 + c, i) > at(4 + best, i)) best = c;
        if (at(4 + best, i) >= min_confidence) found.push_back({i, best, at(4 + best, i)});
    }
    std::stable_sort(found.begin(), found.end(), [](auto& a, auto& b) { return a.score > b.score; });

    std::vector<Detection> kept;
    for (const auto& c : found) {
        const float cx = float((at(0, c.index) - dx) / k), cy = float((at(1, c.index) - dy) / k);
        const float w = float(at(2, c.index) / k), h = float(at(3, c.index) / k);
        Quad box = corners(cx, cy, w, h, at(rows - 1, c.index));
        const bool overlaps = std::any_of(kept.begin(), kept.end(),
                                          [&](const Detection& d) { return iou(box, d.box) >= nms_iou; });
        if (!overlaps) kept.push_back({box, PlateKind(c.cls), c.score});
    }
    return kept;
}

}  // namespace whoosh
