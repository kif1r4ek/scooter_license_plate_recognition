#include "whoosh_lpr/pipeline.hpp"

#include <algorithm>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace whoosh {

struct Recognizer::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "whoosh-lpr"};
    Ort::SessionOptions options;
    std::unique_ptr<Detector> detector;
    std::unique_ptr<Reader> reader;
};

Recognizer::Recognizer(const std::filesystem::path& model_dir, const Options& options)
    : impl_(std::make_unique<Impl>()), options_(options) {
    for (const char* name : {"detector.onnx", "ocr.onnx"})
        if (!std::filesystem::exists(model_dir / name))
            throw std::runtime_error("model not found: " + (model_dir / name).string());
    impl_->options.SetIntraOpNumThreads(options.threads);
    impl_->options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    impl_->detector = std::make_unique<Detector>(impl_->env, model_dir / "detector.onnx", impl_->options);
    impl_->reader = std::make_unique<Reader>(impl_->env, model_dir / "ocr.onnx", impl_->options);
}

Recognizer::~Recognizer() = default;

Result Recognizer::recognize(const cv::Mat& bgr) const {
    Result result;
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    for (const Detection& det : impl_->detector->detect(bgr)) {
        Reading r = impl_->reader->read(gray, det);
        if (r.confidence < options_.min_confidence) continue;
        result.plates.push_back({r.text, det.kind, r.confidence, det.confidence, r.box});
    }

    for (const Plate& p : result.plates) {
        auto it = std::find_if(result.numbers.begin(), result.numbers.end(),
                               [&](const Number& n) { return n.text == p.text; });
        if (it == result.numbers.end()) it = result.numbers.insert(result.numbers.end(), {p.text, 0});
        it->confidence = std::max(it->confidence, p.confidence);
        (p.kind == PlateKind::Front ? it->front : it->rear) = true;
    }
    std::sort(result.numbers.begin(), result.numbers.end(),
              [](const Number& a, const Number& b) { return a.confidence > b.confidence; });
    return result;
}

cv::Mat load_photo(const std::filesystem::path& path) {
    // imread turns the picture by its EXIF orientation, like the phone's gallery does
    cv::Mat img = cv::imread(path.string(), cv::IMREAD_COLOR);
    if (img.empty()) throw std::runtime_error("cannot read image: " + path.string());
    return img;
}

}  // namespace whoosh
