#include "whoosh_lpr/reader.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "whoosh_lpr/cv_compat.hpp"

namespace whoosh {
namespace {

constexpr char kLetters[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
constexpr char kDigits[] = "0123456789";

cv::Mat warp(const cv::Mat& gray, const Quad& q, int w, int h) {
    const cv::Point2f target[4] = {{0, 0}, {float(w), 0}, {float(w), float(h)}, {0, float(h)}};
    cv::Mat out;
    cv::warpPerspective(gray, out, cv::getPerspectiveTransform(q.data(), target), {w, h}, cv::INTER_LINEAR,
                        cv::BORDER_REPLICATE);
    return out;
}

// Most likely class and its probability.
std::pair<int, float> best_of(const float* logits, int n) {
    const int best = int(std::max_element(logits, logits + n) - logits);
    double sum = 0;
    for (int i = 0; i < n; ++i) sum += std::exp(logits[i] - logits[best]);
    return {best, float(1.0 / sum)};
}

}  // namespace

cv::Mat plate_input(const cv::Mat& gray, const Quad& box, PlateKind kind) {
    using R = Reader;
    if (kind == PlateKind::Front) return warp(gray, box, R::kWidth, R::kHeight);
    const auto& [tl, tr, br, bl] = box;
    const cv::Point2f ml = (tl + bl) / 2, mr = (tr + br) / 2;
    cv::Mat out;
    cv::hconcat(warp(gray, {tl, tr, mr, ml}, R::kWidth / 2, R::kHeight),
                warp(gray, {ml, mr, br, bl}, R::kWidth / 2, R::kHeight), out);
    return out;
}

Reader::Reader(Ort::Env& env, const std::filesystem::path& model, const Ort::SessionOptions& options)
    : session_(env, model.c_str(), options) {}

Reading Reader::read(const cv::Mat& gray, const Detection& det) const {
    // Bring the plate down to the scale of the training crops: warpPerspective
    // has no area filter and would alias a close-up shot.
    Quad box = long_side_first(det.box);
    cv::Mat image = gray;
    const double side = std::max(cv::norm(box[1] - box[0]), cv::norm(box[3] - box[0]));
    if (const double k = kMaxPlateSide / side; k < 1) {
        cv::resize(gray, image, {}, k, k, cv::INTER_AREA);
        for (auto& p : box) p *= float(k);
    }

    // A front plate lies along its box; a rear plate can be narrower than tall.
    const std::vector<int> turns = det.kind == PlateKind::Front ? std::vector<int>{0, 2} : std::vector<int>{0, 1, 2, 3};
    const int n = int(turns.size());
    std::vector<float> input(size_t(n) * kHeight * kWidth);
    for (int i = 0; i < n; ++i) {
        cv::Mat dst(kHeight, kWidth, CV_32F, input.data() + size_t(i) * kHeight * kWidth);
        plate_input(image, roll(box, turns[i]), det.kind).convertTo(dst, CV_32F);
    }

    const std::array<int64_t, 4> shape{n, 1, kHeight, kWidth};
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value tensor = Ort::Value::CreateTensor<float>(memory, input.data(), input.size(), shape.data(), shape.size());
    const char* in_names[] = {"image"};
    const char* out_names[] = {"letters", "digits", "upright"};
    auto out = session_.Run(Ort::RunOptions{nullptr}, in_names, &tensor, 1, out_names, 3);
    const float* letters = out[0].GetTensorData<float>();
    const float* digits = out[1].GetTensorData<float>();
    const float* upright = out[2].GetTensorData<float>();

    Reading best{"", 0, box};
    double best_score = -1e30;
    for (int i = 0; i < n; ++i) {
        std::string text;
        double score = 0;
        float weakest = 1;
        for (int p = 0; p < 6; ++p) {
            const bool letter = p < 2;
            const auto [cls, prob] = letter ? best_of(letters + (i * 2 + p) * 26, 26)
                                            : best_of(digits + (i * 4 + p - 2) * 10, 10);
            text += letter ? kLetters[cls] : kDigits[cls];
            score += std::log(prob);
            weakest = std::min(weakest, prob);
        }
        const float up = 1.0f / (1.0f + std::exp(-upright[i]));
        score += std::log(up);
        if (score > best_score) {
            best_score = score;
            Quad turned = roll(box, turns[i]);
            if (image.data != gray.data)  // back to the photo's pixels
                for (auto& p : turned) p *= float(side / kMaxPlateSide);
            best = {text, weakest * up, turned};
        }
    }
    return best;
}

}  // namespace whoosh
