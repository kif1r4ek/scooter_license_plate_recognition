#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "whoosh_lpr/detector.hpp"
#include "whoosh_lpr/reader.hpp"

namespace whoosh {

struct Plate {
    std::string text;
    PlateKind kind;
    float confidence;  // of the reading
    float detection;   // of the box
    Quad box;
};

// A number seen in a photo, possibly on both the front and the rear plate.
struct Number {
    std::string text;
    float confidence;  // best of its plates
    bool front = false;
    bool rear = false;
};

struct Result {
    std::vector<Plate> plates;   // every plate read with enough confidence
    std::vector<Number> numbers; // distinct numbers, most confident first
};

struct Options {
    float min_confidence = 0.3f;  // below this a reading is dropped (asphalt, tiles, worn paint)
    int threads = 0;              // 0: all cores
};

class Recognizer {
public:
    Recognizer(const std::filesystem::path& model_dir, const Options& options);
    ~Recognizer();

    Result recognize(const cv::Mat& bgr) const;

    const Options& options() const { return options_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    Options options_;
};

// Opens a photo the way the camera meant it to be seen (EXIF orientation).
cv::Mat load_photo(const std::filesystem::path& path);

}  // namespace whoosh
