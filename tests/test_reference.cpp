// The C++ port must give the same answers as the Python pipeline the models
// were measured with (training/eval_e2e.py). Reference: training/make_reference.py.
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>

#include <nlohmann/json.hpp>

#include "whoosh_lpr/pipeline.hpp"

namespace fs = std::filesystem;

namespace {
// The repository: compiled in, or WHOOSH_LPR_ROOT at run time (tests in Docker mount it elsewhere).
const fs::path kRoot = std::getenv("WHOOSH_LPR_ROOT") ? std::getenv("WHOOSH_LPR_ROOT") : WHOOSH_LPR_ROOT;
}

TEST(Reference, SameNumbersAsPython) {
    const fs::path ref = kRoot / "tests/data/reference.json";
    if (!fs::exists(kRoot / "models/ocr.onnx") || !fs::exists(kRoot / "data/raw"))
        GTEST_SKIP() << "needs models/ and data/raw/";
    std::ifstream in(ref);
    ASSERT_TRUE(in) << ref;
    const auto photos = nlohmann::json::parse(in);

    whoosh::Options options;
    options.min_confidence = 0;  // compare every reading, not only confident ones
    const whoosh::Recognizer recognizer(kRoot / "models", options);
    int compared = 0;
    for (const auto& photo : photos) {
        const auto file = photo["file"].get<std::string>();
        const auto result = recognizer.recognize(whoosh::load_photo(kRoot / "data/raw" / file));
        // Low-confidence junk may flip on a rounding difference; confident readings must not.
        for (const auto& want : photo["plates"]) {
            if (want["confidence"].get<float>() < 0.3f) continue;
            const auto it = std::find_if(result.plates.begin(), result.plates.end(), [&](const whoosh::Plate& p) {
                return p.text == want["text"].get<std::string>();
            });
            ASSERT_NE(it, result.plates.end()) << file << ": no " << want["text"];
            EXPECT_NEAR(it->confidence, want["confidence"].get<float>(), 0.02) << file << " " << it->text;
            EXPECT_EQ(it->kind == whoosh::PlateKind::Rear, want["rear"].get<bool>()) << file << " " << it->text;
            ++compared;
        }
    }
    EXPECT_GT(compared, 20);
}
