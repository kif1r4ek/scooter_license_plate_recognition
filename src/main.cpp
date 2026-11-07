// whoosh-lpr: prints the Whoosh scooter numbers seen in phone photos.
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "whoosh_lpr/pipeline.hpp"

namespace fs = std::filesystem;
using whoosh::PlateKind;

namespace {

const std::set<std::string> kExtensions{".jpg", ".jpeg", ".png", ".webp", ".bmp"};

bool is_photo(const fs::path& p) {
    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
    return kExtensions.count(ext) > 0;
}

std::vector<fs::path> collect(const std::vector<std::string>& inputs, bool recursive) {
    std::vector<fs::path> photos;
    for (const auto& in : inputs) {
        const fs::path p(in);
        if (fs::is_directory(p)) {
            std::vector<fs::path> found;
            auto add = [&](const fs::directory_entry& e) {
                if (e.is_regular_file() && is_photo(e.path())) found.push_back(e.path());
            };
            if (recursive)
                for (const auto& e : fs::recursive_directory_iterator(p)) add(e);
            else
                for (const auto& e : fs::directory_iterator(p)) add(e);
            std::sort(found.begin(), found.end());
            photos.insert(photos.end(), found.begin(), found.end());
        } else if (fs::is_regular_file(p)) {
            photos.push_back(p);
        } else {
            spdlog::error("нет такого файла или папки: {}", in);
        }
    }
    return photos;
}

// Models: --models, then $WHOOSH_LPR_MODELS, then "models" next to the binary or in the current folder.
fs::path find_models(const std::string& option, const char* argv0) {
    if (!option.empty()) return option;
    if (const char* env = std::getenv("WHOOSH_LPR_MODELS")) return env;
    std::error_code ec;
    const fs::path exe = fs::canonical(fs::path(argv0), ec);
    for (const fs::path& dir : {exe.parent_path() / "models", exe.parent_path().parent_path() / "models",
                                fs::current_path() / "models"})
        if (!ec && fs::exists(dir / "detector.onnx")) return dir;
    return fs::current_path() / "models";
}

std::string describe(const whoosh::Number& n) {
    std::string where = n.front && n.rear ? "спереди и сзади" : n.rear ? "сзади" : "спереди";
    return n.text + "  (" + where + ", уверенность " + std::to_string(int(n.confidence * 100 + 0.5)) + "%)";
}

double rounded(double x, double step) { return std::round(x / step) * step; }

nlohmann::json to_json(const fs::path& photo, const whoosh::Result& r, double seconds) {
    nlohmann::json plates = nlohmann::json::array();
    for (const auto& p : r.plates) {
        nlohmann::json box = nlohmann::json::array();
        for (const auto& c : p.box) box.push_back({rounded(c.x, 0.1), rounded(c.y, 0.1)});
        plates.push_back({{"number", p.text},
                          {"plate", p.kind == PlateKind::Front ? "front" : "rear"},
                          {"confidence", rounded(p.confidence, 0.001)},
                          {"detection", rounded(p.detection, 0.001)},
                          {"box", box}});
    }
    nlohmann::json numbers = nlohmann::json::array();
    for (const auto& n : r.numbers) numbers.push_back(n.text);
    return {{"file", photo.string()}, {"numbers", numbers}, {"plates", plates}, {"seconds", seconds}};
}

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"Распознаёт номера самокатов Whoosh (формат AA0000) на фото с телефона."};
    std::vector<std::string> inputs;
    std::string models;
    bool json = false, recursive = false, verbose = false;
    whoosh::Options options;
    app.add_option("photos", inputs, "Фото или папки с фото")->required();
    app.add_option("-m,--models", models, "Папка с detector.onnx и ocr.onnx");
    app.add_option("-c,--min-confidence", options.min_confidence, "Порог уверенности чтения, 0..1")
        ->capture_default_str()->check(CLI::Range(0.0, 1.0));
    app.add_option("-t,--threads", options.threads, "Потоки CPU (0 = все)")->capture_default_str();
    app.add_flag("-r,--recursive", recursive, "Искать фото и во вложенных папках");
    app.add_flag("--json", json, "Вывод в JSON (по строке на фото)");
    app.add_flag("-v,--verbose", verbose, "Подробный лог");
    CLI11_PARSE(app, argc, argv);

    spdlog::set_level(verbose ? spdlog::level::debug : spdlog::level::warn);
    spdlog::set_pattern("%^[%l]%$ %v");
    const auto photos = collect(inputs, recursive);
    if (photos.empty()) {
        spdlog::error("нет фото для обработки");
        return 2;
    }

    std::unique_ptr<whoosh::Recognizer> recognizer;
    try {
        const fs::path dir = find_models(models, argv[0]);
        spdlog::debug("models: {}", dir.string());
        recognizer = std::make_unique<whoosh::Recognizer>(dir, options);
    } catch (const std::exception& e) {
        spdlog::error("{}", e.what());
        return 2;
    }

    int failed = 0, without = 0;
    for (const auto& photo : photos) {
        const auto start = std::chrono::steady_clock::now();
        whoosh::Result result;
        try {
            result = recognizer->recognize(whoosh::load_photo(photo));
        } catch (const std::exception& e) {
            spdlog::error("{}: {}", photo.string(), e.what());
            ++failed;
            continue;
        }
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        spdlog::debug("{}: {:.2f} s, {} plate(s)", photo.string(), seconds, result.plates.size());
        without += result.numbers.empty();

        if (json) {
            std::cout << to_json(photo, result, rounded(seconds, 0.01)).dump() << '\n';
            continue;
        }
        std::cout << photo.filename().string() << ": ";
        if (result.numbers.empty()) {
            std::cout << "номер не найден\n";
        } else if (result.numbers.size() == 1) {
            std::cout << describe(result.numbers[0]) << '\n';
        } else {
            std::cout << "номеров в кадре: " << result.numbers.size() << '\n';
            for (const auto& n : result.numbers) std::cout << "    " << describe(n) << '\n';
        }
    }
    if (!json && photos.size() > 1)
        std::cout << "\nФото: " << photos.size() << ", без номера: " << without << ", ошибок чтения файла: " << failed
                  << '\n';
    return failed ? 1 : 0;
}
