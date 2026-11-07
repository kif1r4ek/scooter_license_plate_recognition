# whoosh-lpr CLI: build, unit tests, small runtime image.
#
#   docker build -t whoosh-lpr .                      # the CLI
#   docker build --target build -t whoosh-lpr:build .  # compiler + tests, for scripts/test.sh
#   docker run --rm -v "$PWD/photos:/photos:ro" whoosh-lpr .

FROM ubuntu:24.04 AS build
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        g++ cmake ninja-build ca-certificates curl \
        libopencv-dev libcli11-dev nlohmann-json3-dev libspdlog-dev libgtest-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY CMakeLists.txt vcpkg.json CMakePresets.json ./
COPY cmake cmake
COPY include include
COPY src src
COPY tests tests
COPY models/detector.onnx models/ocr.onnx models/
# ONNX Runtime is not packaged by Ubuntu: CMake downloads the official release.
RUN cmake -S . -B /build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /build \
    && /build/whoosh_lpr_tests --gtest_filter='Geometry.*'
# Ubuntu packages the binary needs at run time (found from its shared libraries)
RUN ldd /build/whoosh-lpr | awk '/=> \//{print $3}' | xargs -r readlink -f | xargs -r dpkg -S 2>/dev/null \
      | cut -d: -f1 | sort -u > /build/runtime-packages.txt \
    && cat /build/runtime-packages.txt

FROM ubuntu:24.04 AS runtime
COPY --from=build /build/runtime-packages.txt /tmp/
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        $(cat /tmp/runtime-packages.txt) \
    && rm -rf /var/lib/apt/lists/* /tmp/runtime-packages.txt
COPY --from=build /build/whoosh-lpr /build/libonnxruntime.so.* /opt/whoosh-lpr/
COPY --from=build /build/models /opt/whoosh-lpr/models
RUN /opt/whoosh-lpr/whoosh-lpr --help > /dev/null
WORKDIR /photos
ENTRYPOINT ["/opt/whoosh-lpr/whoosh-lpr"]
CMD ["--help"]
