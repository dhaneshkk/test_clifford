#!/usr/bin/env sh
# Downloads the detector weights (Apache-2.0, Megvii YOLOX release) and checks the SHA-256.
set -eu
cd "$(dirname "$0")/../models"
URL=https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx
SHA=c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d
[ -f yolox_nano.onnx ] || curl -fsSL -o yolox_nano.onnx "$URL"
echo "$SHA  yolox_nano.onnx" | sha256sum -c -
