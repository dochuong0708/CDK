"""Dynamically quantize an exported text-classification ONNX model to INT8."""

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("onnx_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    try:
        from optimum.onnxruntime import ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
    except ImportError as error:
        raise SystemExit("Install optimum[onnxruntime] to quantize ONNX models.") from error

    model_path = next(
        path for path in (args.onnx_dir / "model.onnx", args.onnx_dir / "model" / "model.onnx")
        if path.exists()
    )
    model_dir = model_path.parent
    args.output_dir.mkdir(parents=True, exist_ok=True)
    quantizer = ORTQuantizer.from_pretrained(model_dir, file_name=model_path.name)
    quantizer.quantize(
        save_dir=args.output_dir,
        quantization_config=AutoQuantizationConfig.avx2(is_static=False, per_channel=True),
    )
    for path in model_dir.iterdir():
        if path.is_file() and path.suffix != ".onnx":
            shutil.copy2(path, args.output_dir / path.name)
    print(f"Quantized model saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()