"""JSON batch bridge for the pinned official IndexTTS-2.5 implementation."""

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: indextts_bridge.py MODEL_DIR MANIFEST_JSON")
    model_dir = Path(sys.argv[1]).expanduser().resolve()
    manifest_path = Path(sys.argv[2]).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    print("AURA_INFERENCE\t正在加载 IndexTTS-2.5", flush=True)
    from indextts.infer_v2_5 import IndexTTS2

    tts = IndexTTS2(
        cfg_path=str(model_dir / "config.yaml"),
        model_dir=str(model_dir),
        use_bf16=False,
        use_cuda_kernel=False,
        use_deepspeed=False,
        use_accel=False,
        use_torch_compile=False,
        use_qwen_emo=False,
    )
    items = manifest["items"]
    for index, item in enumerate(items, start=1):
        tts.infer(
            spk_audio_prompt=manifest["reference_audio"],
            text=item["text"],
            lang=item["language"],
            output_path=item["output"],
            duration_factor=manifest["duration_factor"],
            verbose=False,
        )
        print(f"AURA_INFERENCE_PROGRESS\t{index}\t{len(items)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
