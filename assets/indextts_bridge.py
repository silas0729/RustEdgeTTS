"""JSON batch bridge for the pinned official IndexTTS-2.5 implementation."""

import json
import sys
from pathlib import Path


def patch_sampling_switch(source_dir: Path) -> None:
    """Make the v2.5.0 sampling flag reach the official GPT generator.

    The pinned release parses ``do_sample`` but hard-codes ``True`` at the
    final ``inference_speech`` call. Patch that one call in the local cache;
    newer releases that already use the variable are left untouched.
    """
    source_path = source_dir / "indextts" / "infer_v2_5.py"
    try:
        source = source_path.read_text(encoding="utf-8")
    except OSError:
        return
    needle = "                        do_sample=True,\n"
    if needle not in source or "do_sample=do_sample" in source:
        return
    try:
        source_path.write_text(source.replace(needle, "                        do_sample=do_sample,\n", 1), encoding="utf-8")
    except OSError:
        # In a read-only installation the official default remains usable;
        # only the optional deterministic switch cannot be applied.
        return


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: indextts_bridge.py MODEL_DIR MANIFEST_JSON")
    model_dir = Path(sys.argv[1]).expanduser().resolve()
    manifest_path = Path(sys.argv[2]).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    print("AURA_INFERENCE\t正在加载 IndexTTS-2.5", flush=True)
    source_root = model_dir.parent / "official-v2.5.0"
    patch_sampling_switch(source_root)
    from indextts.infer_v2_5 import IndexTTS2

    tts = IndexTTS2(
        cfg_path=str(model_dir / "config.yaml"),
        model_dir=str(model_dir),
        use_bf16=False,
        use_cuda_kernel=False,
        use_deepspeed=False,
        use_accel=False,
        use_torch_compile=False,
        # Load the optional emotion model only when emotion-text guidance is
        # enabled. The required Qwen emotion checkpoint is part of the
        # validated offline model tree.
        use_qwen_emo=bool(manifest.get("use_emo_text", False)),
    )
    items = manifest["items"]
    generation = manifest.get("generation", {})
    for index, item in enumerate(items, start=1):
        tts.infer(
            spk_audio_prompt=manifest["reference_audio"],
            text=item["text"],
            lang=item["language"],
            output_path=item["output"],
            duration_factor=manifest["duration_factor"],
            text_normalization=manifest.get("text_normalization", True),
            max_text_tokens_per_segment=manifest.get("max_text_tokens_per_segment", 120),
            interval_silence=manifest.get("interval_silence_ms", 200),
            use_random=manifest.get("use_random", False),
            emo_alpha=manifest.get("emo_alpha", 1.0),
            use_emo_text=manifest.get("use_emo_text", False),
            emo_text=manifest.get("emo_text") or None,
            do_sample=generation.get("do_sample", True),
            temperature=generation.get("temperature", 0.8),
            top_k=generation.get("top_k", 30),
            top_p=generation.get("top_p", 0.8),
            repetition_penalty=generation.get("repetition_penalty", 10.0),
            length_penalty=generation.get("length_penalty", 0.0),
            num_beams=generation.get("num_beams", 3),
            max_mel_tokens=generation.get("max_mel_tokens", 1500),
            verbose=False,
        )
        print(f"AURA_INFERENCE_PROGRESS\t{index}\t{len(items)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
