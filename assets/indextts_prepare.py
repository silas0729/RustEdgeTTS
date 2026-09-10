"""Aura bridge: prepare the pinned official IndexTTS-2.5 model cache."""

import sys
import threading
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


MODEL_ID = "IndexTeam/IndexTTS-2.5"
AUXILIARY_FILES = {
    "amphion/MaskGCT": {"semantic_codec/model.safetensors"},
    "funasr/campplus": {"campplus_cn_common.bin"},
    "nvidia/bigvgan_v2_22khz_80band_256x": {"config.json", "bigvgan_generator.pt"},
}


def emit(phase: str, current: int = 0, total: int = 0) -> None:
    print(f"AURA_MODEL_PROGRESS\t{phase}\t{current}\t{total}", flush=True)


def tree_size_without_hf_metadata(root: Path) -> int:
    size = 0
    if not root.exists():
        return size
    for path in root.rglob("*"):
        if not path.is_file() or ".cache/huggingface" in path.as_posix():
            continue
        try:
            size += path.stat().st_size
        except OSError:
            pass
    metadata = root / ".cache" / "huggingface" / "download"
    if metadata.exists():
        for path in metadata.rglob("*.incomplete"):
            try:
                size += path.stat().st_size
            except OSError:
                pass
    return size


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: indextts_prepare.py MODEL_DIR")
    model_dir = Path(sys.argv[1]).expanduser().resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    emit("正在读取 IndexTTS-2.5 官方模型清单")
    api = HfApi()
    entries = list(api.list_repo_tree(MODEL_ID, recursive=True, expand=True))
    total = sum(int(getattr(entry, "size", 0) or 0) for entry in entries)
    # Calculate the exact files selected by the pinned official helper so the
    # UI keeps showing aggregate byte progress during auxiliary downloads too.
    w2v_entries = list(api.list_repo_tree("facebook/w2v-bert-2.0", recursive=True, expand=True))
    total += sum(int(getattr(entry, "size", 0) or 0) for entry in w2v_entries)
    for repo_id, required_paths in AUXILIARY_FILES.items():
        entries = api.list_repo_tree(repo_id, recursive=True, expand=True)
        total += sum(
            int(getattr(entry, "size", 0) or 0)
            for entry in entries
            if getattr(entry, "path", None) in required_paths
        )
    stop = threading.Event()

    def monitor(phase: str) -> None:
        while not stop.wait(1.0):
            emit(phase, min(tree_size_without_hf_metadata(model_dir), total), total)

    watcher = threading.Thread(
        target=monitor,
        args=("正在下载 IndexTTS-2.5 官方模型",),
        name="aura-download-progress",
        daemon=True,
    )
    watcher.start()
    try:
        snapshot_download(repo_id=MODEL_ID, local_dir=str(model_dir))
    finally:
        stop.set()
        watcher.join(timeout=1)
    emit("IndexTTS-2.5 主模型下载完成", min(tree_size_without_hf_metadata(model_dir), total), total)

    # The official helper fetches Wav2Vec2-BERT, CAMPPlus, MaskGCT and BigVGAN
    # into the same model directory. Keeping this call in the pinned checkout
    # guarantees that auxiliary revisions match v2.5.0 inference code.
    stop.clear()
    watcher = threading.Thread(
        target=monitor,
        args=("正在下载 IndexTTS-2.5 辅助模型",),
        name="aura-aux-download-progress",
        daemon=True,
    )
    watcher.start()
    emit("正在下载 IndexTTS-2.5 辅助模型", min(tree_size_without_hf_metadata(model_dir), total), total)
    from indextts.utils.model_download import ensure_models_available

    try:
        ensure_models_available(str(model_dir))
    finally:
        stop.set()
        watcher.join(timeout=1)
    emit("IndexTTS-2.5 模型已就绪", total, total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
