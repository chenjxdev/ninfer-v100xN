"""Materialize the production QAD artifact from its base and replacement tensors."""
import fcntl
import os
import tempfile
from pathlib import Path

from tools.artifact.container import Artifact, ArtifactWriter, ResourceSpec, TensorSpec

MODELS = Path("/data/models")
BASE = MODELS / "qwen3_8_27b_nvfp4.ninfer"
DELTA = MODELS / "qad-q6emb-w8head-q4mtp" / "weights.delta"
NAME = "qwen3_8_27b_q6emb_w8head_q4mtp.ninfer"
SIZE = 18_239_639_552
Q4 = "Q4G64_F16S"
REPLACEMENTS = {
    "text/token_embedding": "Q6G64_F16S",
    "vision/layers/13/attention/qkv": Q4,
    "mtp/layer/attention/query_key_gate_value": Q4,
    "mtp/layer/attention/output": Q4,
    "mtp/layer/mlp/gate_up": Q4,
    "mtp/layer/mlp/down": Q4,
}


def object_spec(obj):
    if obj.kind == "resource":
        return ResourceSpec(obj.name, obj.encoding, obj.bytes)
    return TensorSpec(obj.name, obj.shape, obj.format, obj.layout, obj.bytes)


def source_version(path):
    st = path.stat()
    return f"{st.st_dev}:{st.st_ino}:{st.st_size}:{st.st_mtime_ns}"


def materialize(path):
    with Artifact.open(BASE) as base, Artifact.open(DELTA) as delta:
        if base.identity != delta.identity or base.find("text/output_head").format != "W8G32_F16S":
            raise ValueError("unexpected QAD base or output head")
        if {o.name: o.format for o in delta.objects} != REPLACEMENTS:
            raise ValueError("unexpected tensors in QAD delta")
        sources = [(delta if o.name in REPLACEMENTS else base, o.name) for o in base.objects]
        specs = [object_spec(src.find(name)) for src, name in sources]
        with ArtifactWriter(path, base.identity, specs) as writer:
            for src, name in sources:
                writer.write(name, src.payload(name))
    with Artifact.open(path) as result:
        if result.file_bytes != SIZE or len(result.objects) != 1216:
            raise ValueError("incomplete QAD artifact")
        if any(result.find(name).format != fmt for name, fmt in REPLACEMENTS.items()):
            raise ValueError("incorrect QAD precision")


def main():
    cache = Path("/dev/shm") / f"ninfer-qad-{os.getuid()}"
    cache.mkdir(mode=0o700, exist_ok=True)
    if cache.is_symlink() or cache.stat().st_uid != os.getuid() or cache.stat().st_mode & 0o077:
        raise ValueError(f"expected a private cache directory at {cache}")
    model = cache / NAME
    stamp = cache / "sources"
    with (cache / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        version = "\n".join(map(source_version, (BASE, DELTA, Path(__file__))))
        if model.is_file() and model.stat().st_size == SIZE and stamp.is_file() and stamp.read_text() == version:
            print(model)
            return
        with tempfile.NamedTemporaryFile(dir=cache, delete=False) as file:
            temporary = Path(file.name)
        try:
            materialize(temporary)
            temporary.replace(model)
            stamp.write_text(version)
        finally:
            temporary.unlink(missing_ok=True)
    print(model)


if __name__ == "__main__":
    main()
