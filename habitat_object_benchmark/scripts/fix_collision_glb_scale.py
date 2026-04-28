#!/usr/bin/env python3
"""Copy the scale node transform from each render GLB into its collision and vhacd GLBs.

The render GLBs have the real-world scale baked into node[0].scale.
The collision/vhacd GLBs have no scale (normalized to ~1m).
This script patches the collision/vhacd GLBs in-place to add the same scale.

Usage:
    python3 habitat_object_benchmark/scripts/fix_collision_glb_scale.py \
        --extracted-dir data/datasets/amara-spatial-10k/extracted
"""

import argparse
import json
import struct
from pathlib import Path

from tqdm import tqdm


def read_glb_json(path: Path) -> tuple[dict, bytes]:
    data = path.read_bytes()
    json_length = struct.unpack_from("<I", data, 12)[0]
    gltf = json.loads(data[20:20 + json_length])
    return gltf, data


def write_glb_json(path: Path, gltf: dict, original_data: bytes):
    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    # GLB JSON chunk must be 4-byte aligned, padded with spaces
    while len(json_bytes) % 4 != 0:
        json_bytes += b" "

    original_json_length = struct.unpack_from("<I", original_data, 12)[0]
    # Preserve BIN chunk header + data unchanged (starts right after JSON chunk)
    bin_chunk_start = 20 + original_json_length
    bin_chunk = original_data[bin_chunk_start:] if bin_chunk_start < len(original_data) else b""

    new_length = 12 + 8 + len(json_bytes) + len(bin_chunk)
    header = struct.pack("<III", 0x46546C67, 2, new_length)
    json_chunk_header = struct.pack("<II", len(json_bytes), 0x4E4F534A)

    path.write_bytes(header + json_chunk_header + json_bytes + bin_chunk)


def get_render_scale(render_glb: Path):
    gltf, _ = read_glb_json(render_glb)
    for node in gltf.get("nodes", []):
        if node.get("scale"):
            return node["scale"]
    return None


def patch_glb_scale(glb_path: Path, scale: list[float]) -> bool:
    gltf, data = read_glb_json(glb_path)
    nodes = gltf.get("nodes", [])
    if not nodes:
        return False
    if nodes[0].get("scale") == scale:
        return False  # already correct
    nodes[0]["scale"] = scale
    gltf["nodes"] = nodes
    write_glb_json(glb_path, gltf, data)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extracted-dir", required=True, type=Path)
    args = parser.parse_args()

    render_glbs = sorted(args.extracted_dir.glob("*.glb"))
    render_glbs = [g for g in render_glbs
                   if not g.name.endswith(".collision.glb")
                   and not g.name.endswith(".vhacd.glb")]

    patched, skipped, missing_scale = 0, 0, 0
    for render_glb in tqdm(render_glbs, desc="patching"):
        scale = get_render_scale(render_glb)
        if scale is None:
            missing_scale += 1
            continue

        stem = render_glb.stem
        for suffix in [".collision.glb", ".vhacd.glb"]:
            target = args.extracted_dir / (stem + suffix)
            if not target.exists():
                continue
            if patch_glb_scale(target, scale):
                patched += 1
            else:
                skipped += 1

    print(f"Patched: {patched}  Already correct: {skipped}  No scale in render: {missing_scale}")


if __name__ == "__main__":
    main()
