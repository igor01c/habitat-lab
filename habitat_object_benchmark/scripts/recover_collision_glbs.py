#!/usr/bin/env python3
"""Recover collision/vhacd GLBs corrupted by the missing BIN chunk header bug.

The bug in fix_collision_glb_scale.py wrote:
  [GLB header][JSON chunk header][JSON data][BIN data]   ← missing BIN chunk header

This script adds the 8-byte BIN chunk header back and re-patches the scale.

Usage:
    python3 habitat_object_benchmark/scripts/recover_collision_glbs.py \
        --extracted-dir data/datasets/amara-spatial-10k/extracted
"""

import argparse
import json
import struct
from pathlib import Path

from tqdm import tqdm

BIN_CHUNK_TYPE = 0x004E4942  # "BIN\x00"
JSON_CHUNK_TYPE = 0x4E4F534A  # "JSON"


def read_glb_json(data: bytes) -> dict:
    json_length = struct.unpack_from("<I", data, 12)[0]
    return json.loads(data[20:20 + json_length])


def get_render_scale(render_glb: Path):
    data = render_glb.read_bytes()
    gltf = read_glb_json(data)
    for node in gltf.get("nodes", []):
        if node.get("scale"):
            return node["scale"]
    return None


def is_corrupted(data: bytes) -> bool:
    """Check if BIN chunk header is missing (our specific corruption)."""
    json_length = struct.unpack_from("<I", data, 12)[0]
    bin_header_start = 20 + json_length
    if bin_header_start + 8 > len(data):
        return False  # no BIN chunk at all — not corrupted, just JSON-only
    chunk_type = struct.unpack_from("<I", data, bin_header_start + 4)[0]
    return chunk_type != BIN_CHUNK_TYPE


def recover_and_patch(glb_path: Path, scale: list):
    data = glb_path.read_bytes()

    if not is_corrupted(data):
        return False  # file is fine, skip

    # Read current JSON chunk
    json_length = struct.unpack_from("<I", data, 12)[0]
    gltf = json.loads(data[20:20 + json_length])

    # Patch scale into node[0]
    nodes = gltf.get("nodes", [])
    if not nodes:
        return False
    nodes[0]["scale"] = scale
    gltf["nodes"] = nodes

    # Serialize new JSON, 4-byte aligned
    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    while len(json_bytes) % 4 != 0:
        json_bytes += b" "

    # The corrupted file has BIN data starting right after JSON (no BIN header)
    bin_data = data[20 + json_length:]

    # Reconstruct proper GLB with BIN chunk header restored
    bin_chunk_header = struct.pack("<II", len(bin_data), BIN_CHUNK_TYPE)
    new_total = 12 + 8 + len(json_bytes) + 8 + len(bin_data)
    glb_header = struct.pack("<III", 0x46546C67, 2, new_total)
    json_chunk_header = struct.pack("<II", len(json_bytes), JSON_CHUNK_TYPE)

    glb_path.write_bytes(glb_header + json_chunk_header + json_bytes +
                         bin_chunk_header + bin_data)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extracted-dir", required=True, type=Path)
    args = parser.parse_args()

    render_glbs = sorted(args.extracted_dir.glob("*.glb"))
    render_glbs = [g for g in render_glbs
                   if not g.name.endswith(".collision.glb")
                   and not g.name.endswith(".vhacd.glb")]

    recovered, skipped, missing_scale = 0, 0, 0
    for render_glb in tqdm(render_glbs, desc="recovering"):
        scale = get_render_scale(render_glb)
        if scale is None:
            missing_scale += 1
            continue

        stem = render_glb.stem
        for suffix in [".collision.glb", ".vhacd.glb"]:
            target = args.extracted_dir / (stem + suffix)
            if not target.exists():
                continue
            if recover_and_patch(target, scale):
                recovered += 1
            else:
                skipped += 1

    print(f"Recovered+patched: {recovered}  Skipped: {skipped}  No scale in render: {missing_scale}")


if __name__ == "__main__":
    main()
