#!/usr/bin/env python3

import argparse
import tarfile
from pathlib import Path

from tqdm import tqdm


def extract_shards(shard_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    shards = sorted(shard_dir.glob("shard-*.tar"))
    if not shards:
        raise FileNotFoundError(f"No shard-*.tar files found in {shard_dir}")

    for shard_path in tqdm(shards, desc="Extracting shards"):
        with tarfile.open(shard_path) as tar:
            members = tar.getmembers()
            to_extract = [m for m in members if not (out_dir / m.name).exists()]
            if not to_extract:
                tqdm.write(f"  {shard_path.name}: already extracted, deleting tar")
                shard_path.unlink()
                continue
            for member in tqdm(to_extract, desc=shard_path.name, leave=False):
                tar.extract(member, path=out_dir)

        shard_path.unlink()
        tqdm.write(f"  {shard_path.name}: extracted and deleted")

    print(f"Done. GLBs in: {out_dir}")
    print(f"  Total files: {len(list(out_dir.glob('*.glb')))}")


def main():
    parser = argparse.ArgumentParser(description="Extract mesh shards to a flat directory")
    parser.add_argument("--shard-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    extract_shards(args.shard_dir, args.out_dir)


if __name__ == "__main__":
    main()
