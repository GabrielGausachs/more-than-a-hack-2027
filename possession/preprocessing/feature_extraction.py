"""Build SkillCorner or Metrica archives in the shared tracking schema."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from games import GAME

from .common_features import schema_document
from .metrica import extract_metrica_tactical
from .skillcorner import (
    DEFAULT_DATA_ROOT, DEFAULT_LABEL_ROOT, discover_matches,
    extract_skillcorner_match,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "derived" / "feature_extracted"


def save_archive(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Atomically write one compressed feature archive."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.stem + ".", suffix=".npz", dir=str(path.parent))
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build(provider: str, match_ids: Optional[Sequence[int]] = None,
          data_root: Optional[Path] = None,
          output_dir: Path = DEFAULT_OUTPUT_DIR,
          label_root: Path = DEFAULT_LABEL_ROOT,
          game: str = GAME) -> Sequence[Path]:
    """Build one provider's archives and the shared schema document."""

    if provider not in ("skillcorner", "metrica"):
        raise ValueError("provider must be 'skillcorner' or 'metrica'")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "schema.json").write_text(
        json.dumps(schema_document(), indent=2) + "\n", encoding="utf-8")

    written = []
    if provider == "skillcorner":
        source_root = Path(data_root) if data_root is not None else DEFAULT_DATA_ROOT
        available = discover_matches(source_root)["local_ids"]
        selected = [int(value) for value in (match_ids or available)]
        for index, match_id in enumerate(selected, 1):
            print("[{}/{}] extracting SkillCorner features for match {}".format(
                index, len(selected), match_id))
            archive = output_dir / "skillcorner" / (str(match_id) + ".npz")
            save_archive(
                archive,
                extract_skillcorner_match(
                    match_id, data_root=source_root, label_root=Path(label_root)),
            )
            written.append(archive)
    else:
        print("[1/1] extracting Metrica features for {} tactical".format(game))
        archive = output_dir / "metrica" / (game + "_tactical.npz")
        save_archive(
            archive,
            extract_metrica_tactical(game=game, data_dir=data_root),
        )
        written.append(archive)
    return written


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("provider", choices=("skillcorner", "metrica"))
    parser.add_argument("--match-ids", nargs="+", type=int)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label-root", type=Path, default=DEFAULT_LABEL_ROOT)
    parser.add_argument("--game", default=GAME)
    args = parser.parse_args(argv)
    build(args.provider, args.match_ids, args.data_root, args.output_dir,
          args.label_root, args.game)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
