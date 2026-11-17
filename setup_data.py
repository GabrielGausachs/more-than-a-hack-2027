"""Download and install the official Metrica competition data."""

from __future__ import annotations

import argparse
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import BinaryIO, Dict, Optional, Sequence

from games import asset_paths


ROOT = Path(__file__).resolve().parent
DRIVE_URL = "https://drive.google.com/drive/folders/18KTDrUQX4sZXFfWnRcz09-FTehburY_K"
DEFAULT_DATA_DIR = ROOT / "data" / "metrica"
ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")


def _required_files(data_dir: Path) -> Dict[str, Path]:
    """Expected non-video competition filename to its destination."""

    return {
        Path(path).name: Path(path)
        for source in ("tactical", "broadcast")
        for asset, path in asset_paths(data_dir=str(data_dir), source=source).items()
        if asset != "video"
    }


def _present(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _is_archive(name: str) -> bool:
    return name.lower().endswith(ARCHIVE_SUFFIXES)


def _install_stream(source: BinaryIO, target: Path) -> None:
    """Atomically install one non-empty asset."""

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part")
    with temporary.open("wb") as output:
        shutil.copyfileobj(source, output, length=1024 * 1024)
    if not _present(temporary):
        temporary.unlink(missing_ok=True)
        raise ValueError("empty asset: {}".format(target.name))
    temporary.replace(target)
    print("Ready: {}".format(target), flush=True)


def _install_bundle(bundle: Path, required: Dict[str, Path]) -> None:
    """Install expected assets from a folder, ZIP or TAR without extract-all."""

    def target_for(name: str) -> Optional[Path]:
        target = required.get(Path(name).name)
        return target if target is not None and not _present(target) else None

    if bundle.is_dir():
        paths = sorted(path for path in bundle.rglob("*") if path.is_file())
        for path in paths:
            target = target_for(path.name)
            if target is not None:
                with path.open("rb") as source:
                    _install_stream(source, target)
        for path in paths:
            if _is_archive(path.name):
                _install_bundle(path, required)
        return

    if zipfile.is_zipfile(bundle):
        with zipfile.ZipFile(bundle) as archive:
            for member in archive.infolist():
                target = None if member.is_dir() else target_for(member.filename)
                if target is not None:
                    with archive.open(member) as source:
                        _install_stream(source, target)
        return

    if tarfile.is_tarfile(bundle):
        with tarfile.open(bundle) as archive:
            for member in archive:
                target = target_for(member.name) if member.isfile() else None
                if target is not None:
                    source = archive.extractfile(member)
                    if source is None:
                        raise ValueError("cannot read archive member: {}".format(member.name))
                    with source:
                        _install_stream(source, target)
        return

    raise ValueError("expected a folder, ZIP or TAR bundle: {}".format(bundle))


def _download_drive(data_dir: Path, required: Dict[str, Path], url: str) -> None:
    try:
        import gdown
    except ImportError as error:
        raise RuntimeError("install dependencies with: python -m pip install -r requirements.txt") from error

    staging = data_dir / ".downloads"
    staging.mkdir(parents=True, exist_ok=True)
    try:
        entries = gdown.download_folder(
            url=url, output=str(staging), skip_download=True, quiet=True)
        if not entries:
            raise RuntimeError("Google Drive returned no downloadable files")

        for entry in entries:
            name = Path(entry.path).name
            target = required.get(name)
            if target is not None and not _present(target):
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_name(target.name + ".part")
                if not gdown.download(id=entry.id, output=str(partial), resume=True):
                    raise RuntimeError("download failed: {}".format(name))
                if not _present(partial):
                    raise RuntimeError("empty download: {}".format(name))
                partial.replace(target)
                print("Ready: {}".format(target), flush=True)
            elif _is_archive(name) and any(not _present(path) for path in required.values()):
                archive = staging / (entry.id + "-" + name)
                if not _present(archive):
                    partial = archive.with_name(archive.name + ".part")
                    if not gdown.download(id=entry.id, output=str(partial), resume=True):
                        raise RuntimeError("download failed: {}".format(name))
                    if not _present(partial):
                        raise RuntimeError("empty download: {}".format(name))
                    partial.replace(archive)
                _install_bundle(archive, required)
    except Exception as error:
        raise RuntimeError(
            "cannot download the competition data: {}\n"
            "If Google Drive requires sign-in, download the bundle in your browser and run "
            "this command again with --bundle PATH.".format(error)
        ) from error


def download(data_dir: Path = DEFAULT_DATA_DIR, drive_url: str = DRIVE_URL,
             bundle: Optional[Path] = None) -> None:
    data_dir = data_dir.expanduser().resolve()
    required = _required_files(data_dir)
    if bundle is not None:
        _install_bundle(bundle.expanduser().resolve(), required)
    elif any(not _present(path) for path in required.values()):
        _download_drive(data_dir, required, drive_url)

    missing = [str(path) for path in required.values() if not _present(path)]
    if missing:
        raise RuntimeError("competition data is incomplete; missing:\n  " + "\n  ".join(missing))
    print("Metrica non-video data are ready in {}".format(data_dir))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--drive-url", default=DRIVE_URL)
    parser.add_argument("--bundle", type=Path,
                        help="install an already downloaded ZIP/TAR file or extracted folder")
    args = parser.parse_args(argv)
    try:
        download(args.data_dir, args.drive_url, args.bundle)
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, "Data setup failed: {}\n".format(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
