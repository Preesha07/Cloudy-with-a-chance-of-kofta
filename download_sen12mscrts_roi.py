#!/usr/bin/env python3

"""Download a region-specific SEN12MS-CR-TS shard set."""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path


REGIONS = ["africa", "america", "asiaEast", "asiaWest", "europa"]
TRAIN_BASE = "https://dataserv.ub.tum.de/s/m1639953/download?path=/&files="
TEST_BASE = "https://dataserv.ub.tum.de/s/m1659251/download?path=/&files="
MONO_BASE = "https://dataserv.ub.tum.de/s/m1554803/download?path=/&files="
ARCHIVE_SIZES = {
    "s2_africa.tar.gz": 98233900,
    "s2_america.tar.gz": 110245004,
    "s2_asiaEast.tar.gz": 113948560,
    "s2_asiaWest.tar.gz": 96082796,
    "s2_europa.tar.gz": 196669740,
    "s2_africa_test.tar.gz": 25421744,
    "s2_america_test.tar.gz": 25421824,
    "s2_asiaEast_test.tar.gz": 40534760,
    "s2_asiaWest_test.tar.gz": 15012924,
    "s2_europa_test.tar.gz": 79568460,
    "s1_africa.tar.gz": 60544524,
    "s1_america.tar.gz": 67947416,
    "s1_asiaEast.tar.gz": 70230104,
    "s1_asiaWest.tar.gz": 59218848,
    "s1_europa.tar.gz": 121213836,
    "s1_africa_test.tar.gz": 15668120,
    "s1_america_test.tar.gz": 15668160,
    "s1_asiaEast_test.tar.gz": 24982736,
    "s1_asiaWest_test.tar.gz": 9252904,
    "s1_europa_test.tar.gz": 49040432,
    "ROIs1158_spring_s2.tar.gz": 48568904,
    "ROIs1868_summer_s2.tar.gz": 56425520,
    "ROIs1970_fall_s2.tar.gz": 68291864,
    "ROIs2017_winter_s2.tar.gz": 30580552,
    "ROIs1158_spring_s2_cloudy.tar.gz": 48569368,
    "ROIs1868_summer_s2_cloudy.tar.gz": 56426004,
    "ROIs1970_fall_s2_cloudy.tar.gz": 68292448,
    "ROIs2017_winter_s2_cloudy.tar.gz": 30580812,
    "ROIs1158_spring_s1.tar.gz": 15026120,
    "ROIs1868_summer_s1.tar.gz": 17456784,
    "ROIs1970_fall_s1.tar.gz": 21127832,
    "ROIs2017_winter_s1.tar.gz": 9460956,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download SEN12MS-CR (mono-temporal) by default, or SEN12MS-CR-TS when requested."
    )
    parser.add_argument(
        "--dataset",
        choices=["mono", "ts"],
        default="mono",
        help="Choose mono-temporal SEN12MS-CR or time-series SEN12MS-CR-TS.",
    )
    parser.add_argument(
        "--region",
        choices=["all"] + REGIONS,
        default="asiaEast",
        help="ROI region to download in TS mode. AsiaEast is the India-side shard.",
    )
    parser.add_argument(
        "--include-s1",
        action="store_true",
        help="Also download the Sentinel-1 archives for the chosen dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "SEN12MSCR"),
        help="Directory that will contain the merged ROI structure.",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Directory used to stage partial downloads. Defaults to a persistent folder next to the output.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download the tar.gz archives and keep them on disk for later extraction.",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only extract previously downloaded archives from the work-dir archives folder.",
    )
    parser.add_argument(
        "--archives",
        nargs="+",
        default=None,
        help="Explicit archive filenames to fetch (for example: ROIs1158_spring_s1.tar.gz).",
    )
    return parser.parse_args()


def selected_regions(region: str) -> list[str]:
    return REGIONS if region == "all" else [region]


def archives_to_download(dataset: str, region: str, include_s1: bool) -> list[tuple[str, str]]:
    archives: list[tuple[str, str]] = []
    if dataset == "mono":
        mono_archives = [
            ("S2 spring clear", f"{MONO_BASE}ROIs1158_spring_s2.tar.gz"),
            ("S2 summer clear", f"{MONO_BASE}ROIs1868_summer_s2.tar.gz"),
            ("S2 fall clear", f"{MONO_BASE}ROIs1970_fall_s2.tar.gz"),
            ("S2 winter clear", f"{MONO_BASE}ROIs2017_winter_s2.tar.gz"),
            ("S2 spring cloudy", f"{MONO_BASE}ROIs1158_spring_s2_cloudy.tar.gz"),
            ("S2 summer cloudy", f"{MONO_BASE}ROIs1868_summer_s2_cloudy.tar.gz"),
            ("S2 fall cloudy", f"{MONO_BASE}ROIs1970_fall_s2_cloudy.tar.gz"),
            ("S2 winter cloudy", f"{MONO_BASE}ROIs2017_winter_s2_cloudy.tar.gz"),
        ]
        archives.extend(mono_archives)
        if include_s1:
            archives.extend([
                ("S1 spring", f"{MONO_BASE}ROIs1158_spring_s1.tar.gz"),
                ("S1 summer", f"{MONO_BASE}ROIs1868_summer_s1.tar.gz"),
                ("S1 fall", f"{MONO_BASE}ROIs1970_fall_s1.tar.gz"),
                ("S1 winter", f"{MONO_BASE}ROIs2017_winter_s1.tar.gz"),
            ])
        return archives

    for split in ["train", "test"]:
        base = TRAIN_BASE if split == "train" else TEST_BASE
        suffix = "" if split == "train" else "_test"
        for selected_region in selected_regions(region):
            archives.append((f"S2 {split} {selected_region}", f"{base}s2_{selected_region}{suffix}.tar.gz"))
            if include_s1:
                archives.append((f"S1 {split} {selected_region}", f"{base}s1_{selected_region}{suffix}.tar.gz"))
    return archives


def archive_size_bytes(url: str) -> int:
    return ARCHIVE_SIZES[Path(url.split("files=")[-1]).name]


def resolve_custom_archives(archive_names: list[str]) -> list[tuple[str, str]]:
    archives: list[tuple[str, str]] = []
    for name in archive_names:
        if name not in ARCHIVE_SIZES:
            raise ValueError(f"Unsupported archive name: {name}")

        if name.startswith("ROIs"):
            url = f"{MONO_BASE}{name}"
        elif "_test" in name:
            url = f"{TEST_BASE}{name}"
        else:
            url = f"{TRAIN_BASE}{name}"
        archives.append((name, url))
    return archives


def format_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    units = ["KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    for unit in units:
        value /= 1024.0
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
    return f"{value:.2f} TiB"


def print_estimate(archives: list[tuple[str, str]]) -> None:
    total_bytes = sum(archive_size_bytes(url) for _, url in archives)
    print(f"Selected archives: {len(archives)}")
    print(f"Estimated download size: {format_size(total_bytes)} ({total_bytes} bytes)")


class ProgressReader:
    def __init__(self, response, total_bytes: int | None, label: str):
        self.response = response
        self.total_bytes = total_bytes
        self.label = label
        self.downloaded = 0

    def read(self, size: int = -1) -> bytes:
        chunk = self.response.read(size)
        if chunk:
            self.downloaded += len(chunk)
            if self.total_bytes:
                percent = self.downloaded * 100.0 / self.total_bytes
                print(
                    f"\r  {self.label}: {format_size(self.downloaded)} / {format_size(self.total_bytes)} ({percent:5.1f}%)",
                    end="",
                    flush=True,
                )
            else:
                print(f"\r  {self.label}: {format_size(self.downloaded)} downloaded", end="", flush=True)
        else:
            print()
        return chunk


def download_archive(url: str, destination: Path, label: str) -> None:
    def reporthook(blocknum: int, blocksize: int, totalsize: int) -> None:
        downloaded = blocknum * blocksize
        total = totalsize
        if total:
            downloaded = min(downloaded, total)
            percent = downloaded * 100.0 / total
            print(
                f"\r  {label}: {format_size(downloaded)} / {format_size(total)} ({percent:5.1f}%)",
                end="",
                flush=True,
            )
        else:
            print(f"\r  {label}: {format_size(downloaded)} downloaded", end="", flush=True)

    urllib.request.urlretrieve(url, destination, reporthook=reporthook)
    print()


def safe_member_path(base_dir: Path, member_name: str) -> Path:
    member_path = (base_dir / member_name).resolve()
    base = base_dir.resolve()
    if base not in member_path.parents and member_path != base:
        raise RuntimeError(f"Unsafe archive member path: {member_name}")
    return member_path


def stream_extract_archive(url: str, extract_dir: Path, label: str) -> None:
    with urllib.request.urlopen(url) as response:
        total = response.headers.get("Content-Length")
        total_bytes = int(total) if total is not None else None
        reader = ProgressReader(response, total_bytes, label)

        with tarfile.open(fileobj=reader, mode="r|gz") as archive:
            for member in archive:
                target_path = safe_member_path(extract_dir, member.name)
                if member.isdir():
                    target_path.mkdir(parents=True, exist_ok=True)
                    continue
                if member.isfile():
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        raise RuntimeError(f"Could not extract {member.name}")
                    with target_path.open("wb") as handle:
                        shutil.copyfileobj(extracted, handle)
                    continue
                if member.issym() or member.islnk():
                    raise RuntimeError(f"Unsupported archive entry: {member.name}")


def merge_path(source: Path, target: Path) -> None:
    if not target.exists():
        shutil.move(str(source), str(target))
        return

    if source.is_dir() and target.is_dir():
        for child in source.iterdir():
            merge_path(child, target / child.name)
        if not any(source.iterdir()):
            source.rmdir()
        return

    if source.is_file() and target.is_file():
        target.unlink()
        shutil.move(str(source), str(target))
        return

    raise RuntimeError(f"Cannot merge {source} into {target}")


def merge_extracted_tree(extracted_root: Path, output_dir: Path) -> None:
    top_level = [item for item in extracted_root.iterdir() if item.name not in {"."}]
    flatten_one_level = not any(item.is_dir() and item.name.startswith("ROIs") for item in top_level)

    if flatten_one_level:
        for item in top_level:
            if item.is_dir():
                for child in item.iterdir():
                    merge_path(child, output_dir / child.name)
                if not any(item.iterdir()):
                    item.rmdir()
            else:
                merge_path(item, output_dir / item.name)
        return

    for item in top_level:
        merge_path(item, output_dir / item.name)


def process_archive(
    label: str,
    url: str,
    output_dir: Path,
    work_dir: Path,
    download_only: bool,
    extract_only: bool,
) -> None:
    archive_name = Path(url.split("files=")[-1]).name
    archive_path = work_dir / "archives" / archive_name
    extract_dir = work_dir / archive_name.replace(".tar.gz", "")
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    if not extract_only:
        print(f"Downloading {label}: {archive_name}")
        download_archive(url, archive_path, label)

        if download_only:
            print(f"Saved {label} to {archive_path}")
            return
    else:
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive not found for extraction-only mode: {archive_path}")

    extract_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {label}")
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive:
            target_path = safe_member_path(extract_dir, member.name)
            if member.isdir():
                target_path.mkdir(parents=True, exist_ok=True)
                continue
            if member.isfile():
                target_path.parent.mkdir(parents=True, exist_ok=True)
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise RuntimeError(f"Could not extract {member.name}")
                with target_path.open("wb") as handle:
                    shutil.copyfileobj(extracted, handle)
                continue
            if member.issym() or member.islnk():
                raise RuntimeError(f"Unsupported archive entry: {member.name}")

    print(f"Merging {label} into {output_dir}")
    merge_extracted_tree(extract_dir, output_dir)


def main() -> int:
    args = parse_args()
    if args.download_only and args.extract_only:
        print("Use only one of --download-only or --extract-only.", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    work_dir = Path(args.work_dir).resolve() if args.work_dir else output_dir.parent / f".{output_dir.name}_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"Staging partial files in: {work_dir}")

    if args.archives:
        archives = resolve_custom_archives(args.archives)
    else:
        archives = archives_to_download(args.dataset, args.region, args.include_s1)
    if not archives:
        print("No archives selected.", file=sys.stderr)
        return 1

    if args.dataset == "mono" and args.region != "asiaEast":
        print("Note: region is ignored in mono-temporal mode.")

    print_estimate(archives)

    for label, url in archives:
        process_archive(label, url, output_dir, work_dir, args.download_only, args.extract_only)

    print(f"Done. ROI data is in: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())