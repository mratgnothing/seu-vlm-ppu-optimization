"""Download complete EEGMMIDB from its public S3 mirror, verifying source hashes."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import time
import urllib.request

SOURCE = "https://physionet.org/files/eegmmidb/1.0.0/"
ARCHIVE = "https://physionet.org/content/eegmmidb/get-zip/1.0.0/"
PAGE = "https://physionet.org/content/eegmmidb/1.0.0/"
S3 = "https://physionet-open.s3.amazonaws.com/eegmmidb/1.0.0/"


def request(url, headers=None, method=None):
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}, method=method), timeout=90)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if str(root).lower() != r"e:\bci\datasets\physionet-eegmmidb-1.0.0":
        raise RuntimeError("Unexpected download destination")
    root.mkdir(parents=True, exist_ok=True)
    meta = root / "metadata"
    parts_dir = root / "download_parts"
    raw_dir = root / "raw"
    for directory in (meta, parts_dir, raw_dir):
        directory.mkdir(exist_ok=True)
    for name, url in (("SHA256SUMS.official.txt", SOURCE + "SHA256SUMS.txt"), ("physionet_dataset_page.html", PAGE)):
        target = meta / name
        if not target.exists():
            with request(url) as response, target.open("xb") as out:
                shutil.copyfileobj(response, out)
    expected = {}
    for line in (meta / "SHA256SUMS.official.txt").read_text().splitlines():
        digest, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        if relative.startswith("./"):
            relative = relative[2:]
        if ".." in PurePosixPath(relative).parts or PurePosixPath(relative).is_absolute():
            raise RuntimeError("Unsafe checksum path")
        expected[relative] = digest
    started = time.monotonic()
    print(f"PUBLIC S3 MIRROR; {len(expected)} official checksum entries", flush=True)
    for subject in range(1, 110):
        (raw_dir / f"S{subject:03d}").mkdir(exist_ok=True)

    def download(relative):
        target = (raw_dir / relative).resolve()
        if not target.is_relative_to(raw_dir):
            raise RuntimeError("Unsafe download target")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha256(target) != expected[relative]:
                raise RuntimeError(f"Existing file hash mismatch: {relative}")
            return target.stat().st_size
        partial = target.with_name(target.name + ".download")
        for attempt in range(5):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                with request(S3 + relative, headers) as response:
                    if offset and response.status != 206:
                        raise RuntimeError("Server did not honor resume range")
                    with partial.open("ab") as out:
                        shutil.copyfileobj(response, out, 256 * 1024)
                if sha256(partial) != expected[relative]:
                    # Only clear an invalid temporary file created by this downloader.
                    partial.unlink()
                    raise RuntimeError("Downloaded file SHA-256 mismatch")
                partial.rename(target)
                return target.stat().st_size
            except Exception as exc:
                print(f"RETRY {relative} attempt {attempt+1}: {type(exc).__name__}: {exc}", flush=True)
                if attempt == 4:
                    raise
                time.sleep(min(2 ** attempt, 10))
    def priority(relative):
        name = PurePosixPath(relative).name
        return (0 if name.endswith(("R04.edf", "R08.edf", "R12.edf")) else 1, relative)
    done_bytes = 0
    failures = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download, name):name for name in sorted(expected, key=priority)}
        for done, future in enumerate(cf.as_completed(futures), 1):
            try:
                done_bytes += future.result()
            except Exception as exc:
                relative = futures[future]
                failures.append(relative)
                print(f"DOWNLOAD ERROR {relative}: {type(exc).__name__}: {exc}", flush=True)
            if done % 50 == 0 or done == len(expected):
                elapsed = time.monotonic() - started
                print(f"DOWNLOAD {done}/{len(expected)} files; {done_bytes/1024**2:.1f} MiB; {done_bytes/1024**2/max(elapsed,1):.2f} MiB/s", flush=True)
    for relative in failures:
        print(f"Sequential recovery: {relative}", flush=True)
        download(relative)
    print("Checking every extracted file against the official SHA-256 manifest...", flush=True)
    verified = []
    for index, (relative, digest) in enumerate(expected.items(), 1):
        path = raw_dir / relative
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"Official SHA-256 mismatch: {relative}")
        verified.append({"path": relative, "bytes": path.stat().st_size, "sha256": digest})
        if index % 400 == 0:
            print(f"HASH {index}/{len(expected)}", flush=True)
    edfs = []
    for path in sorted(raw_dir.glob("S*/*.edf")):
        with path.open("rb") as f:
            fixed = f.read(256)
            if fixed[:8].strip() != b"0":
                raise RuntimeError(f"Invalid EDF header: {path.name}")
            header_bytes = int(fixed[184:192])
            records = int(fixed[236:244])
            duration = float(fixed[244:252])
            channels = int(fixed[252:256])
            rest = f.read(header_bytes - 256)
            sample_start = channels * 216
            samples = [int(rest[sample_start + i*8:sample_start + (i+1)*8]) for i in range(channels)]
            if channels != 65 or records < 0 or header_bytes + records * sum(samples) * 2 != path.stat().st_size:
                raise RuntimeError(f"EDF structural mismatch: {path.name}")
        edfs.append({"subject":path.parent.name,"run":int(path.stem[-2:]),"path":path.relative_to(root).as_posix(),"duration_s":records*duration,"eeg_channels":channels-1,"sampling_hz":samples[0]/duration,"bytes":path.stat().st_size})
    subset = [row for row in edfs if row["run"] in (4, 8, 12)]
    subjects = sorted({row["subject"] for row in edfs})
    if len(edfs) != 109 * 14 or len(subset) != 109 * 3 or len(subjects) != 109:
        raise RuntimeError("Record inventory does not match the official dataset")
    for name, rows in (("all_edf_inventory.csv", edfs), ("left_right_imagery_runs_04_08_12.csv", subset)):
        with (meta / name).open("w", newline="", encoding="utf-8-sig") as out:
            writer = csv.DictWriter(out, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    report = {"dataset":"PhysioNet EEGMMIDB","version":"1.0.0","source":PAGE,"download_source":S3,"completed_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"official_sha256_verified_files":len(verified),"raw_bytes":sum(row["bytes"] for row in verified),"subjects":len(subjects),"edf_files":len(edfs),"edf_structural_checks":"all passed","initial_runs":[4,8,12],"initial_edf_files":len(subset),"initial_edf_bytes":sum(row["bytes"] for row in subset),"events_files":len(list(raw_dir.glob("S*/*.event"))),"sampling_hz_values":sorted({row["sampling_hz"] for row in edfs}),"download_elapsed_s":round(time.monotonic()-started,2),"verified_files":verified}
    (meta / "download_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    readme_source = Path(__file__).with_name("EEGMMIDB_训练说明.md")
    if readme_source.exists():
        shutil.copy2(readme_source, root / "训练说明.md")
    shutil.copy2(Path(__file__), meta / Path(__file__).name)
    # Remove only this script's abandoned numbered archive parts, after verification.
    for index in range(60):
        path = (parts_dir / f"part-{index:03d}.bin").resolve()
        if path.is_relative_to(parts_dir) and path.is_file():
            path.unlink()
    if not any(parts_dir.iterdir()):
        parts_dir.rmdir()
    print("VERIFIED COMPLETE " + json.dumps({key:value for key,value in report.items() if key != "verified_files"}), flush=True)


if __name__ == "__main__":
    main()
