"""Batch-convert PDFs under a directory to Markdown via the MinerU parse API.

Docs: https://mineru.net/apiManage/docs

Behavior:
  1. Recursively find all .pdf files under <target_dir>
  2. Skip a PDF whose same-named output dir already contains full.md
     (unless --force)
  3. Submit the rest in batches: signed URL -> PUT upload -> poll -> download zip
  4. Extract into a sibling folder named after the source
     (e.g. foo/bar.pdf -> foo/bar/)
  5. Rename zip members <uuid>_content_list.json / _model.json / _origin.pdf
     to <basename>_content_list.json / _model.json / _origin.pdf
  6. Show overall progress with tqdm; stop immediately on quota / auth failure

Example:
  export MINERU_TOKEN="xxx"
  python3 corpus_scripts/mineru_parse.py quantum_physics --model vlm --language en
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import requests
from tqdm import tqdm

API_BASE = "https://mineru.net/api/v4"
SUBMIT_FILE_BATCH = f"{API_BASE}/file-urls/batch"
QUERY_BATCH = f"{API_BASE}/extract-results/batch/{{batch_id}}"

# Hard API limit: at most 50 files per batch
MAX_BATCH_SIZE = 50

# Daily quota exhausted / HTML quota exhausted
QUOTA_ERROR_CODES = {-60018, -60019}
# Bad or expired token
AUTH_ERROR_CODES = {"A0202", "A0211"}

RENAME_SUFFIXES = ("_content_list.json", "_model.json", "_origin.pdf")
DONE_MARKER = "full.md"


# ---------------- exceptions ----------------


class MineruError(RuntimeError):
    def __init__(self, code: Any, msg: str) -> None:
        super().__init__(f"MinerU API error code={code} msg={msg}")
        self.code = code
        self.msg = msg


class QuotaExhaustedError(MineruError):
    """Daily parse quota exhausted (-60018 / -60019)."""


class AuthError(MineruError):
    """Bad or expired token (A0202 / A0211)."""


# ---------------- HTTP helpers ----------------


def _auth_headers(token: str) -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }


def _check_response(body: dict[str, Any]) -> None:
    code = body.get("code")
    if code == 0:
        return
    msg = body.get("msg", "")
    if code in QUOTA_ERROR_CODES:
        raise QuotaExhaustedError(code, msg)
    if str(code) in AUTH_ERROR_CODES:
        raise AuthError(code, msg)
    raise MineruError(code, msg)


# ---------------- file discovery ----------------


def find_pdfs(root: Path) -> list[Path]:
    """Recursively find PDFs, skipping files that already live in a MinerU output dir.

    A previous run writes <basename>_origin.pdf inside <basename>/. If those
    were submitted again they would nest as <basename>_origin/<basename>_origin_origin.pdf.
    Treat a PDF as in-output if its parent directory contains DONE_MARKER (full.md).
    """
    pdfs: list[Path] = []
    for p in root.rglob("*.pdf"):
        if not p.is_file():
            continue
        if (p.parent / DONE_MARKER).is_file():
            continue
        pdfs.append(p)
    return sorted(pdfs)


def output_dir_for(pdf: Path) -> Path:
    return pdf.parent / pdf.stem


def already_done(pdf: Path) -> bool:
    out = output_dir_for(pdf)
    return out.is_dir() and (out / DONE_MARKER).is_file()


# ---------------- API calls ----------------


def submit_batch(
    token: str,
    pdfs: list[Path],
    *,
    model_version: str,
    language: str,
    is_ocr: bool,
    enable_formula: bool,
    enable_table: bool,
    page_ranges: str | None,
    extra_formats: list[str] | None,
) -> tuple[str, list[str], list[str]]:
    """Submit one batch. Returns (batch_id, upload_urls, data_ids).

    data_id is 1:1 with pdfs so results can be matched even if file names collide.
    """
    files: list[dict[str, Any]] = []
    data_ids: list[str] = []
    for i, pdf in enumerate(pdfs):
        data_id = f"f{i:03d}"
        data_ids.append(data_id)
        entry: dict[str, Any] = {
            "name": pdf.name,
            "is_ocr": is_ocr,
            "data_id": data_id,
        }
        if page_ranges:
            entry["page_ranges"] = page_ranges
        files.append(entry)

    payload: dict[str, Any] = {
        "files": files,
        "model_version": model_version,
        "enable_formula": enable_formula,
        "enable_table": enable_table,
        "language": language,
    }
    if extra_formats:
        payload["extra_formats"] = extra_formats

    resp = requests.post(SUBMIT_FILE_BATCH, headers=_auth_headers(token), json=payload, timeout=30)
    resp.raise_for_status()
    body = resp.json()
    _check_response(body)

    data = body["data"]
    upload_urls: list[str] = data.get("file_urls") or []
    if len(upload_urls) != len(pdfs):
        raise MineruError(
            "client",
            f"upload url count mismatch: got {len(upload_urls)}, expected {len(pdfs)}",
        )
    return data["batch_id"], upload_urls, data_ids


def upload_file(local_path: Path, upload_url: str) -> None:
    """PUT the file to OSS. Do not send a Content-Type header."""
    with local_path.open("rb") as f:
        resp = requests.put(upload_url, data=f, timeout=600)
    if resp.status_code not in (200, 201):
        raise RuntimeError(
            f"upload failed for {local_path.name}: HTTP {resp.status_code} body={resp.text[:300]}"
        )


def query_batch(token: str, batch_id: str) -> list[dict[str, Any]]:
    resp = requests.get(QUERY_BATCH.format(batch_id=batch_id), headers=_auth_headers(token), timeout=30)
    resp.raise_for_status()
    body = resp.json()
    _check_response(body)
    return body["data"].get("extract_result", [])


def download_zip(url: str) -> bytes:
    resp = requests.get(url, timeout=600)
    resp.raise_for_status()
    return resp.content


# ---------------- result extraction ----------------


def extract_and_rename(zip_bytes: bytes, out_dir: Path, basename: str) -> None:
    """Extract into out_dir and rename <uuid>_xxx to <basename>_xxx."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        zf.extractall(out_dir)

    for f in out_dir.iterdir():
        if not f.is_file():
            continue
        for suffix in RENAME_SUFFIXES:
            if f.name.endswith(suffix) and f.name != f"{basename}{suffix}":
                target = out_dir / f"{basename}{suffix}"
                if target.exists():
                    target.unlink()
                f.rename(target)
                break


# ---------------- batch processing ----------------


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def process_batch(
    token: str,
    pdfs: list[Path],
    *,
    options: dict[str, Any],
    poll_interval: int,
    batch_timeout: int,
    pbar: tqdm,
    root: Path,
) -> Counter:
    """Submit, upload, poll, and extract one batch. Returns {state: count}."""
    stats: Counter = Counter()

    batch_id, upload_urls, data_ids = submit_batch(token, pdfs, **options)
    id_to_pdf = dict(zip(data_ids, pdfs))
    tqdm.write(f"[batch] submitted {len(pdfs)} files, batch_id={batch_id}")

    pbar.set_postfix_str(f"uploading 0/{len(pdfs)}")
    for i, (pdf, url) in enumerate(zip(pdfs, upload_urls), 1):
        try:
            upload_file(pdf, url)
        except Exception as e:
            tqdm.write(f"[upload-fail] {_rel(pdf, root)}: {e}")
            stats["upload_failed"] += 1
        pbar.set_postfix_str(f"uploading {i}/{len(pdfs)}")

    completed_ids: set[str] = set()
    last_state_log = ""
    start = time.time()

    while len(completed_ids) < len(pdfs):
        try:
            results = query_batch(token, batch_id)
        except (QuotaExhaustedError, AuthError):
            raise
        except Exception as e:
            tqdm.write(f"[poll-warn] {e}; retry in {poll_interval}s")
            time.sleep(poll_interval)
            continue

        state_counts: Counter = Counter()
        for r in results:
            data_id = r.get("data_id")
            state = r.get("state", "?")
            state_counts[state] += 1

            if data_id in completed_ids or data_id not in id_to_pdf:
                continue

            pdf = id_to_pdf[data_id]
            rel = _rel(pdf, root)

            if state == "done":
                zip_url = r.get("full_zip_url")
                if not zip_url:
                    tqdm.write(f"[no-zip] {rel}: state=done but no full_zip_url")
                    stats["done_no_zip"] += 1
                else:
                    try:
                        zip_bytes = download_zip(zip_url)
                        extract_and_rename(zip_bytes, output_dir_for(pdf), pdf.stem)
                        tqdm.write(f"[done] {rel}")
                        stats["done"] += 1
                    except Exception as e:
                        tqdm.write(f"[extract-fail] {rel}: {e}")
                        stats["extract_failed"] += 1
                completed_ids.add(data_id)
                pbar.update(1)
            elif state == "failed":
                err_code = r.get("err_code", "")
                err_msg = r.get("err_msg", "")
                tqdm.write(f"[failed] {rel}: code={err_code} msg={err_msg}")
                stats["failed"] += 1
                completed_ids.add(data_id)
                pbar.update(1)

        state_log = ", ".join(f"{k}={v}" for k, v in sorted(state_counts.items()))
        if state_log != last_state_log:
            pbar.set_postfix_str(f"batch[{state_log}]")
            last_state_log = state_log

        if len(completed_ids) >= len(pdfs):
            break
        if time.time() - start > batch_timeout:
            tqdm.write(
                f"[timeout] batch {batch_id} exceeded {batch_timeout}s, "
                f"completed {len(completed_ids)}/{len(pdfs)}"
            )
            stats["timeout"] += len(pdfs) - len(completed_ids)
            for did in id_to_pdf:
                if did not in completed_ids:
                    pbar.update(1)
            break

        time.sleep(poll_interval)

    return stats


# ---------------- CLI ----------------


def _chunked(seq: list, size: int) -> Iterable[list]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert every PDF under a directory to Markdown via the MinerU API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("target", help="Target directory, e.g. quantum_physics")
    parser.add_argument(
        "--model",
        default="vlm",
        choices=["pipeline", "vlm"],
        help="Parse model version",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="OCR language (used on the OCR path only)",
    )
    parser.add_argument("--page-ranges", default=None, help="Page ranges, e.g. '1-5,8'")
    parser.add_argument("--is-ocr", action="store_true", help="Force the OCR path")
    parser.add_argument("--no-formula", action="store_true", help="Disable formula recognition")
    parser.add_argument("--no-table", action="store_true", help="Disable table recognition")
    parser.add_argument(
        "--extra-formats",
        default=None,
        help="Extra export formats, comma-separated: docx/html/latex",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help=f"Files per submit (API cap {MAX_BATCH_SIZE})",
    )
    parser.add_argument("--poll-interval", type=int, default=5, help="Batch poll interval (seconds)")
    parser.add_argument("--batch-timeout", type=int, default=3600, help="Max wait per batch (seconds)")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N pending PDFs (debug)")
    parser.add_argument("--force", action="store_true", help="Ignore existing output dirs and rerun")
    parser.add_argument("--dry-run", action="store_true", help="Print pending files only; do not call the API")
    parser.add_argument(
        "--token",
        default=os.environ.get("MINERU_TOKEN"),
        help="MinerU token (default: MINERU_TOKEN env)",
    )
    args = parser.parse_args(argv)

    target = Path(args.target).expanduser().resolve()
    if not target.is_dir():
        print(f"ERROR: not a directory: {target}", file=sys.stderr)
        return 2

    all_pdfs = find_pdfs(target)
    if not all_pdfs:
        print(f"No .pdf files found under {target}")
        return 0

    if args.force:
        pending = list(all_pdfs)
        skipped: list[Path] = []
    else:
        pending = []
        skipped = []
        for pdf in all_pdfs:
            (skipped if already_done(pdf) else pending).append(pdf)

    print(
        f"Scan {target}: {len(all_pdfs)} pdf files, "
        f"{len(skipped)} already done, {len(pending)} pending."
    )
    if args.limit is not None:
        pending = pending[: args.limit]
        print(f"--limit: processing {len(pending)} file(s).")

    if not pending:
        print("Nothing to do.")
        return 0

    if args.dry_run:
        print("--dry-run, pending files:")
        for pdf in pending:
            print(f"  {_rel(pdf, target)}")
        return 0

    if not args.token:
        print("ERROR: no token; set MINERU_TOKEN or pass --token", file=sys.stderr)
        return 2

    extra_formats = (
        [s.strip() for s in args.extra_formats.split(",") if s.strip()]
        if args.extra_formats
        else None
    )
    options = {
        "model_version": args.model,
        "language": args.language,
        "is_ocr": args.is_ocr,
        "enable_formula": not args.no_formula,
        "enable_table": not args.no_table,
        "page_ranges": args.page_ranges,
        "extra_formats": extra_formats,
    }

    batch_size = max(1, min(args.batch_size, MAX_BATCH_SIZE))
    overall: Counter = Counter()
    exit_code = 0

    pbar = tqdm(total=len(pending), desc="Total", unit="pdf", dynamic_ncols=True)
    try:
        for batch in _chunked(pending, batch_size):
            try:
                stats = process_batch(
                    args.token,
                    batch,
                    options=options,
                    poll_interval=args.poll_interval,
                    batch_timeout=args.batch_timeout,
                    pbar=pbar,
                    root=target,
                )
                overall.update(stats)
            except QuotaExhaustedError as e:
                tqdm.write(f"[FATAL] MinerU API quota exhausted: {e}")
                exit_code = 3
                break
            except AuthError as e:
                tqdm.write(f"[FATAL] MinerU token auth failed: {e}")
                exit_code = 4
                break
            except MineruError as e:
                tqdm.write(f"[FATAL] MinerU API error: {e}")
                exit_code = 5
                break
            except requests.HTTPError as e:
                status = e.response.status_code if e.response is not None else "?"
                tqdm.write(f"[FATAL] HTTP {status}: {e}")
                exit_code = 6
                break
    finally:
        pbar.close()

    print("\n========== summary ==========")
    print(f"target            : {target}")
    print(f"pdf scanned       : {len(all_pdfs)}")
    print(f"already done      : {len(skipped)}")
    print(f"submitted         : {len(pending)}")
    for k in ("done", "failed", "upload_failed", "extract_failed", "done_no_zip", "timeout"):
        if overall.get(k):
            print(f"  {k:<16}: {overall[k]}")
    if exit_code != 0:
        print(f"Stopped early, exit code {exit_code}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
