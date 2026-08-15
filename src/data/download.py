"""Acquire the raw datasets.

HotpotQA (distractor) ships its own supporting and distractor paragraphs, so no
Wikipedia dump is needed. FEVER ships claims plus the *titles* of supporting pages,
so its page text has to be pulled separately from the wiki_pages configuration -
that is the extra step FEVER needs and HotpotQA does not.

Raw data lands in data/raw/<dataset>/ as JSONL, decoupled from the HuggingFace cache
so later stages do not depend on the cache surviving.
"""
from __future__ import annotations

from pathlib import Path

from ..config.config import RAW_DIR
from ..utils.io import write_json, write_jsonl
from ..utils.logging_utils import get_logger

log = get_logger()


def _require_datasets():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "the 'datasets' package is required to download data:\n"
            "    pip install datasets"
        ) from exc
    return load_dataset


def download_hotpotqa(split: str = "validation", out_dir: Path | None = None,
                      cache_dir: str | None = None, limit: int = 0,
                      force: bool = False) -> Path:
    """Download HotpotQA distractor and dump it to JSONL.

    Each row keeps the question, the answer, the ten context paragraphs and the
    supporting-fact titles, which become the gold document ids downstream.

    Skips the download (and the network entirely) if the target file already exists,
    unless `force=True` - the raw dump does not change between runs, so there is no
    reason to refetch it.
    """
    out_dir = Path(out_dir or RAW_DIR / "hotpotqa")
    path = out_dir / f"{split}.jsonl"
    if path.exists() and not force:
        log.info("using cached %s (pass force=True to redownload)", path)
        return path

    load_dataset = _require_datasets()
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("downloading hotpot_qa/distractor split=%s", split)
    ds = load_dataset("hotpot_qa", "distractor", split=split,
                      cache_dir=cache_dir, trust_remote_code=True)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    n = write_jsonl(path, (
        {
            "id": row["id"],
            "question": row["question"],
            "answer": row["answer"],
            "type": row.get("type", ""),
            "level": row.get("level", ""),
            "context": row["context"],
            "supporting_facts": row["supporting_facts"],
        }
        for row in ds
    ))
    write_json(out_dir / f"{split}.meta.json",
               {"dataset": "hotpot_qa", "config": "distractor", "split": split, "n_rows": n})
    log.info("wrote %s (%d rows)", path, n)
    return path


def download_fever_claims(split: str = "labelled_dev", out_dir: Path | None = None,
                          cache_dir: str | None = None, limit: int = 0,
                          force: bool = False) -> Path:
    """Download FEVER claims and their evidence page titles.

    Skips the download if the target file already exists, unless `force=True`.
    """
    out_dir = Path(out_dir or RAW_DIR / "fever")
    path = out_dir / f"claims_{split}.jsonl"
    if path.exists() and not force:
        log.info("using cached %s (pass force=True to redownload)", path)
        return path

    load_dataset = _require_datasets()
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("downloading fever/v1.0 split=%s", split)
    ds = load_dataset("fever", "v1.0", split=split, cache_dir=cache_dir,
                      trust_remote_code=True)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    n = write_jsonl(path, (
        {
            "id": row["id"],
            "claim": row["claim"],
            "label": row["label"],
            "evidence_wiki_url": row.get("evidence_wiki_url", ""),
            "evidence_sentence_id": row.get("evidence_sentence_id", -1),
        }
        for row in ds
    ))
    write_json(out_dir / f"claims_{split}.meta.json",
               {"dataset": "fever", "config": "v1.0", "split": split, "n_rows": n})
    log.info("wrote %s (%d rows)", path, n)
    return path


def download_fever_pages(out_dir: Path | None = None, cache_dir: str | None = None,
                         max_pages: int = 0, min_chars: int = 50,
                         lead_chars: int = 1200, wanted_titles: set[str] | None = None,
                         scan_limit: int = 0, distractor_pages: int = 0,
                         distractor_every: int = 200, force: bool = False) -> Path:
    """Download the FEVER wiki_pages dump and dump page text to JSONL.

    Full (~5GB), because claims ship without page text. Only the lead section of each
    page is kept by default: FEVER evidence is almost always in the lead, and keeping
    whole pages inflates the corpus by an order of magnitude for little recall gain.
    Pass lead_chars=0 to keep the full page.

    `wanted_titles`, if given, switches to a streamed pass that keeps only pages whose
    title is in that set and stops as soon as they are all found (or `scan_limit` rows
    have been scanned) - this is what lets a small claims subset get real evidence
    coverage without pulling the full dump. Without it, `max_pages` is applied to the
    (still fully downloaded) dump, matching the old behaviour.

    The wiki_pages dump is roughly alphabetically ordered, so wanted titles are spread
    across the whole scan, not clustered near the start - `distractor_pages` keeps one
    non-wanted page every `distractor_every` scanned rows along the way, so the result
    is not merely the gold passages (which would make retrieval trivially easy).

    Skips the download entirely if the target file already exists, unless
    `force=True`.
    """
    out_dir = Path(out_dir or RAW_DIR / "fever")
    path = out_dir / "wiki_pages.jsonl"
    if path.exists() and not force:
        log.info("using cached %s (pass force=True to redownload)", path)
        return path

    load_dataset = _require_datasets()
    out_dir.mkdir(parents=True, exist_ok=True)

    streaming = wanted_titles is not None
    if streaming:
        log.info("downloading fever/wiki_pages (streamed, targeting %d titles)",
                 len(wanted_titles))
    else:
        log.info("downloading fever/wiki_pages (large - expect a long download)")
    ds = load_dataset("fever", "wiki_pages", split="wikipedia_pages",
                      cache_dir=cache_dir, trust_remote_code=True, streaming=streaming)

    from .preprocess import clean_fever_text

    remaining = set(wanted_titles) if wanted_titles else None
    n_distractors = 0

    def rows():
        nonlocal n_distractors
        written = 0
        scanned = 0
        for row in ds:
            scanned += 1
            title = row.get("id", "")
            is_wanted = remaining is not None and title in remaining
            if remaining is not None and not is_wanted:
                take_distractor = (distractor_pages and n_distractors < distractor_pages
                                   and scanned % distractor_every == 0)
                if not take_distractor:
                    if scan_limit and scanned >= scan_limit:
                        return
                    continue
                n_distractors += 1
            text = clean_fever_text(row.get("text", ""))
            if not title or len(text) < min_chars:
                continue
            yield {"title": title, "text": text[:lead_chars] if lead_chars else text}
            written += 1
            if remaining is not None:
                remaining.discard(title)
            if written % 10_000 == 0:
                log.info("  %d pages (%d rows scanned)", written, scanned)
            if max_pages and written >= max_pages:
                return
            if remaining is not None and not remaining:
                return
            if scan_limit and scanned >= scan_limit:
                return

    n = write_jsonl(path, rows())
    if remaining:
        log.warning("%d of %d wanted evidence titles were not found within the scan "
                    "limit - those questions will have incomplete gold coverage",
                    len(remaining), len(wanted_titles))
    log.info("wrote %s (%d pages)", path, n)
    return path
