#!/usr/bin/env python3
"""Check that every blockquote in a decisions file is really in the rulebook.

A corrected answer's blockquotes become the `evidence:` of the promoted test
case, so a quote that does not match the corpus verbatim turns into a retrieval
regression that looks like a ranking bug months later. This catches that, plus
the rulebook-name and page-number mistakes that silently produce unscoreable
evidence.

    python verify_quotes.py data/eval_gen/2026-08-29/decisions/decisions.json

Checks each blockquote in every `corrected_answer`:
  * the citation parses as `(<Rulebook Name>, p. <number>)`
  * `<Rulebook Name>` matches a `name:` in that game's rulebooks.yaml
  * the quoted text appears verbatim on that printed page **of that rulebook**

Exit status is non-zero if anything fails.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
RULES = REPO / "data" / "ingestion" / "rules"
ARTIFACTS = REPO / "data" / "ingestion" / "artifacts"

CITATION = re.compile(r"\((?P<book>.+?),\s*p\.\s*(?P<page>\d+)\)\s*$")


def package_for(game: str) -> str:
    """Ingestion package holding `game`'s rulebooks.

    The package directory is usually the game id but need not be -- munchkin
    lives in `munchkin_rules` -- so prefer the declared game_id and fall back to
    the directory name.
    """
    for cfg in sorted(RULES.glob("*/rulebooks.yaml")):
        if (yaml.safe_load(cfg.read_text()) or {}).get("game_id") == game:
            return cfg.parent.name
    if (RULES / game).is_dir():
        return game
    raise SystemExit(f"no ingestion package for game {game!r} under {RULES}")


def normalise(text: str) -> str:
    """Fold the OCR's presentation noise so real mismatches stand out.

    The artifacts carry LaTeX-wrapped glyphs (`\\( ^{♦} \\)`), curly quotes, and
    line wrapping that an answer reasonably renders plainly. None of that is a
    quoting error; a changed word is.
    """
    text = re.sub(r"\\\(\s*\^?\{?(.*?)\}?\s*\\\)", r"\1", text)
    text = text.replace("\\(", "").replace("\\)", "")
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\(\s*([^()\s]+)\s*\)", r"(\1)", text)
    return text.strip()


def load_pages(package: str) -> dict[str, dict[int, str]]:
    """Rulebook name -> printed page number -> normalised page text.

    Kept per rulebook rather than merged. A package usually holds several books
    whose page numbers overlap completely, so one flat page->text map silently
    accepts a quote lifted from one book and cited to the other: oathsworn's two
    books both run from page 1, and a merged map validates every citation
    against either of them. That is the mistake most worth catching here,
    because the quote reads as correct and only the ref name is wrong.

    Printed numbers come from the `page_numbers/` artifact, never from the file
    index: packages skip pages (waterdeep's fourth file is printed page 7), and
    a citation off by three is worse than no citation.
    """
    root = ARTIFACTS / package
    if not (root / "text").is_dir():
        raise SystemExit(f"no OCR text under {root}/text -- has this package been ingested?")

    cfg = yaml.safe_load((RULES / package / "rulebooks.yaml").read_text()) or {}
    # Artifact directories are named for the base64 of the source PDF filename.
    dir_to_book = {
        base64.b64encode(rb["path"].encode()).decode(): rb["name"]
        for rb in cfg.get("rulebooks", [])
    }

    books: dict[str, dict[int, str]] = {name: {} for name in dir_to_book.values()}
    for doc_dir in sorted((root / "text").iterdir()):
        if not doc_dir.is_dir():
            continue
        book = dir_to_book.get(doc_dir.name)
        if book is None:
            # Ingested text with no entry in rulebooks.yaml. Nothing can cite it
            # by name, so it cannot verify a quote either way.
            continue
        pages = books[book]
        for md in sorted(doc_dir.glob("[0-9]*.md")):
            pn = root / "page_numbers" / doc_dir.name / f"{md.stem}.txt"
            if not pn.exists():
                continue
            page = int(pn.read_text().strip())
            # Several files can share one printed page; concatenate them.
            pages[page] = pages.get(page, "") + "\n" + md.read_text()

    return {book: {k: normalise(v) for k, v in pages.items()}
            for book, pages in books.items()}


def locate(books: dict[str, dict[int, str]], parts: list[str]) -> str:
    """Where the quote really lives, for the failure message.

    A sentence copied accurately but attributed to the wrong book -- or to the
    right book's wrong page -- is the common case, and naming the real page
    turns the failure from a puzzle into a one-line edit.
    """
    hits = sorted(f"{book} p.{page}"
                  for book, pages in books.items()
                  for page, text in pages.items()
                  if all(part in text for part in parts))
    return f" -- it is on {', '.join(hits)}" if hits else ""


def blockquotes(answer: str) -> list[list[str]]:
    """Each blockquote as its list of un-prefixed lines."""
    out = []
    for block in re.findall(r"((?:^>.*\n?)+)", answer, re.M):
        out.append([re.sub(r"^>\s?", "", ln) for ln in block.strip().split("\n")])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("decisions", type=Path, help="path to a decisions.json")
    args = ap.parse_args()

    doc = json.loads(args.decisions.read_text())
    cache: dict[str, dict[str, dict[int, str]]] = {}

    checked = failures = 0
    for row in doc.get("decisions", []):
        answer = row.get("corrected_answer")
        if not answer:
            continue
        game = row["game"]
        if game not in cache:
            cache[game] = load_pages(package_for(game))
        books = cache[game]

        blocks = blockquotes(answer)
        if not blocks:
            print(f"FAIL {row['id']}: corrected_answer has no blockquote, so it "
                  f"will promote with empty evidence")
            failures += 1
            continue

        for lines in blocks:
            checked += 1
            m = CITATION.match(lines[-1].strip())
            if not m:
                print(f"FAIL {row['id']}: last line of a blockquote is not a "
                      f"'(Book, p. N)' citation, so this quote is dropped from "
                      f"evidence: {lines[-1].strip()[:80]!r}")
                failures += 1
                continue
            book, page = m.group("book").strip(), int(m.group("page"))
            if book not in books:
                print(f"FAIL {row['id']}: rulebook {book!r} is not a name in "
                      f"rulebooks.yaml ({sorted(books)})")
                failures += 1
                continue
            pages = books[book]
            if page not in pages:
                print(f"FAIL {row['id']}: no ingested page {page} for {book!r}")
                failures += 1
                continue
            body = normalise(" ".join(ln for ln in lines[:-1] if ln.strip()))
            # `...` is the answers' elision marker; each side must still match.
            parts = [part.strip() for part in body.split("...") if part.strip()]
            if all(part in pages[page] for part in parts):
                continue
            print(f"FAIL {row['id']} [{book} p.{page}]: not verbatim on that "
                  f"page: {body[:110]!r}{locate(books, parts)}")
            failures += 1

    print(f"\n{checked} blockquote(s) checked, {failures} problem(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
