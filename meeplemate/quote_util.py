import re
import unicodedata
from dataclasses import dataclass
from typing import List, Tuple, TypedDict, Literal

from rapidfuzz import fuzz


BLOCKQUOTE_PATTERN = re.compile(r'(?:^>.*(?:\n^>.*)*)(?:\n^>.*$)?', re.MULTILINE)
CITATION_PAGE_TITLE_REGEX = re.compile(
    r'''^\s*(?P<title>.*[^, ]),?\s*(pg?[.])\s*(?P<page>[0-9]+)\s*$'''
)
INLINE_QUOTE_CITATION_REGEX = re.compile(
    r'''(?<!>\s)(["])(?P<quote>[^"]+?)\1\s*\((?P<citation>([^()]+|["][^"]+["])?,?\s*pg?[.]\s*[0-9]+)\)'''
)
ELLIPSIS_PAT = re.compile(r"\.\.\.|…")


class ExtractedCitation(TypedDict):
    text: str
    ref_name: str
    page: str
    start_index: int
    end_index: int


class ExtractedQuote(TypedDict):
    text: str
    quote: str
    quote_type: Literal["blockquote", "inline"]
    start_index: int
    end_index: int
    citation: ExtractedCitation | None


@dataclass(frozen=True)
class MatchResult:
    score: float
    start: int   # original doc index (inclusive)
    end: int     # original doc index (exclusive)
    matched_text: str
    part_scores: list[float]


def normalize_quotes_and_parens(text: str) -> str:
    # Normalize quotes and parantheses
    normalized_quote = text
    # Replace all open parens with '(' and close parens with ')'
    normalized_quote = re.sub(r'[（﹙❨]', '(', normalized_quote)
    normalized_quote = re.sub(r'[）﹚❩]', ')', normalized_quote)
    # Replace all open double quotes with '"' and close double quotes with '"'
    normalized_quote = re.sub(r'[“«„‟❝❛"]', '"', normalized_quote)
    normalized_quote = re.sub(r'[”»“‟❞❜"]', '"', normalized_quote)
    # Replace all open single quotes with "'" and close single quotes with "'"
    normalized_quote = re.sub(r"[‘‹']", "'", normalized_quote)
    normalized_quote = re.sub(r"[’›'']", "'", normalized_quote)
    return normalized_quote


def extract_blockquotes(text: str) -> List[Tuple[int, int, str]]:
    """Extract block quotes from the given text."""
    result: List[Tuple[int, int, str]] = []
    matches = BLOCKQUOTE_PATTERN.finditer(text)
    if not matches:
        return result
    
    for match in matches:
        text = match.group()
        start_idx = match.start()
        end_idx = match.end()
        result.append((start_idx, end_idx, text))
    return result


def extract_inline_quotes(text: str) -> List[Tuple[int, int, str]]:
    """Extract inline quotes from the given text."""
    result: List[Tuple[int, int, str]] = []
    normalized = normalize_quotes_and_parens(text)
    # Regex to find inline quotes with citations
    matches = INLINE_QUOTE_CITATION_REGEX.finditer(normalized)
    if not matches:
        return result
    for match in matches:
        start_idx = match.start()
        end_idx = match.end()
        quote_text = text[start_idx:end_idx]
        result.append((start_idx, end_idx, quote_text))
    return result


def extract_ref_name_and_page(citation: str) -> Tuple[str, str]|None:
    # Remove surrounding parens if present
    normalized = normalize_quotes_and_parens(citation)
    if normalized.startswith('('):
        citation = citation[1:]
    if normalized.endswith(')'):
        citation = citation[:-1]
    
    # Use regex to extract title and page number
    match = CITATION_PAGE_TITLE_REGEX.match(citation)
    if not match:
        return None
    
    # Remove any surrounding quotes from title
    title = match.group('title')
    normalized_title = normalize_quotes_and_parens(title)
    if normalized_title.startswith('"') and normalized_title.endswith('"'):
        title = title[1:-1].strip()
    elif normalized_title.startswith("'") and normalized_title.endswith("'"):
        title = title[1:-1].strip()
    
    # Remove any surrounding markdown formatting from title
    for md_format in ['**', '*', '__', '_', '~~']:
        if title.startswith(md_format) and title.endswith(md_format):
            title = title[len(md_format):-len(md_format)].strip()
    
    page = match.group('page').strip()
    return (title, page)


def extract_citation_text(quote: str) -> Tuple[int, int, str]|None:
    """Extract the citation from a blockquote string."""
    # Normalize quotes and parantheses
    normalized_quote = normalize_quotes_and_parens(quote)

    stack = []
    if normalized_quote[-1] != ')':
        return None
    
    for i in range(len(normalized_quote) - 1, -1, -1):
        char = normalized_quote[i]
        if char == ')':
            stack.append(char)
        elif char == '(':
            if not stack:
                return None
            stack.pop()
            if not stack:
                citation = normalized_quote[i:].strip()
                return (i, len(quote), citation)
    return None


def strip_quotes(quote: str) -> str:
    quote = quote.strip()
    normalized = normalize_quotes_and_parens(quote)
    if normalized.startswith('"') and normalized.endswith('"'):
        quote = quote[1:-1].strip()
    return quote


def strip_blockquote_markers_and_quotes(quote: str) -> str:
    """Strip the blockquote markers (>) from the given blockquote text."""
    lines = quote.splitlines()
    stripped_lines = []
    for line in lines:
        stripped_line = line.lstrip('> ').rstrip()
        stripped_lines.append(stripped_line)
    quote = ' '.join(stripped_lines)
    quote = quote.strip()
    return strip_quotes(quote)


def find_quotes_in_text(text: str) -> List[ExtractedQuote]:
    """Find all quotes (block and inline) in the given text."""
    extracted_quotes: List[ExtractedQuote] = []
    
    # Extract blockquotes
    blockquotes = extract_blockquotes(text)
    for start_idx, end_idx, quote_text in blockquotes:
        citation_info = extract_citation_text(quote_text)
        citation: ExtractedCitation | None = None
        if citation_info:
            cit_start, cit_end, cit_text = citation_info
            ref_name_page = extract_ref_name_and_page(cit_text)
            if ref_name_page:
                ref_name, page = ref_name_page
                citation = ExtractedCitation(
                    text=cit_text,
                    ref_name=ref_name,
                    page=page,
                    start_index=cit_start,
                    end_index=cit_end
                )
        citation_start = citation["start_index"] if citation else -1
        extracted_quotes.append(
            ExtractedQuote(
                text=quote_text,
                quote=strip_blockquote_markers_and_quotes(quote_text[:citation_start]),
                quote_type="blockquote",
                start_index=start_idx,
                end_index=end_idx,
                citation=citation
            )
        )
    
    # Extract inline quotes
    inline_quotes = extract_inline_quotes(text)
    for start_idx, end_idx, quote_text in inline_quotes:
        citation_info = extract_citation_text(quote_text)
        citation: ExtractedCitation | None = None
        if citation_info:
            cit_start, cit_end, cit_text = citation_info
            ref_name_page = extract_ref_name_and_page(cit_text)
            if ref_name_page:
                ref_name, page = ref_name_page
                citation = ExtractedCitation(
                    text=cit_text,
                    ref_name=ref_name,
                    page=page,
                    start_index=cit_start,
                    end_index=cit_end
                )
        citation_start = citation["start_index"] if citation else -1
        extracted_quotes.append(
            ExtractedQuote(
                text=quote_text,
                quote=strip_quotes(quote_text[:citation_start]),
                quote_type="inline",
                start_index=start_idx,
                end_index=end_idx,
                citation=citation
            )
        )
    
    return extracted_quotes


def normalize_with_map(s: str) -> tuple[str, list[int]]:
    """
    Normalize while keeping a mapping from each normalized character index
    back to an original-string index.

    Mapping guarantee: norm_to_orig[i] is an index into *original* `s`.
    """
    out_chars: list[str] = []
    out_map: list[int] = []

    def emit(ch: str, orig_i: int) -> None:
        out_chars.append(ch)
        out_map.append(orig_i)

    # Normalize per original character to preserve mapping even when NFKC expands chars.
    for i, ch in enumerate(s):
        for nch in unicodedata.normalize("NFKC", ch):
            nch = nch.lower()
            if nch in ("“", "”"):
                nch = '"'
            elif nch in ("‘", "’"):
                nch = "'"

            if nch.isalnum() or nch == "_":
                emit(nch, i)
            elif nch.isspace():
                emit(" ", i)
            else:
                # punctuation -> space (keeps boundaries)
                emit(" ", i)

    # Collapse whitespace runs to a single space (keep mapping to first orig idx in run)
    norm_chars: list[str] = []
    norm_map: list[int] = []
    prev_space = False
    for ch, oi in zip(out_chars, out_map):
        if ch == " ":
            if not prev_space:
                norm_chars.append(" ")
                norm_map.append(oi)
            prev_space = True
        else:
            norm_chars.append(ch)
            norm_map.append(oi)
            prev_space = False

    # Trim leading/trailing spaces
    while norm_chars and norm_chars[0] == " ":
        norm_chars.pop(0); norm_map.pop(0)
    while norm_chars and norm_chars[-1] == " ":
        norm_chars.pop(); norm_map.pop()

    return "".join(norm_chars), norm_map


def normalize_no_map(s: str) -> str:
    return normalize_with_map(s)[0]


def split_on_ellipsis(quote: str) -> list[str]:
    return [p.strip() for p in ELLIPSIS_PAT.split(quote) if p.strip()]


def find_quote_with_gaps(
    doc: str,
    quote: str,
    *,
    min_part_score: int = 85,
    max_gap_norm_chars: int = 1500,
    # Candidate gen knobs (docs are small, so windowing is fine)
    window_size: int = 1200,
    window_step: int = 200,
    top_k_windows: int = 8,
) -> MatchResult | None:
    norm_doc, norm_to_orig = normalize_with_map(doc)

    parts = split_on_ellipsis(quote)
    norm_parts = [normalize_no_map(p) for p in parts if p.strip()]
    if not norm_parts:
        return None

    first = norm_parts[0]
    if not norm_doc:
        return None

    # --- 1) Candidate generation: scan overlapping windows and keep top K ---
    candidates: list[tuple[float, int, int]] = []  # (score, win_start, win_end)
    n = len(norm_doc)
    if window_size >= n:
        candidates = [(fuzz.partial_ratio(first, norm_doc), 0, n)]
    else:
        for ws in range(0, n - window_size + 1, window_step):
            we = ws + window_size
            score = fuzz.partial_ratio(first, norm_doc[ws:we])
            candidates.append((score, ws, we))
        candidates.sort(reverse=True, key=lambda x: x[0])
        candidates = candidates[:top_k_windows]

    best: tuple[float, int, int, list[float]] | None = None  # score, norm_start, norm_end, part_scores

    # --- 2) For each candidate window, do precise alignment + chain the remaining parts in order ---
    for _, ws, we in candidates:
        window = norm_doc[ws:we]
        a0 = fuzz.partial_ratio_alignment(first, window)
        if a0.score < min_part_score:
            continue

        norm_start = ws + a0.dest_start
        prev_end = ws + a0.dest_end
        part_scores = [float(a0.score)]
        ok = True

        for part in norm_parts[1:]:
            seg_start = prev_end
            seg_end = min(len(norm_doc), prev_end + max_gap_norm_chars)
            if seg_start >= seg_end:
                ok = False
                break

            seg = norm_doc[seg_start:seg_end]
            a = fuzz.partial_ratio_alignment(part, seg)
            if a.score < min_part_score:
                ok = False
                break

            prev_end = seg_start + a.dest_end
            part_scores.append(float(a.score))

        if not ok:
            continue

        norm_end = prev_end
        score = sum(part_scores) / len(part_scores)

        if best is None or score > best[0]:
            best = (score, norm_start, norm_end, part_scores)

    if best is None:
        return None

    score, norm_start, norm_end, part_scores = best

    # --- 3) Map normalized span back to original indices ---
    orig_start = norm_to_orig[norm_start]
    orig_end = norm_to_orig[norm_end - 1] + 1  # exclusive

    # Extend orig_end to complete the current word and include trailing punctuation
    # that was normalized away. This handles two cases:
    # 1. Fuzzy match may end mid-word (e.g., "her" when it should be "here")
    # 2. Trailing punctuation gets normalized to spaces and trimmed
    while orig_end < len(doc):
        ch = doc[orig_end]
        if ch.isspace():
            break  # Stop at whitespace
        # Continue through alphanumeric (complete the word) and punctuation
        orig_end += 1

    return MatchResult(
        score=score,
        start=orig_start,
        end=orig_end,
        matched_text=doc[orig_start:orig_end],
        part_scores=part_scores,
    )
