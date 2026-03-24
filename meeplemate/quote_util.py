import html
import re
import sys
import unicodedata
from dataclasses import dataclass
from typing import List, Tuple, TypedDict, Literal

from cydifflib import SequenceMatcher


BLOCKQUOTE_PATTERN = re.compile(r'(?:^>.*(?:\n^>.*)*)(?:\n^>.*$)?', re.MULTILINE)
CITATION_PAGE_TITLE_REGEX = re.compile(
    r'''^\s*(?P<title>.*[^, ]),?\s*(pg?[.])\s*(?P<page>[0-9]+)\s*$'''
)
INLINE_QUOTE_CITATION_REGEX = re.compile(
    r'''(?<!>\s)(["])(?P<quote>[^"]+?)\1\s*\((?P<citation>([^()]+|["][^"]+["])?,?\s*pg?[.]\s*[0-9]+)\)'''
)
QUOTE_SPLIT_PAT = re.compile(r"\.\.\.|…|\n")


class ExtractedCitation(TypedDict):
    text: str
    ref_name: str
    page: str
    start_index: int
    end_index: int


class ExtractedQuote(TypedDict):
    text: str
    """The original quote text, including citation if present and any blockquote markers and additional formatting."""
    quote: str
    """The cleaned quote text. No blockquote markers. No citation. No quotation marks."""
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
    """Extract block quotes from the given text, including lazy continuation lines."""
    result: List[Tuple[int, int, str]] = []
    matches = BLOCKQUOTE_PATTERN.finditer(text)
    if not matches:
        return result

    for match in matches:
        quote_text = match.group()
        start_idx = match.start()
        end_idx = match.end()

        # Check for lazy continuation (lines that are part of the quote but don't start with >)
        # These lines continue until we hit a blank line or a citation
        remaining_text = text[end_idx:]

        # Look for continuation lines (not starting with >, not blank)
        # until we hit a blank line or end of text
        continuation_match = re.match(r'^(\n[^\n>][^\n]*(?:\n[^\n>][^\n]*)*)', remaining_text)
        if continuation_match:
            # Include the continuation lines
            continuation_text = continuation_match.group(1)
            end_idx += len(continuation_text)
            quote_text += continuation_text

        # Now check if there's a citation on the next line (common LLM pattern)
        # Look for pattern: \n\n(Citation, p. X) after the blockquote
        remaining_text = text[end_idx:]
        citation_on_next_line = re.match(r'^[\s\n]*(\([^)]+,?\s*pg?[.]\s*[0-9]+\))', remaining_text)
        if citation_on_next_line:
            # Include the citation as part of the blockquote
            citation_text = citation_on_next_line.group(1)
            # Find the actual end of the citation in the original text
            citation_end = end_idx + citation_on_next_line.end(1)
            quote_text = text[start_idx:citation_end]
            end_idx = citation_end

        result.append((start_idx, end_idx, quote_text))
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
    """Strip the blockquote markers (>) from the given blockquote text.

    Consecutive non-empty lines are joined with a single space (continuation
    lines within the same paragraph).  Blank ``>`` lines are preserved as
    ``\\n\\n`` paragraph breaks so that downstream matching can treat each
    paragraph independently.
    """
    lines = quote.splitlines()
    stripped_lines = [line.lstrip('> ').rstrip() for line in lines]

    # Group consecutive non-empty lines into paragraphs
    paragraphs: list[str] = []
    current: list[str] = []
    for line in stripped_lines:
        if line:
            current.append(line)
        else:
            if current:
                paragraphs.append(' '.join(current))
                current = []
    if current:
        paragraphs.append(' '.join(current))

    quote = '\n\n'.join(paragraphs)
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
        if citation:
            quote_without_citation = quote_text[:citation["start_index"]]
        else:
            quote_without_citation = quote_text
        extracted_quotes.append(
            # TODO: Maybe we should fix the table element escaping here instead
            # of in the blockquote formatting function?
            ExtractedQuote(
                text=quote_text,
                quote=strip_blockquote_markers_and_quotes(quote_without_citation),
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
        if citation:
            quote_without_citation = quote_text[:citation["start_index"]]
        else:
            quote_without_citation = quote_text
        extracted_quotes.append(
            ExtractedQuote(
                text=quote_text,
                quote=strip_quotes(quote_without_citation),
                quote_type="inline",
                start_index=start_idx,
                end_index=end_idx,
                citation=citation
            )
        )
    
    return extracted_quotes


# ---------------------------------------------------------------------------
# HTML tag stripping
# ---------------------------------------------------------------------------

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def strip_html_tags_with_map(s: str) -> tuple[str, list[int]]:
    """Strip HTML tags, keeping a char→original-index map.

    Every ``<…>`` region is removed; all other characters pass through.
    Returns ``(stripped_text, index_map)`` with the same contract as
    :func:`normalize_with_map`.
    """
    out_chars: list[str] = []
    out_map: list[int] = []

    tag_regions = [(m.start(), m.end()) for m in _HTML_TAG_RE.finditer(s)]

    cursor = 0
    region_idx = 0

    while cursor < len(s):
        if region_idx < len(tag_regions) and cursor == tag_regions[region_idx][0]:
            # Emit a space so tags act as word boundaries (e.g. between
            # table cells).  The normalization step collapses runs of
            # spaces, so this doesn't introduce extra tokens.
            out_chars.append(" ")
            out_map.append(tag_regions[region_idx][0])
            cursor = tag_regions[region_idx][1]
            region_idx += 1
        else:
            out_chars.append(s[cursor])
            out_map.append(cursor)
            cursor += 1

    return "".join(out_chars), out_map


def strip_html_tags(s: str) -> str:
    """Strip HTML tags (replaced with spaces), returning only the cleaned text."""
    return strip_html_tags_with_map(s)[0]


# ---------------------------------------------------------------------------
# LaTeX inline-math stripping
# ---------------------------------------------------------------------------

_INLINE_MATH_RE = re.compile(r"\\\(.*?\\\)", re.DOTALL)

_LATEX_COMMAND_REPLACEMENTS: dict[str, str] = {
    r"\prime": "'",
    r"\circ": "\u00b0",  # degree sign
}


def _emit_latex_inner(
    inner: str,
    base_offset: int,
    out_chars: list[str],
    out_map: list[int],
) -> None:
    """Process content between \\( and \\), emitting chars with original-index mapping."""
    i = 0
    while i < len(inner):
        orig_i = base_offset + i

        # --- backslash commands ---
        if inner[i] == "\\":
            # Check known replacements first
            matched = False
            for cmd, replacement in _LATEX_COMMAND_REPLACEMENTS.items():
                if inner[i:].startswith(cmd):
                    for ch in replacement:
                        out_chars.append(ch)
                        out_map.append(orig_i)
                    i += len(cmd)
                    matched = True
                    break
            if matched:
                continue
            # Unknown \command – skip backslash + any alpha chars (command name)
            j = i + 1
            while j < len(inner) and inner[j].isalpha():
                j += 1
            i = j if j > i + 1 else i + 1
            continue

        # --- superscript ^{...} – emit contents, strip ^ and braces ---
        if inner[i] == "^" and i + 1 < len(inner) and inner[i + 1] == "{":
            brace_start = i + 2
            depth = 1
            j = brace_start
            while j < len(inner) and depth > 0:
                if inner[j] == "{":
                    depth += 1
                elif inner[j] == "}":
                    depth -= 1
                j += 1
            # Recurse into brace contents
            _emit_latex_inner(
                inner[brace_start : j - 1],
                base_offset + brace_start,
                out_chars,
                out_map,
            )
            i = j
            continue

        # --- lone braces – skip ---
        if inner[i] in "{}":
            i += 1
            continue

        # --- everything else (digits, spaces, operators) – pass through ---
        out_chars.append(inner[i])
        out_map.append(orig_i)
        i += 1


def strip_latex_with_map(s: str) -> tuple[str, list[int]]:
    """Strip LaTeX inline math notation, keeping a char→original-index map.

    Within ``\\(...\\)`` regions the delimiters are removed and commands like
    ``\\prime`` and ``\\circ`` are replaced with their Unicode equivalents.
    Text outside math regions passes through unchanged.

    Returns ``(stripped_text, index_map)`` with the same contract as
    :func:`normalize_with_map`.
    """
    out_chars: list[str] = []
    out_map: list[int] = []

    # Pre-compute math regions
    math_regions: list[tuple[int, int]] = [
        (m.start(), m.end()) for m in _INLINE_MATH_RE.finditer(s)
    ]

    cursor = 0
    region_idx = 0

    while cursor < len(s):
        if region_idx < len(math_regions) and cursor == math_regions[region_idx][0]:
            region_start, region_end = math_regions[region_idx]
            inner = s[region_start + 2 : region_end - 2]  # between \( and \)
            _emit_latex_inner(inner, region_start + 2, out_chars, out_map)
            cursor = region_end
            region_idx += 1
        else:
            out_chars.append(s[cursor])
            out_map.append(cursor)
            cursor += 1

    return "".join(out_chars), out_map


def strip_latex(s: str) -> str:
    """Strip LaTeX inline math notation, returning only the cleaned text."""
    return strip_latex_with_map(s)[0]


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


def split_quote_parts(quote: str) -> list[str]:
    """Split a quote on ellipsis (``...``/``…``) and newlines."""
    return [p.strip() for p in QUOTE_SPLIT_PAT.split(quote) if p.strip()]


def find_quote_with_gaps(
    doc: str,
    quote: str,
    *,
    min_score: float = 92.0,
) -> MatchResult | None:
    # Strip markup (HTML tags, LaTeX math) before normalizing so that tag
    # names and LaTeX commands don't introduce spurious alphanumeric tokens
    # (e.g. "td", "prime") that break fuzzy matching against the LLM's
    # plain-text quotes.

    if "&lt;table&gt;" in quote:
        # Unescape html
        quote = html.unescape(quote)
    if "<table>" in quote:
        quote = strip_html_tags(quote)

    html_stripped, html_map = strip_html_tags_with_map(doc)
    latex_stripped, latex_map = strip_latex_with_map(html_stripped)
    norm_doc, norm_to_latex = normalize_with_map(latex_stripped)
    norm_to_orig = [html_map[latex_map[i]] for i in norm_to_latex]

    cleaned_quote = strip_html_tags(strip_latex(quote))
    norm_quote = normalize_no_map(cleaned_quote)
    if not norm_quote or not norm_doc:
        return None

    s = SequenceMatcher(None, norm_quote, norm_doc, autojunk=False)
    blocks = [b for b in s.get_matching_blocks() if b.size > 0]
    if not blocks:
        return None

    matched = sum(b.size for b in blocks)
    score = matched / len(norm_quote) * 100
    if score < min_score:
        return None

    norm_start = blocks[0].b
    norm_end = blocks[-1].b + blocks[-1].size

    orig_start = norm_to_orig[norm_start]
    orig_end = norm_to_orig[norm_end - 1] + 1  # exclusive

    # Extend orig_end to complete the current word and include trailing punctuation
    # that was normalized away.
    while orig_end < len(doc):
        if doc[orig_end].isspace():
            break
        orig_end += 1

    return MatchResult(
        score=score,
        start=orig_start,
        end=orig_end,
        matched_text=doc[orig_start:orig_end],
        part_scores=[score],
    )


def expand_to_full_paragraphs(doc_markdown: str, quote: str, *, max_additional_chars: int = sys.maxsize) -> str:
    """Expand the matched quote to full paragraphs in the original markdown.

    If there's a heading (line starting with '#') immediately above the paragraph,
    it will be included in the expansion (subject to max_additional_chars constraint).
    """
    # Find the match in the original markdown
    match = find_quote_with_gaps(doc_markdown, quote)
    if match is None:
        return quote  # Fallback: return original quote

    start = match.start
    end = match.end

    # Expand start backwards to the beginning of the paragraph
    para_start = doc_markdown.rfind('\n\n', 0, start)
    if para_start == -1:
        para_start = 0
    else:
        para_start += 2  # Move past the double newline

    # Expand end forwards to the end of the paragraph
    para_end = doc_markdown.find('\n\n', end)
    if para_end == -1:
        para_end = len(doc_markdown)

    # Look for heading above paragraph
    heading_start = para_start  # default: no heading
    if para_start > 0:
        # Skip backwards over newlines
        pos = para_start - 1
        while pos >= 0 and doc_markdown[pos] in '\n\r':
            pos -= 1

        if pos >= 0:
            # Find start of this line
            line_start = doc_markdown.rfind('\n', 0, pos)
            line_start = 0 if line_start == -1 else line_start + 1

            # Check if it's a heading
            if line_start < len(doc_markdown) and doc_markdown[line_start] == '#':
                heading_start = line_start

    # Check max_additional_chars with graceful fallback
    additional_chars_para = (start - para_start) + (para_end - end)
    if additional_chars_para > max_additional_chars:
        return match.matched_text  # Return matched text without expansion

    # Determine final start position
    if heading_start < para_start:
        additional_chars_with_heading = (start - heading_start) + (para_end - end)
        if additional_chars_with_heading <= max_additional_chars:
            final_start = heading_start
        else:
            final_start = para_start  # Graceful fallback: include paragraph but not heading
    else:
        final_start = para_start

    expanded_quote = doc_markdown[final_start:para_end].strip()
    return expanded_quote


def are_blockquotes_adjacent(quote1: ExtractedQuote, quote2: ExtractedQuote, text: str) -> bool:
    # Two blockquotes are adjacent if there is no content inbetween them except
    # for whitespace characters

    # Make it so that quote1 represents the earlier quote in the text
    if quote1["start_index"] > quote2["start_index"]:
        quote1, quote2 = quote2, quote1
    
    # Now lets get the content between the two quotes
    between_text = text[quote1['end_index']:quote2['start_index']]
    return between_text.strip() == ''