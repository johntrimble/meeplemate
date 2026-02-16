
import asyncio
import base64
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import AsyncIterator, Sequence, Tuple

from PIL import Image
from openai import AsyncOpenAI
import yaml

from meeplemate.ingest.gamepackage import (
    GamePackage,
    Page,
    document_keys,
    get_page,
    get_pages_iter,
    load_game_package,
    page_md_path,
    page_structured,
    page_structured_fixed_path,
    page_structured_path,
    page_number_raw_path,
    page_number_path,
)
from meeplemate.util import (
    achain,
    amap,
    aslurp,
    aspit,
    aspit_json,
    aspit_yaml,
    pipeline,
    sink_into_queue,
    to_async_iter,
    xf_amap
)

from structlog import get_logger

logger = get_logger(__name__)


@dataclass
class PageResult:
    page_num: int
    image_path: Path
    structured_path: Path
    structured: Sequence[dict]


def get_page_count(pdf_path: Path) -> int:
    from pdf2image import pdfinfo_from_path

    info = pdfinfo_from_path(str(pdf_path))
    return int(info.get("Pages", 0))


def maybe_resize_image(image, max_size: int | None):
    if max_size is None:
        return image
    size = image.size
    if max(size) < max_size:
        return image
    ratio = max_size / max(size)
    new_width = int(image.width * ratio)
    new_height = int(image.height * ratio)
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized_image


def postprocess_and_save(image, output_path: Path, max_size: int | None) -> Path:
    image = image.convert("RGB")
    image = maybe_resize_image(image, max_size)
    image.save(output_path, format="PNG")
    return output_path


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


async def pdf_page_images_iter(pdf_path: Path, chunk_size: int = 4, dpi: int = 300) -> AsyncIterator[Tuple[int, Image.Image]]:
    """
    Yield (page_num, image) in order, loading the PDF in small chunks to keep memory bounded.
    """
    from pdf2image import convert_from_path

    page_count = get_page_count(pdf_path)
    for start in range(1, page_count + 1, chunk_size):
        end = min(start + chunk_size - 1, page_count)
        images = await asyncio.to_thread(
            convert_from_path,
            str(pdf_path),
            dpi=dpi,
            first_page=start,
            last_page=end,
        )
        for idx, image in enumerate(images):
            # Use zero-based page numbering for stable filenames/order
            yield start + idx - 1, image


async def page_and_image_iter(gp: GamePackage, document_key: str, pdf_path: Path, chunk_size: int = 10, dpi: int = 300) -> AsyncIterator[Tuple[Page, Image.Image]]:
    async for page_num, image in pdf_page_images_iter(pdf_path, chunk_size=chunk_size, dpi=dpi):
        page = get_page(gp, document_key, page_num)
        yield (page, image)


def re_match(text):
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.finditer(pattern, text, re.DOTALL)
    return matches


def parse_bounding_boxes(text):
    text = text.strip()
    # Remove leading and trailing brackets
    if text.startswith("["):
        text = text[1:]
    if text.endswith("]"):
        text = text[:-1]
    # Split by '], [' to get individual bounding boxes
    box_strings = re.split(r'\],?\s*\[', text)
    bounding_boxes = []
    for box_str in box_strings:
        try:
            # Remove any remaining brackets
            box_str = box_str.replace("[", "").replace("]", "")
            # Split by comma or space and convert to numbers
            box = [int(num.strip()) for num in re.split(r'[,\s]+', box_str) if num.strip()]
            bounding_boxes.append(box)
        except ValueError:
            # Handle the case where conversion to int fails
            print(f"Failed to parse bounding box: {box_str}")
    return bounding_boxes


def markdown_text_to_structured_metadata(markdown_text: str) -> Sequence[dict]:
    matches = list(re_match(markdown_text))
    structured = []
    if not matches:
        structured.append(
            {
                "type": "text",
                "content": markdown_text
            }
        )
    if matches and matches[0].start() != 0:
        structured.append(
            {
                "type": "text",
                "content": markdown_text[: matches[0].start()]
            }
        )
    for i, m in enumerate(matches):
        type = m.group(2)
        bboxes = parse_bounding_boxes(m.group(3))
        # We want all the text after this match until the next match
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown_text)
        content = markdown_text[start:end].strip()
        structured.append(
            {
                "type": type,
                "bboxes": bboxes,
                "content": content
            }
        )
    return structured


def has_sentence_end(s):
    s = s.strip()
    enders = ['.', '?', '!', '.)', '?)', '!)', '...']
    return any(s.endswith(ender) for ender in enders)


def drop_empty_blocks(structured_metadata: Sequence[dict]) -> Sequence[dict]:
    return [item for item in structured_metadata if item["content"].strip()]


def fix_dangling_sentences(structured_metadata: Sequence[dict]) -> Sequence[dict]:
    fixed = []
    for item in structured_metadata:
        if len(fixed) == 0:
            fixed.append(item)
            continue

        last_item = fixed[-1]
        are_both_text = last_item["type"] == "text" and item["type"] == "text"
        if are_both_text and not has_sentence_end(last_item["content"]):
            # Merge the two
            last_item["content"] += " " + item["content"]
            last_item["bboxes"] = last_item.get("bboxes", []) + item.get("bboxes", [])
        else:
            fixed.append(item)
    return fixed


def markdown_with_metadata_to_plain_markdown(markdown_with_metadata: Sequence[dict]) -> str:
    # Remove all sections with no content
    markdown_with_metadata = [item for item in markdown_with_metadata if item["content"].strip()]

    # Merge consecutive text sections into one if they are adjacent and the first does not end with a sentence end
    merged = []
    for item in markdown_with_metadata:
        if merged and item["type"] == "text" and merged[-1]["type"] == "text" and not has_sentence_end(merged[-1]["content"]):
            merged[-1]["content"] += " " + item["content"]
        else:
            merged.append(item)
    
    plain_markdown = ""
    for item in merged:
        content = item["content"]
        plain_markdown += content + "\n\n"
    return plain_markdown.strip()


def _parse_title(text: str) -> tuple[int | None, str]:
    """
    Parse a markdown-style header line. Returns (level, cleaned_title_text).
    If no leading hashes are found, returns (None, stripped_text).
    """
    line = text.strip().splitlines()[0]
    match = re.match(r"^(#+)\s*(.+)$", line)
    if not match:
        return None, line
    hashes, title_text = match.groups()
    return len(hashes), title_text.strip()


def annotate_headers(blocks: Sequence[dict], header_stack: list[tuple[int, str]] | None = None):
    """
    Walk through structured blocks, maintaining a stack of titles to annotate each block
    with the headers it falls under.
    """
    stack = list(header_stack or [])
    annotated = []
    for item in blocks:
        item = dict(item)
        if "title" in item.get("type", ""):
            level, title_text = _parse_title(item.get("content", ""))
            if level is not None:
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, title_text))
            item["headers"] = list(stack)
        else:
            item["headers"] = list(stack)
        annotated.append(item)
    return annotated, stack


def merge_page_boundaries(pages: list[Sequence[dict]]) -> list[Sequence[dict]]:
    """
    Merge trailing/leading text blocks across pages when the previous page ends
    mid-sentence.
    """
    merged: list[Sequence[dict]] = []
    for page_data in pages:
        page_data = list(page_data) if page_data else []
        if merged and page_data:
            last_block = merged[-1][-1] if merged[-1] else None
            first_block = page_data[0]
            if (
                last_block
                and first_block
                and last_block["type"] == "text"
                and first_block["type"] == "text"
                and not has_sentence_end(last_block["content"])
            ):
                last_block["content"] += " " + first_block["content"]
                last_block["bboxes"] = last_block.get("bboxes", []) + first_block.get("bboxes", [])
                page_data = page_data[1:]
        merged.append(page_data)
    return merged


async def image_to_deepseek_ocr(ocr_client, image_path: Path) -> str:
    encoded_image = encode_image(image_path)
    image_url = f"data:image/png;base64,{encoded_image}"

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                },
                {
                    "type": "text",
                    "text": "<|grounding|>Convert the document to markdown."
                }
            ]
        }
    ]

    response = await ocr_client.chat.completions.create(
        model="deepseek-ai/DeepSeek-OCR-2",
        messages=messages,
        max_tokens=2048,
        temperature=0.0,
        extra_body={
            "skip_special_tokens": False,
            # args used to control custom logits processor
            "vllm_xargs": {
                "ngram_size": 30,
                "window_size": 90,
                # whitelist: <td>, </td>
                "whitelist_token_ids": [128821, 128822],
            },
        },
    )
    return response.choices[0].message.content


async def image_to_markdown_structured(ocr_client, image_path: Path) -> Sequence[dict]:
    markdown_text = await image_to_deepseek_ocr(ocr_client, image_path)
    structured_metadata = markdown_text_to_structured_metadata(markdown_text)
    # markdown = markdown_with_metadata_to_plain_markdown(structured_metadata)
    return structured_metadata


def page_image_path(page: Page):
    page_base = (page.gp["path"] / page.document_key / f"{page.page_num:04d}")
    image_path = page_base.with_suffix(".png")
    return image_path


async def write_fixed_structure_and_page_markdown(gp: GamePackage, document_key: str) -> None:
    fixed_pages = []
    async for page in get_pages_iter(gp, document_key):
        structured = await page_structured(page)
        fixed_pages.append(fix_dangling_sentences(structured))

    merged_pages = merge_page_boundaries(fixed_pages)
    annotated_pages = []
    header_stack: list[tuple[int, str]] = []
    for merged_page in merged_pages:
        annotated, header_stack = annotate_headers(merged_page, header_stack)
        annotated_pages.append(annotated)
    
    page_annotated_page_iter = amap(
        lambda page, annotated_page: (page, annotated_page),
        get_pages_iter(gp, document_key),
        to_async_iter(annotated_pages)
    )

    page_markdown_strings = []
    write_tasks = []
    async for page, annotated_page in page_annotated_page_iter:
        markdown = markdown_with_metadata_to_plain_markdown(annotated_page)
        page_markdown_strings.append(markdown)
        write_tasks.append(
            asyncio.create_task(
                aspit_json(annotated_page, page_structured_fixed_path(page))
            )
        )
        write_tasks.append(
            asyncio.create_task(aspit(markdown, page_md_path(page)))
        )


    output_subdir = gp["path"] / document_key
    output_path = output_subdir / "document.md"
    merged_markdown = "\n\n".join(page_markdown_strings).strip()
    write_tasks.append(aspit(merged_markdown, output_path))
    await asyncio.gather(*write_tasks)


async def image_to_page_number(ocr_client: AsyncOpenAI, image_path: Path) -> str | None:
    encoded_image = encode_image(image_path)
    image_url = f"data:image/png;base64,{encoded_image}"

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                },
                {
                    "type": "text",
                    "text": '''请按下列JSON格式输出图中信息:
{
    "page_number": ""
}'''
                }
            ]
        }
    ]

    response = await ocr_client.chat.completions.create(
        model="glm-ocr",
        messages=messages,
        max_tokens=2048,
        temperature=0.0,
    )
    text = response.choices[0].message.content

    # Clean the text, sometimes it puts fences around it
    if text is None:
        text = ""
    text = text.strip().strip("```json").strip("```").strip()

    if not text:
        logger.warning("OCR response had no text", image_path=image_path)
        return None

    try:
        data = json.loads(text)
    except:
        logger.exception("Failed to parse OCR response as JSON", text=text)
        return None
    return data.get("page_number") or None


_ROMAN_VALUES = [
    ("xl", 40), ("xxxix", 39), ("xxxviii", 38), ("xxxvii", 37), ("xxxvi", 36),
    ("xxxv", 35), ("xxxiv", 34), ("xxxiii", 33), ("xxxii", 32), ("xxxi", 31),
    ("xxx", 30), ("xxix", 29), ("xxviii", 28), ("xxvii", 27), ("xxvi", 26),
    ("xxv", 25), ("xxiv", 24), ("xxiii", 23), ("xxii", 22), ("xxi", 21),
    ("xx", 20), ("xix", 19), ("xviii", 18), ("xvii", 17), ("xvi", 16),
    ("xv", 15), ("xiv", 14), ("xiii", 13), ("xii", 12), ("xi", 11),
    ("x", 10), ("ix", 9), ("viii", 8), ("vii", 7), ("vi", 6),
    ("v", 5), ("iv", 4), ("iii", 3), ("ii", 2), ("i", 1),
]

_INT_TO_ROMAN = {v: r for r, v in _ROMAN_VALUES}


def _roman_to_int(s: str) -> int | None:
    """Convert a lowercase roman numeral string to an integer, or None if invalid."""
    s = s.strip().lower()
    if not s or not re.fullmatch(r"[ivxlcdm]+", s):
        return None
    for roman, val in _ROMAN_VALUES:
        if s == roman:
            return val
    return None


def _int_to_roman(n: int) -> str | None:
    """Convert a positive integer to a lowercase roman numeral, or None if out of range."""
    return _INT_TO_ROMAN.get(n)


def _parse_page_number(s: str) -> tuple[int | None, str | None]:
    """
    Parse a page number string into (int_value, format_type).
    format_type is "arabic", "roman", or None if unparseable/empty.
    """
    s = s.strip()
    if not s:
        return None, None
    # Try arabic integer
    try:
        return int(s), "arabic"
    except ValueError:
        pass
    # Try roman numeral
    val = _roman_to_int(s)
    if val is not None:
        return val, "roman"
    return None, None


def _format_page_number(n: int, fmt: str) -> str:
    """Convert an integer to a page number string in the given format."""
    if fmt == "roman":
        roman = _int_to_roman(n)
        if roman is not None:
            return roman
    # Fall back to arabic (also used for negative numbers even in roman sections)
    return str(n)


def _remove_outliers(anchors: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    """
    Remove anchors whose values are inconsistent with the majority.
    Two anchors "agree" if their value difference equals their index difference.
    Anchors that agree with no others are outliers.
    """
    if len(anchors) <= 1:
        return list(anchors)

    agreement_counts = [0] * len(anchors)
    for i in range(len(anchors)):
        for j in range(i + 1, len(anchors)):
            idx_diff = anchors[j][0] - anchors[i][0]
            val_diff = anchors[j][1] - anchors[i][1]
            if idx_diff == val_diff:
                agreement_counts[i] += 1
                agreement_counts[j] += 1

    if max(agreement_counts) == 0:
        return list(anchors)

    reliable = [a for a, count in zip(anchors, agreement_counts) if count > 0]
    return reliable if reliable else list(anchors)


def fixup_page_number_sequence(page_numbers: list[str]) -> list[str]:
    """
    This function takes a sequence of page numbers (as strings) and fills in any
    missing page numbers by looking for patterns in the existing page numbers.
    For example, if it sees ["1", "2", "", "4"], it can infer that the missing
    page number is "3". It should be able to handle simple cases of missing page
    numbers, but does not need to be perfect.

    This function will also fix any single page numbers that are clearly wrong
    based on the surrounding page numbers. For example, if it sees ["1", "2",
    "100", "4"], it can infer that "100" is likely a misread "3" and fix it.

    Numbering formats may vary, for example "i", "ii", "iii", or "1", "2", "3".
    Multiple formats may appear in sequence (e.g. roman front matter then arabic
    content). Each format segment is processed independently.

    Will add negative page numbers if needed for initial unnumbered pages.

    The list length is never changed - each element corresponds to a physical page.
    """
    n = len(page_numbers)
    if n <= 1:
        return list(page_numbers)

    # Step 1: Parse all entries into anchors
    anchors = []  # list of (index, int_value, format)
    for i, pn in enumerate(page_numbers):
        val, fmt = _parse_page_number(pn)
        if val is not None:
            anchors.append((i, val, fmt))

    if not anchors:
        # No parseable page numbers - assume pages start at 1
        return [str(i + 1) for i in range(n)]

    # Step 2: Detect format segments (contiguous runs of the same format)
    segments = []  # list of (format, [anchors...])
    cur_fmt = anchors[0][2]
    cur_anchors = [anchors[0]]
    for anchor in anchors[1:]:
        if anchor[2] == cur_fmt:
            cur_anchors.append(anchor)
        else:
            segments.append((cur_fmt, cur_anchors))
            cur_fmt = anchor[2]
            cur_anchors = [anchor]
    segments.append((cur_fmt, cur_anchors))

    # Step 3: Determine index ranges for each segment
    # Blanks between segments are assigned to the preceding segment.
    segment_ranges = []  # list of (start, end, format, anchors)
    for seg_idx, (fmt, seg_anchors) in enumerate(segments):
        start = 0 if seg_idx == 0 else seg_anchors[0][0]
        if seg_idx == len(segments) - 1:
            end = n - 1
        else:
            end = segments[seg_idx + 1][1][0][0] - 1
        segment_ranges.append((start, end, fmt, seg_anchors))

    # Step 4: Within each segment, remove outliers and interpolate blanks
    result = list(page_numbers)
    for start, end, fmt, seg_anchors in segment_ranges:
        reliable = _remove_outliers(seg_anchors)
        if not reliable:
            continue

        reliable_positions = {idx for idx, _, _ in reliable}

        for i in range(start, end + 1):
            if i in reliable_positions:
                continue
            nearest_idx, nearest_val, _ = min(reliable, key=lambda a: abs(a[0] - i))
            expected_val = nearest_val + (i - nearest_idx)
            result[i] = _format_page_number(expected_val, fmt)

    return result


@dataclass
class PageNumberFixUpJob:
    path: Path
    _gp: GamePackage | None = None

    @property
    def gp(self) -> GamePackage:
        # We lazy load this as it may not exist until init_game_package
        # is called
        if self._gp is None:
            self._gp = load_game_package(self.path)
        return self._gp

    async def run(self):

        async def fixup_page_numbers_for_document(document_key: str):
            page_number_paths = []
            fixed_page_number_paths = []
            async for page in get_pages_iter(self.gp, document_key):
                page_number_paths.append(page_number_raw_path(page))
                fixed_page_number_paths.append(page_number_path(page))

            page_numbers = []
            for path in page_number_paths:
                if not path.exists():
                    logger.warning("Page number file does not exist", path=path)
                    page_numbers.append("")
                    continue
                text = await aslurp(path)
                page_numbers.append(text.strip())

            fixed_page_numbers = fixup_page_number_sequence(page_numbers)

            write_tasks = []
            for path, fixed in zip(fixed_page_number_paths, fixed_page_numbers):
                write_tasks.append(aspit(fixed, path))
            await asyncio.gather(*write_tasks)

        tasks = []
        for document_key in document_keys(self.gp):
            tasks.append(fixup_page_numbers_for_document(document_key))
        await asyncio.gather(*tasks)


@dataclass
class PageNumberOcrJob:
    path: Path
    ocr_client: AsyncOpenAI
    max_ocr_workers: int = 2
    _gp: GamePackage | None = None

    @property
    def gp(self) -> GamePackage:
        # We lazy load this as it may not exist until init_game_package
        # is called
        if self._gp is None:
            self._gp = load_game_package(self.path)
        return self._gp

    async def run(self):
        tasks = []
        sem = asyncio.Semaphore(self.max_ocr_workers)
        async def ocr_and_write_page_number(page: Page):
            async with sem:
                page_number_text = await image_to_page_number(
                    self.ocr_client, 
                    page_image_path(page)
                )

                if page_number_text is None:
                    page_number_text = ""
                await aspit(page_number_text, page_number_raw_path(page))

        for document_key in document_keys(self.gp):
            async for page in get_pages_iter(self.gp, document_key):
                tasks.append(
                    asyncio.create_task(ocr_and_write_page_number(page))
                )

        await asyncio.gather(*tasks)


@dataclass
class OcrJob:
    path: Path
    ocr_client: AsyncOpenAI
    _gp: GamePackage | None = None
    max_size: int | None = 2000
    max_ocr_workers: int = 2
    max_image_workers: int = 4
    page_queue_size: int = 4
    pdf_page_chunk: int = 4
    rulebook_concurrency: int = 2
    chunk_size: int = 500
    chunk_overlap: int = 50
    child_chunk_size: int = 125
    child_chunk_overlap: int = 12

    @property
    def gp(self) -> GamePackage:
        # We lazy load this as it may not exist until init_game_pacakge
        # is called
        if self._gp is None:
            self._gp = load_game_package(self.path)
        return self._gp

    async def run(self):
        # Load PDF images
        pages_and_images = self.all_page_and_pdf_images_iter()

        # Resize images and save
        pages_and_images = amap(self.postprocess_and_save, pages_and_images)

        # Enqueue pages as they are saved
        page_image_ready_queue: asyncio.Queue = asyncio.Queue(maxsize=self.page_queue_size)    
        loader_task = asyncio.create_task(
            sink_into_queue(page_image_ready_queue, pages_and_images)
        )

        # Run OCR on the images. Some of the pages may take longer to generate
        # output for than others, so we don't process them strictly in order.
        await pipeline(
            sink=None,
            xf=xf_amap(self.process_pdf_image),
            source=page_image_ready_queue,
            concurrency=self.max_ocr_workers,
        )
        await loader_task

        # Now we cleanup any dangling sentences across page boundaries and dump the
        # the fixed structured JSON files. This needs to be done in order.
        tasks = []
        for document_key in document_keys(self.gp):
            tasks.append(
                asyncio.create_task(
                    write_fixed_structure_and_page_markdown(self.gp, document_key)
                )
            )
        await asyncio.gather(*tasks)


    async def process_pdf_image(self, page: Page):
        image_path = page_image_path(page)
        markdown_text = await image_to_deepseek_ocr(self.ocr_client, image_path)
        structured = markdown_text_to_structured_metadata(markdown_text)
        structured = drop_empty_blocks(structured)
        await aspit_json(structured, page_structured_path(page))
        return page


    def all_page_and_pdf_images_iter(self) -> AsyncIterator[Tuple[Page, Image.Image]]:
        pdf_image_iterables = []
        for rulebook in self.gp["rulebooks"]:
            pdf_path = (self.gp["path"] / "raw_documents" / rulebook["path"])

            # Save the pdf images
            pages_and_images = page_and_image_iter(self.gp, rulebook["document_key"], pdf_path)
            pdf_image_iterables.append(pages_and_images)
        pdf_images = achain(*pdf_image_iterables)
        return pdf_images


    async def postprocess_and_save(self, page_and_image) -> Page:
        page, image = page_and_image
        image_path = page_image_path(page)
        logger.info("Saving page image", image_path=image_path, document_key=page.document_key)
        await asyncio.to_thread(
            postprocess_and_save, image, image_path, self.max_size
        )
        return page
