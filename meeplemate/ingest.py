import base64
import copy
from glob import glob
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Coroutine, Protocol, Sequence, Tuple, cast

import asyncio
import click
import yaml
from openai import AsyncOpenAI
from PIL import Image
from typing import NotRequired, TypedDict

from langchain_core.retrievers import BaseRetriever

from meeplemate.component_system import System, afactory, factory, subsystem
from meeplemate.config import Config, create_app_system
from langchain_core.documents.base import Document

from langchain_core.stores import BaseStore

from langchain_core.language_models.chat_models import BaseChatModel

from meeplemate.util import amap, apairwise, arepeat, atee, achain, atakewhile, aslurp, aslurp_json, aspit_yaml, sem_guard, pipeline, aspit_json, xf_amap, to_async_iter, aspit, sink_into_queue
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_classic.output_parsers.regex import RegexParser

import structlog

logger = structlog.get_logger(__name__)

class RulebookDescriptor(TypedDict):
    name: str
    path: str
    url: str
    document_key: str
    strategy: NotRequired[str]
    page_count: int


class Manifest(TypedDict):
    name: str
    game_id: str
    rulebooks: Sequence[RulebookDescriptor]


class GamePackage(Manifest, TypedDict):
    path: Path


@dataclass
class Page:
    gp: GamePackage
    page_num: int
    document_key: str


@dataclass
class PageResult:
    page_num: int
    image_path: Path
    structured_path: Path
    structured: Sequence[dict]


def document_keys(manifest: Manifest) -> Sequence[str]:
    document_keys: list[str] = []
    for rulebook in manifest["rulebooks"]:
        document_keys.append(rulebook["document_key"])
    return document_keys


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
        model="deepseek-ai/DeepSeek-OCR",
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


def load_manifest(target_dir: Path) -> Manifest:
    manifest_path = target_dir / "rulebooks.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    return manifest


def load_game_package(target_dir: Path) -> GamePackage:
    manifest = load_manifest(target_dir)
    if "path" in manifest:
        del manifest["path"]
    gp: GamePackage = GamePackage(**manifest, path=target_dir)
    return gp


def page_image_path(page):
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


@dataclass
class IngestJob:
    input_dir: Path
    output_dir: Path
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
            self._gp = load_game_package(self.output_dir)
        return self._gp


    async def run(self):
        # Initialize the game package directory
        await self.init_game_package()

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


    async def init_game_package(self):
        # Load the partial manifest from the source directory
        rulebooks_yaml_path = self.input_dir / "rulebooks.yaml"
        manifest = yaml.safe_load(rulebooks_yaml_path.read_text())

        # Update each rulebook descriptor with a document_key if needed
        rulebooks = manifest["rulebooks"]
        for rulebook in rulebooks:
            if "document_key" not in rulebook:
                path = rulebook["path"]
                document_key = base64.b64encode(path.encode()).decode()
                rulebook["document_key"] = document_key
    
        # Update each rulebook descriptor with a page count
        rulebooks = manifest["rulebooks"]
        for rulebook in rulebooks:
            relative_path = Path(rulebook["path"])
            pdf_path = (self.input_dir / relative_path).resolve()
            rulebook["page_count"] = get_page_count(pdf_path)

        # Determine path of the manifest
        manifest_target_path = self.output_dir / "rulebooks.yaml"

        # Ensure output directory exists
        if not self.output_dir.exists():
            self.output_dir.mkdir()
        
        # Ensure all document directories exist
        for document_key in document_keys(manifest):
            document_dir = self.output_dir / document_key
            if not document_dir.exists():
                document_dir.mkdir()

        # Write the manifest
        await aspit_yaml(manifest, manifest_target_path)
    

    def all_page_and_pdf_images_iter(self) -> AsyncIterator[Tuple[Page, Image.Image]]:
        pdf_image_iterables = []
        for rulebook in self.gp["rulebooks"]:
            pdf_path = (self.input_dir / rulebook["path"])

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


class ImportDocumentsServices(TypedDict):
    retriever: BaseRetriever


@dataclass
class ImportDocumentsJob:
    gp: GamePackage
    retriever: BaseRetriever
    full_page_store: BaseStore
    game_data_store: BaseStore
    input_dir: Path
    output_dir: Path
    concurrency: int


def get_page_path_bases(gp: GamePackage, doc_key: str|None=None) -> list[Tuple[str, int, Path]]:
    page_paths: list[Tuple[str, int, Path]] = []
    for rulebook in gp["rulebooks"]:
        if doc_key is not None and doc_key != rulebook["document_key"]:
            continue

        directory = (gp["path"] / rulebook["document_key"]).resolve()
        page_pattern = re.compile(r"^(\d+)(\..*)?$")
        paths = glob("*.md", root_dir=directory)
        
        for path in paths:
            match = page_pattern.match(Path(path).stem)
            if match:
                page_num = int(match.group(1))
                path = directory / path
                page_paths.append((rulebook["document_key"], page_num, path.with_suffix("")))

    page_paths.sort(key=lambda x: (x[0], x[1]))
    return page_paths


def page_md_path(page: Page) -> Path:
    page_base = (page.gp["path"] / page.document_key / f"{page.page_num:04d}")
    markdown_path = page_base.with_suffix(".md")
    return markdown_path


def page_structured_path(page: Page) -> Path:
    page_base = (page.gp["path"] / page.document_key / f"{page.page_num:04d}")
    structured_path = page_base.with_suffix(".structured.json")
    return structured_path


def page_structured_fixed_path(page: Page) -> Path:
    page_base = (page.gp["path"] / page.document_key / f"{page.page_num:04d}")
    structured_path = page_base.with_suffix(".structured.fixed.json")
    return structured_path


async def page_md(page: Page) -> str:
    markdown_path = page_md_path(page)
    markdown = await aslurp(markdown_path)
    return markdown


async def page_structured(page: Page) -> Sequence[dict]:
    structured_path = page_structured_path(page)
    structured = await aslurp_json(structured_path)
    return structured


async def page_structured_fixed(page: Page) -> Sequence[dict]:
    structured_path = page_structured_fixed_path(page)
    structured = await aslurp_json(structured_path)
    return structured


def get_page(gp:GamePackage, document_key: str, page_num: int) -> Page:
    page_base = (gp["path"] / document_key / f"{page_num:04d}")
    page_num = int(page_base.stem)
    page = Page(
        gp=gp,
        document_key=document_key,
        page_num=page_num,
    )
    return page


async def get_pages_iter(gp: GamePackage, document_key: str|None = None) -> AsyncIterator[Page]:
    rulebook_descriptors = gp["rulebooks"]
    for descriptor in rulebook_descriptors:
        if document_key is not None and descriptor["document_key"] != document_key:
            continue

        for page_num in range(descriptor["page_count"]):
            page = get_page(gp, descriptor["document_key"], page_num)
            yield page


def get_page_metadata(page: Page) -> dict:
    rulebook = next(
        rb for rb in page.gp["rulebooks"] if rb["document_key"] == page.document_key
    )
    game_id = page.gp["game_id"]
    metadata = {
        "game_name": page.gp["name"],
        "game_id": game_id,
        "rulebook_name": rulebook["name"],
        "document_key": page.document_key,
        "page_num": page.page_num,
    }
    return metadata


async def page_to_document(page: Page) -> Document:
    game_id = page.gp["game_id"]
    metadata = get_page_metadata(page)
    page_key = get_page_id(game_id, page.document_key, page.page_num)
    markdown = await page_md(page)
    return Document(id=page_key, page_content=markdown, metadata=metadata)


def get_page_id(game_id: str, document_key: str, page_num: int) -> str:
    return f"{game_id}#{document_key}#{page_num}"


async def import_game_data(job: ImportDocumentsJob) -> None:
    game_data = copy.deepcopy(job.gp)
    for rulebook in game_data["rulebooks"]:
        rulebook.pop("strategy", None)
        rulebook.pop("path", None)
    
    await job.game_data_store.amset([(game_data["game_id"], game_data)])


async def run_import_documents(job: ImportDocumentsJob) -> None:
    async def process_page(doc_key: str, page_num: int):
        # Get the page content
        page = get_page(job.gp, doc_key, page_num)
        # Get document for the page
        document = await page_to_document(page)
        # Add to retriever which will chunk and store in vector store
        await job.retriever.aadd_documents([document])
        # Also store full page in full page store
        await job.full_page_store.amset([(document.id, document)])

    concurrency = job.concurrency
    sem = asyncio.Semaphore(concurrency)

    async def sem_guard(coro: Coroutine[Any, Any, Any], sem: asyncio.Semaphore) -> Any:
        async with sem:
            return await coro

    doc_page_number_bases = get_page_path_bases(job.gp)

    tasks = []
    for doc_key, page_num, page_base in doc_page_number_bases:
        coro = sem_guard(process_page(doc_key, page_num), sem)
        tasks.append(asyncio.create_task(coro))

    await asyncio.gather(*tasks)
    await import_game_data(job)


async def get_document_page_aiter(gp: GamePackage) -> AsyncIterator[Document]:
    for document_key, page_num, page_base in get_page_path_bases(gp):
        page = get_page(gp, document_key, page_num)
        document = await page_to_document(page)
        yield document

DOCUMENT_TEMPLATE = """\
<document name="{name}" page="{page_number}">
{content}
</document>
"""

SUMMARY_SYSTEM_TEMPLATE = """\
You are an expert at board game design and rule analysis. Your goal is to create a concise and informative summary of the rules for the board game "{game_name}" based on the provided rulebook content. The summary should help players quickly understand the game's objectives, setup, and key mechanics. You will receive a current summary ond one or more rulebook pages that should be used to update and improve the summary.
"""

SUMMARY_TEMPLATE = """\
## Current Summary

<summary>
{current_summary}
</summary>

## Rulebook Pages

{rulebook_pages}

## Instructions

- Read the current summary and the provided rulebook pages carefully.
- Provide a step-by-step analysis of how to update the summary based on the new information from the rulebook pages.
- Identify any important rules, mechanics, or clarifications that are missing from the current summary.
- Update the summary to include these new details while ensuring it remains concise and easy to understand.
- Maintain a neutral and informative tone throughout the summary.
- Ensure that the summary is well-organized, using headings or bullet points if necessary to enhance readability.
- Avoid adding unnecessary details or overly complex explanations; focus on the most relevant information for players.
- Ignore fluff or non-essential content from the rulebook pages that do not contribute to understanding the game's rules.
- The final summary should be comprehensive yet succinct, providing players with a clear understanding of the game's rules.
- If there is no new information to add, retain and output the current summary as is.
- Keep summary length to a maximum of 500 words.
- Output the updated summary in markdown format.

## Formatting

- Output reasoning steps as a bulleted list.
- Output the summary enclosed within <summary> and </summary> tags.
"""

SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SUMMARY_SYSTEM_TEMPLATE),
        ("human", SUMMARY_TEMPLATE),
    ]
)

@dataclass
class GenerateGameReferenceJob:
    output_dir: Path
    gp: GamePackage
    chat_model: BaseChatModel
    tokenizer: Any

    async def run(self):
        # We need to be careful here. We want pairs of consecutive pages to
        # ensure we don't split important context. However, if we just use
        # apairwise, we might not process any pages is there's only one page.

        # Get the page iterator
        page_iter = get_document_page_aiter(self.gp)

        # Convert pages to string content
        document_prompt = PromptTemplate.from_template(
            DOCUMENT_TEMPLATE,
            template_format="f-string",
        )
        def page_to_string(page: Document) -> str:
            s = document_prompt.format(
                name=page.metadata.get("rulebook_name", "Unknown"),
                page_number=page.metadata.get("page_num", -1),
                content=page.page_content,
            )
            return s
        page_iter = amap(page_to_string, page_iter)

        # Have the iterator produce None when exhausted
        page_iter = achain(page_iter, arepeat(None))

        # Now we get pairs
        page_pairs_iter = apairwise(page_iter)

        # Only take pairs while the first element is not None
        # This will give us all non-None pairs for documents with 2+ pages and
        # a single pair of (page, None) for documents with 1 page.
        page_pairs_iter = atakewhile(lambda pair: pair[0] is not None, page_pairs_iter)

        output_parser = RegexParser(
            regex=r".*<summary>([\s\S]*)</summary>.*",
            output_keys=["summary"],
        )

        summary_chain = (
            SUMMARY_PROMPT | self.chat_model | output_parser
        ).with_retry()

        current_summary = ""
        async for page1, page2 in page_pairs_iter:
            assert page1 is not None
            if page2 is None:
                pages_content = page1
            else:
                assert page2 is not None
                pages_content = '\n'.join([page1, page2])
            try:
                output = await summary_chain.ainvoke(
                    {
                        "game_name": self.gp["name"],
                        "current_summary": current_summary,
                        "rulebook_pages": pages_content,
                    }
                )
                current_summary = output["summary"]
                print(f"Updated Summary:\n{current_summary}\n\n\n")
            except:
                logger.exception("Failed to process pages")


EXTRACT_TERMINOLOGY_SYSTEM_TEMPLATE = """\
You are an expert at board game design and rule analysis. Your goal is to extract and define key terminology used in the rules for the board game "{{game_name}}" based on the provided rulebook content. The extracted terminology should help players quickly understand important terms and concepts used in the game's rules. You will receive a current list of terminology definitions and one or more rulebook pages that should be used to update and improve the terminology list. This list will act as both a glossary and an index for the game's rules.
"""

EXTRACT_TERMINOLOGY_TEMPLATE = """\
## Current Terminology

<terminology>
{{current_terminology}}
</terminology>

## Rulebook Pages

{{#documents}}
<document name="{{metadata.rulebook_name}}" page="{{metadata.page_num}}">
{{page_content}}
</document>
{{/documents}}

{{rulebook_pages}}

## Instructions

- Read the current terminology list and the provided rulebook pages carefully.
- Provide a step-by-step analysis of how to update the terminology list based on the new information from the rulebook pages.
- Identify any important terms or concepts that are missing from the current terminology list.
- For each new term, provide a clear and concise definition based on the rulebook content.
- Update the terminology list to include these new definitions while ensuring it remains concise and easy to understand.
- Maintain a neutral and informative tone throughout the terminology list.
- Keep each definition brief and to the point, about one or two sentences.
- Use bullet points for each term and its definition.
- Ignore fluff or non-essential content from the rulebook pages that do not contribute to understanding key terminology.
- The final terminology list should be comprehensive yet succinct, providing players with a clear understanding of important terms used in the game's rules.
- If there is no new terminology to add, retain and output the current terminology list as is.

## Formatting

- Output reasoning steps as a bulleted list.
- Output the terminology list enclosed within <terminology> and </terminology> tags.
"""


EXTRACT_TERMINOLOGY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", EXTRACT_TERMINOLOGY_SYSTEM_TEMPLATE),
        ("human", EXTRACT_TERMINOLOGY_TEMPLATE),
    ]
)


@dataclass
class ExtractTerminologyJob:
    output_dir: Path
    gp: GamePackage
    chat_model: BaseChatModel
    tokenizer: Any
    summary_tag: str = "terminology"

    async def run(self):
        # We need to be careful here. We want pairs of consecutive pages to
        # ensure we don't split important context. However, if we just use
        # apairwise, we might not process any pages is there's only one page.

        # Get the page iterator
        page_iter = get_document_page_aiter(self.gp)

        # Convert pages to dictionaries for use with the prompt template
        page_iter = amap(Document.model_dump, page_iter)

        # Have the iterator produce None when exhausted
        page_iter = achain(page_iter, arepeat(None))

        # Now we get pairs
        page_pairs_iter = apairwise(page_iter)

        # Only take pairs while the first element is not None
        # This will give us all non-None pairs for documents with 2+ pages and
        # a single pair of (page, None) for documents with 1 page.
        page_pairs_iter = atakewhile(lambda pair: pair[0] is not None, page_pairs_iter)

        # Define summary tag regex, being sure to escape any special characters
        # in the tag name
        escaped_tag = re.escape(self.summary_tag)
        regex_pattern = rf".*<{escaped_tag}>([\s\S]*)</{escaped_tag}>.*"

        output_parser = RegexParser(
            regex=regex_pattern,
            output_keys=["summary"],
        )

        summary_chain = (
            EXTRACT_TERMINOLOGY_PROMPT | self.chat_model | output_parser
        ).with_retry()

        current_summary = ""
        async for page1, page2 in page_pairs_iter:
            try:
                output = await summary_chain.ainvoke(
                    {
                        "game_name": self.gp["name"],
                        "current_terminology": current_summary,
                        "documents": [
                            page for page in [page1, page2] if page is not None
                        ],
                    }
                )
                current_summary = output["summary"]
                print(f"Updated Summary:\n{current_summary}\n\n\n")
            except:
                logger.exception("Failed to process pages")


@click.group()
def cli():
    pass


# uv run python -m meeplemate.ingest process --input ./data/rules/munchkin_rules/ --output ./data/ingested/munchkin_rules/

@cli.command()
@click.argument("input", type=Path)
@click.argument("output", type=Path)
def process(input: Path, output: Path):
    settings: Config = Config()
    app_system: System = create_app_system(settings)
    system = subsystem(
        app_system,
        extra_components={
            "ocr_client": (
                factory(AsyncOpenAI)(
                    api_key="EMPTY",
                    base_url="http://vllm-ocr:8000/v1",
                    timeout=3600
                ),
                []
            ),
            "ingest_job": (
                afactory(
                    IngestJob,
                    astart=IngestJob.run
                )(
                    input_dir=input,
                    output_dir=output,
                    max_size=2_000,
                    chunk_size=settings.ingest.chunk_size,
                    chunk_overlap=settings.ingest.chunk_overlap,
                    child_chunk_size=settings.ingest.child_chunk_size,
                    child_chunk_overlap=settings.ingest.child_chunk_overlap,
                ),
                {
                    "ocr_client": "ocr_client"
                }
            )
        }
    )
    async def _run():
        async with system.astart() as services:
            pass
    
    asyncio.run(_run())


@cli.command()
@click.argument("input", type=Path)
@click.argument("output", type=Path)
def import_documents(input: Path, output: Path):
    settings: Config = Config()
    app_system: System = create_app_system(settings)
    system = subsystem(
        app_system,
        extra_components={
            **app_system._components,
            "import_job": (
                factory(ImportDocumentsJob)(
                    input_dir=input,
                    output_dir=output,
                    gp=load_game_package(output),
                    concurrency=100,
                ),
                {
                    "retriever": "retriever",
                    "full_page_store": "full_page_store",
                    "game_data_store": "game_data_store"
                },
            )
        }
    )
    system = System.subsystem(system, names=["import_job", "keyspace_creator"])

    async def _import_documents():
        async with system.astart() as services:
            await run_import_documents(services["import_job"])

    asyncio.run(_import_documents())


@cli.command()
@click.argument("output", type=Path)
def generate_reference(output: Path):
    settings: Config = Config()
    app_system: System = create_app_system(settings)
    system = subsystem(
        app_system,
        extra_components={
            "generate_reference_job": (
                afactory(
                    GenerateGameReferenceJob,
                    astart=GenerateGameReferenceJob.run,
                )(
                    gp=load_game_package(output),
                    output_dir=output,
                ),
                {
                    "chat_model": "chat_model",
                    "tokenizer": "tokenizer",
                }
            )
        },
    )

    async def _run():
        async with system.astart() as services:
            pass
    
    asyncio.run(_run())


@cli.command()
@click.argument("output", type=Path)
def extract_terminology(output: Path):
    settings: Config = Config()
    app_system: System = create_app_system(settings)
    system = subsystem(
        app_system,
        extra_components={
            "extract_terminology_job": (
                afactory(
                    ExtractTerminologyJob,
                    astart=ExtractTerminologyJob.run,
                )(
                    gp=load_game_package(output),
                    output_dir=output,
                ),
                {
                    "chat_model": "chat_model",
                    "tokenizer": "tokenizer",
                }
            )
        },
    )

    async def _run():
        async with system.astart() as services:
            pass
    
    asyncio.run(_run())

if __name__ == "__main__":
    cli()
