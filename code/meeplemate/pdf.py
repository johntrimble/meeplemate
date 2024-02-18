from pathlib import Path
from typing import List
from langchain.schema import Document
from langchain.document_loaders import UnstructuredPDFLoader

def has_sentence_end(s):
    s = s.strip()
    enders = ['.', '?', '!', '.)', '?)', '!)', '...']
    return any(s.endswith(ender) for ender in enders)


def fix_dangling_sentences(elements):
    """Fixes sentences that are split across multiple elements"""
    sequence = []
    for el in elements:
        previous = sequence[-1] if sequence else None
        if el.metadata["category"] == "NarrativeText" or has_sentence_end(el.page_content):
            if previous and previous.metadata["category"] == "NarrativeText":
                if not has_sentence_end(previous.page_content):
                    sequence[-1].page_content += " " + el.page_content
                    continue
        sequence.append(el)
    return sequence


def organize_into_sections(elements):
    sections = [[]]
    for element in elements:
        if element.metadata["category"] == "Title":
            sections.append([element])
        else:
            sections[-1].append(element)
    sections = list(filter(None, sections))
    return sections


def sections_to_pages(sections):
    pages = []
    current_page = -1
    for section in sections:
        if section:
            page = section[0].metadata["page_number"]
            if page != current_page:
                pages.append([section])
                current_page = page
            else:
                pages[-1].append(section)
    return pages


def pages_to_documents(pages):
    documents = []
    for page in pages:
        if page:
            page_content = build_string_from_sections(page)
            metadata = page[0][0].metadata
            page_number = metadata.get('page_number', -1)
            source = metadata.get('source', None)
            documents.append(Document(page_content=page_content, metadata={"page_number": page_number, "source": source}))
    return documents


def build_string_from_sections(sections):
    return "\n\n".join(
        "\n".join(element.page_content for element in section)
        for section in sections
    )


def fix_document_metadata(document):
    metadata = document.metadata
    for k in metadata:
        if not isinstance(metadata[k], (str, int, float, bool)):
            metadata[k] = str(metadata[k])
    return document


def parse_pdf(path:Path) -> List[Document]:
    # To parse the PDFs, there are three strategies available: "fast", "hi_res", and
    # "ocr_only". For the PDFs used here, "fast" retrieves a bunch of duplicate text
    # in the wrong order. "hi_res" doesn't handle columns of text well and produces
    # incoherent results. "ocr_only" seems to work reasonably well in this case.
    loader = UnstructuredPDFLoader(path, mode="elements", strategy="ocr_only")
    result = loader.load()
    result = fix_dangling_sentences(result)
    sections = organize_into_sections(result)
    pages = sections_to_pages(sections)
    # text = build_string_from_sections(sections)
    # metadata = loader._get_metadata()
    documents = pages_to_documents(pages)
    documents = [fix_document_metadata(document) for document in documents]
    return documents
