import structlog

from dataclasses import dataclass
from pathlib import Path

from meeplemate.ingest.gamepackage import (
    GamePackage,
    layout_for,
    get_page_metadata,
    get_page_metadata_path,
    get_pages_iter,
    load_game_package,
    page_md,
    page_number_path,
)
from meeplemate.util import aslurp, aspit_yaml

logger = structlog.get_logger(__name__)


@dataclass
class DocumentMetadataJobJob:
    path: Path

    @property
    def gp(self) -> GamePackage:
        if not getattr(self, "_gp", None):
            self._gp = load_game_package(self.path)
        return self._gp

    async def run(self) -> None:
        for rulebook in self.gp["rulebooks"]:
            document_key = rulebook["document_key"]
            document_path = layout_for(self.gp).document_md(document_key)
            document_text = await aslurp(document_path)

            search_offset = 0
            async for page in get_pages_iter(self.gp, document_key):
                md = await page_md(page)
                start_index = document_text.find(md, search_offset)
                if start_index == -1:
                    logger.warning(
                        "Page markdown not found in document.md",
                        document_key=document_key,
                        page_num=page.page_num,
                    )
                    continue
                end_index = start_index + len(md)

                metadata = get_page_metadata(page)
                metadata["start_index"] = start_index
                metadata["end_index"] = end_index

                # page-number-fixup writes a file for every page, including
                # rulebooks with an explicit page_one_offset, so there is no
                # offset branch to take here any more.
                metadata["page_num"] = (await aslurp(page_number_path(page))).strip()

                await aspit_yaml(metadata, get_page_metadata_path(page))
                logger.info(
                    "Wrote page metadata",
                    document_key=document_key,
                    page_num=page.page_num,
                    start_index=start_index,
                    end_index=end_index,
                )
                search_offset = end_index
