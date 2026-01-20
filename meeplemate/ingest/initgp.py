from pathlib import Path
import base64

from uuid_utils import uuid7
import yaml

from meeplemate.ingest.gamepackage import Manifest, document_keys, load_manifest
from meeplemate.util import aspit_yaml, spit_yaml

def get_page_count(pdf_path: Path) -> int:
    from pdf2image import pdfinfo_from_path

    info = pdfinfo_from_path(str(pdf_path))
    return int(info.get("Pages", 0))


class InitGamePackageJob:
    def __init__(self, input_dir: Path, output_dir: Path):
        self.input_dir = input_dir
        self.output_dir = output_dir
    
    def copy_over_documents(self, manifest: Manifest) -> None:
        raw_documents_dir = self.output_dir / "raw_documents"
        raw_documents_dir.mkdir(exist_ok=True)
        for rulebook in manifest["rulebooks"]:
            relative_path = Path(rulebook["path"])
            source_path = (self.input_dir / relative_path).resolve()
            target_path = (raw_documents_dir / relative_path).resolve()
            # Ensure target directory exists
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if source_path.resolve() == target_path.resolve():
                raise ValueError("Source and target paths for raw document are the same.")
            # Copy the file
            target_path.write_bytes(source_path.read_bytes())

    def load_manifest_from_source(self) -> Manifest:
        rulebooks_yaml_path = self.input_dir / "rulebooks.yaml"
        return yaml.safe_load(rulebooks_yaml_path.read_text())
        # target_path = self.output_dir / "rulebooks.yaml"
        # target_path.write_bytes(rulebooks_yaml_path.read_bytes())

    def update_manifest_document_keys(self, manifest: Manifest) -> None:
        rulebooks = manifest["rulebooks"]
        for rulebook in rulebooks:
            if "document_key" not in rulebook:
                path = rulebook["path"]
                document_key = base64.b64encode(path.encode()).decode()
                rulebook["document_key"] = document_key

    def create_document_directories(self, manifest: Manifest) -> None:
        # Ensure all document directories exist
        for document_key in document_keys(manifest):
            document_dir = self.output_dir / document_key
            if not document_dir.exists():
                document_dir.mkdir()
    
    def update_page_counts(self, manifest: Manifest) -> None:
        rulebooks = manifest["rulebooks"]
        for rulebook in rulebooks:
            relative_path = Path(rulebook["path"])
            pdf_path = (self.output_dir / "raw_documents" / relative_path).resolve()
            rulebook["page_count"] = get_page_count(pdf_path)
    
    def maybe_add_game_version(self, manifest: Manifest) -> None:
        if "game_version" not in manifest:
            game_version = str(uuid7())
            manifest["game_version"] = game_version
        
    async def run(self):
        # Make sure the output directory exists
        self.output_dir.mkdir(exist_ok=True)

        # Load the manifest
        manifest = self.load_manifest_from_source()

        # Copy over the raw documents
        self.copy_over_documents(manifest)

        # Add document keys if needed
        self.update_manifest_document_keys(manifest)

        # Ensure document directories exist
        self.create_document_directories(manifest)

        # Add page counts
        self.update_page_counts(manifest)

        # Add game_version if missing
        self.maybe_add_game_version(manifest)

        # Write out the updated manifest
        manifest_path = self.output_dir / "rulebooks.yaml"
        await aspit_yaml(manifest, manifest_path)
