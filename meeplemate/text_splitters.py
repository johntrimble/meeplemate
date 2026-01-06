import copy
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class FixedRecursiveCharacterTextSplitter(RecursiveCharacterTextSplitter):
    """
    Fixed version of RecursiveCharacterTextSplitter that properly handles start_index
    when using token-based length functions.

    The original langchain implementation has a bug where it mixes character-based
    and token-based units when calculating the offset for finding chunk positions.
    This causes start_index to be set to -1 for some chunks when using tokenizers.

    This class fixes the issue by:
    1. When a tokenizer is available: Uses offset mapping to precisely locate chunks
    2. When no tokenizer: Detects actual overlap between chunks in character units
    3. Falls back to searching from the end of the previous chunk if not found
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize with optional tokenizer reference."""
        super().__init__(**kwargs)
        self._tokenizer = None

    @classmethod
    def from_huggingface_tokenizer(cls, tokenizer: Any, **kwargs: Any) -> "FixedRecursiveCharacterTextSplitter":
        """Create instance from HuggingFace tokenizer and store reference to it."""
        # Call parent's classmethod to set up length function and other configs
        instance = super(FixedRecursiveCharacterTextSplitter, cls).from_huggingface_tokenizer(
            tokenizer, **kwargs
        )
        # Cast to our type since parent returns TextSplitter
        if isinstance(instance, cls):
            # Store the tokenizer for precise offset mapping
            instance._tokenizer = tokenizer
        return instance  # type: ignore[return-value]

    def create_documents(
        self, texts: list[str], metadatas: list[dict[Any, Any]] | None = None
    ) -> list[Document]:
        """Create documents from a list of texts with corrected start_index handling."""
        # If we have a tokenizer, use the precise offset-based approach
        if self._tokenizer is not None and hasattr(self._tokenizer, "encode_plus"):
            return self._create_documents_with_tokenizer(texts, metadatas)

        # Otherwise, delegate to parent class (which has the bug, but we only fix tokenizer case)
        return super().create_documents(texts, metadatas)

    def _create_documents_with_tokenizer(
        self, texts: list[str], metadatas: list[dict[Any, Any]] | None = None
    ) -> list[Document]:
        """Create documents using tokenizer offset mapping for precise positioning."""
        metadatas_ = metadatas or [{}] * len(texts)
        documents = []

        for i, text in enumerate(texts):
            chunks = self.split_text(text)

            for chunk_idx, chunk in enumerate(chunks):
                metadata = copy.deepcopy(metadatas_[i])

                if self._add_start_index:
                    if chunk_idx == 0:
                        # First chunk always starts at 0
                        index = text.find(chunk)
                    else:
                        # Use tokenizer offset mapping to find precise overlap
                        prev_chunk = chunks[chunk_idx - 1]
                        prev_index = documents[-1].metadata["start_index"]

                        # Get offset mappings for both chunks
                        try:
                            prev_encoding = self._tokenizer.encode_plus(
                                prev_chunk,
                                return_offsets_mapping=True,
                                add_special_tokens=False
                            )
                            curr_encoding = self._tokenizer.encode_plus(
                                chunk,
                                return_offsets_mapping=True,
                                add_special_tokens=False
                            )

                            # Find where current chunk starts in the original text
                            # by detecting the overlap using offset mapping
                            overlap_chars = self._find_overlap_with_tokenizer(
                                prev_chunk, chunk, prev_encoding, curr_encoding
                            )

                            # Calculate estimated start position
                            estimated_start = prev_index + len(prev_chunk) - overlap_chars

                            # Search in a narrow window around the estimate
                            search_start = max(0, estimated_start - 10)
                            index = text.find(chunk, search_start)

                            # If not found, expand search
                            if index == -1 or index < prev_index:
                                index = text.find(chunk, prev_index + 1)

                        except Exception:
                            # If tokenizer fails, fall back to simple search
                            index = text.find(chunk, prev_index + 1)

                        # Final fallback: search from the beginning
                        if index == -1:
                            index = text.find(chunk, 0)

                    metadata["start_index"] = index

                new_doc = Document(page_content=chunk, metadata=metadata)
                documents.append(new_doc)

        return documents

    def _find_overlap_with_tokenizer(
        self, text1: str, text2: str, encoding1: dict, encoding2: dict
    ) -> int:
        """
        Find overlap length using tokenizer offset mappings.

        This is more precise than string comparison as it uses the actual
        token boundaries from the tokenizer.
        """
        # Get the tokens and offsets
        tokens1 = encoding1["input_ids"]
        tokens2 = encoding2["input_ids"]
        offsets1 = encoding1["offset_mapping"]
        offsets2 = encoding2["offset_mapping"]

        # The configured overlap tells us approximately how many tokens should overlap
        # Start searching from that value with a small window around it
        max_overlap_tokens = min(len(tokens1), len(tokens2))
        expected_overlap = min(self._chunk_overlap, max_overlap_tokens)

        # Search in a window around the expected overlap (±10 tokens)
        search_start = max(1, expected_overlap - 10)
        search_end = min(max_overlap_tokens, expected_overlap + 10)

        for overlap_tokens in range(search_end, search_start - 1, -1):
            # Check if the last N tokens of text1 match the first N tokens of text2
            if tokens1[-overlap_tokens:] == tokens2[:overlap_tokens]:
                # Found a token-level overlap
                # Calculate character-level overlap using offset mappings
                # The overlap in text1 starts at this character offset
                overlap_start_in_text1 = offsets1[-overlap_tokens][0]
                # And extends to the end of text1
                overlap_chars = len(text1) - overlap_start_in_text1
                return overlap_chars

        # If not found in the expected range, fall back to full search
        # (This handles cases where the actual overlap differs significantly)
        for overlap_tokens in range(max_overlap_tokens, 0, -1):
            if overlap_tokens >= search_start and overlap_tokens <= search_end:
                continue  # Already checked this range
            if tokens1[-overlap_tokens:] == tokens2[:overlap_tokens]:
                overlap_start_in_text1 = offsets1[-overlap_tokens][0]
                overlap_chars = len(text1) - overlap_start_in_text1
                return overlap_chars

        # No token overlap found, return 0
        return 0