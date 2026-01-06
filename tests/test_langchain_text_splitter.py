import pytest
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from meeplemate.text_splitters import FixedRecursiveCharacterTextSplitter
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class FakeTokenizer(PreTrainedTokenizerBase):
    """
    A fake tokenizer for testing that implements the minimal interface needed
    by the text splitters without requiring downloading a real model.

    This tokenizer uses a simple word-based tokenization strategy where each
    word is roughly 1 token, making it easy to predict token counts.
    """

    def __init__(self):
        """Initialize the fake tokenizer."""
        # Skip parent initialization to avoid needing vocab files
        # Just set the minimal required attributes
        self.model_max_length = 1000000
        self.padding_side = "right"
        self.truncation_side = "right"
        self.model_input_names = ["input_ids"]

    def tokenize(self, text: str) -> list[str]:
        """Simple word-based tokenization for length calculation."""
        # Split on whitespace and punctuation
        import re
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return tokens

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Encode text to token IDs."""
        tokens = self.tokenize(text)
        # Simple hash-based token IDs
        token_ids = [hash(token) % 10000 for token in tokens]
        return token_ids

    def encode_plus(self, text: str, return_offsets_mapping: bool = False,
                    add_special_tokens: bool = True) -> dict:
        """
        Encode text and optionally return offset mappings.

        Returns a dictionary with:
        - input_ids: List of token IDs
        - offset_mapping: List of (start, end) character positions for each token
        """
        import re
        result = {
            'input_ids': []
        }

        if return_offsets_mapping:
            # Find all tokens and their positions
            offsets = []
            token_ids = []

            for match in re.finditer(r'\w+|[^\w\s]', text):
                token = match.group()
                start = match.start()
                end = match.end()

                offsets.append((start, end))
                token_ids.append(hash(token) % 10000)

            result['input_ids'] = token_ids
            result['offset_mapping'] = offsets
        else:
            result['input_ids'] = self.encode(text, add_special_tokens)

        return result


def get_test_content() -> str:
    """
    Generate test content that will trigger the langchain bug.

    This content is designed to:
    1. Be long enough to split into multiple chunks
    2. Have some repetitive patterns to test overlap detection
    3. Not require external files or downloads
    """
    return """# Game Rules

## Basic Gameplay

Players take turns rolling dice and moving their pieces around the board. Each turn consists of several phases that must be completed in order.

The first phase is the draw phase. During this phase, players draw cards from the deck. The number of cards drawn depends on the current game state and any special abilities in play.

The second phase is the action phase. Players may play cards from their hand, activate abilities, or perform other actions. Some actions require resources, while others are free.

The third phase is the combat phase. Players may attack opponents or defend against attacks. Combat is resolved by comparing strength values and applying modifiers from cards and abilities.

## Advanced Rules

Advanced players may use optional rules to increase complexity. These rules include special movement patterns, resource management systems, and combo mechanics.

Resource management is crucial for success. Players must balance spending resources for immediate gains versus saving them for powerful late-game plays.

Combo mechanics allow players to chain multiple actions together for increased effectiveness. Discovering powerful combos is a key strategic element.

## Winning Conditions

The game ends when one player reaches the victory threshold or all other players have been eliminated. Victory points are earned through various means including combat victories, objective completion, and resource accumulation.

Alternative victory conditions may be enabled for variety. These include elimination-style gameplay, territory control, or special achievement-based wins.

## Special Rules

Some cards and abilities have special timing rules. These must be resolved in the correct order to ensure fair gameplay. The rulebook provides a comprehensive priority system for resolving simultaneous effects.

When conflicts arise, the active player's effects resolve first, followed by each other player in turn order. This ensures a consistent resolution system that prevents ambiguity.

## Strategy Tips

Experienced players recommend balancing offensive and defensive strategies. Pure aggression leaves you vulnerable, while excessive caution allows opponents to build overwhelming advantages.

Card advantage is important but not everything. Sometimes sacrificing cards for tempo or board position is the correct play. Learning when to trade resources is a key skill.

## Tournament Rules

Official tournaments use standardized rule variants and banned card lists. These restrictions ensure balanced competitive play. Players should review the current tournament rules before participating in sanctioned events.

Time limits are enforced strictly in tournament play. Players must complete their turns within the allotted time or face penalties. Practice efficient play to avoid time-related losses."""


@pytest.mark.xfail(reason="Demonstrates the langchain TextSplitter bug with token-based splitting")
def test_text_splitter_start_index():
    """
    Test that demonstrates the langchain TextSplitter bug with token-based splitting.

    This test will FAIL because langchain's create_documents() method has a bug
    when calculating start_index with token-based length functions.

    The bug: The offset calculation in create_documents() mixes character-based
    lengths with token-based chunk_overlap, causing incorrect start_index values.

    See langchain_text_splitter_bug_analysis.md for details.
    """
    test_content = get_test_content()
    document = Document(page_content=test_content)

    tokenizer = FakeTokenizer()

    parent_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=150,
        chunk_overlap=50,
        add_start_index=True,
    )

    # Split the document into chunks
    documents = parent_splitter.split_documents([document])

    # Since we are only splitting one document, the "start_index" for the
    # generated chunks should be in ascending order. If not, then there is a
    # bug.

    indices = [i for i in range(len(documents))]
    indices_pairs = list(zip(indices, indices[1:]))
    for first, second in indices_pairs:
        first_start_index = documents[first].metadata["start_index"]
        second_start_index = documents[second].metadata["start_index"]
        assert first_start_index < second_start_index, (
            f"Start index of chunk {first} ({first_start_index}) is not less than "
            f"that of chunk {second} ({second_start_index})"
        )


def test_fixed_text_splitter_start_index():
    """
    Test that demonstrates the FixedRecursiveCharacterTextSplitter works correctly.

    This test uses FixedRecursiveCharacterTextSplitter, which is a drop-in replacement
    for RecursiveCharacterTextSplitter that properly handles start_index with tokenizers.
    """
    test_content = get_test_content()
    document = Document(page_content=test_content)

    tokenizer = FakeTokenizer()

    # Use the fixed splitter instead of the original
    parent_splitter = FixedRecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=150,
        chunk_overlap=50,
        add_start_index=True,
    )

    # Split the document into chunks
    documents = parent_splitter.split_documents([document])

    # Verify that all start_index values are correct and in ascending order
    # No workaround needed - the fixed splitter should work correctly!
    indices = [i for i in range(len(documents))]
    indices_pairs = list(zip(indices, indices[1:]))
    for first, second in indices_pairs:
        first_start_index = documents[first].metadata["start_index"]
        second_start_index = documents[second].metadata["start_index"]

        # Verify no -1 values
        assert first_start_index != -1, (
            f"Chunk {first} has invalid start_index: {first_start_index}"
        )
        assert second_start_index != -1, (
            f"Chunk {second} has invalid start_index: {second_start_index}"
        )

        # Verify ascending order
        assert first_start_index < second_start_index, (
            f"Start index of chunk {first} ({first_start_index}) is not less than "
            f"that of chunk {second} ({second_start_index})"
        )

    # Additional verification: check that each chunk can actually be found at its start_index
    for i, doc in enumerate(documents):
        start_idx = doc.metadata["start_index"]
        chunk_text = doc.page_content
        actual_text = test_content[start_idx:start_idx + len(chunk_text)]
        assert actual_text == chunk_text, (
            f"Chunk {i} at start_index {start_idx} does not match the actual text"
        )


def test_fixed_text_splitter_with_repetitive_content():
    """
    Test that FixedRecursiveCharacterTextSplitter handles repetitive text correctly.

    This ensures that the splitter doesn't get confused by repeated patterns
    and correctly identifies the position of each chunk.
    """
    # Create text with significant repetition
    repetitive_content = """Section A: This is a very important rule. """ * 10
    repetitive_content += """Section B: Another important rule here. """ * 10
    repetitive_content += """Section C: Yet another rule to follow. """ * 10
    repetitive_content += """Section A: This is a very important rule. """ * 5

    document = Document(page_content=repetitive_content)

    tokenizer = FakeTokenizer()

    # Use the fixed splitter
    splitter = FixedRecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=50,  # Smaller chunks to force more splits
        chunk_overlap=10,
        add_start_index=True,
    )

    documents = splitter.split_documents([document])

    # Verify that all chunks have valid start_index values
    for i, doc in enumerate(documents):
        start_idx = doc.metadata["start_index"]
        assert start_idx != -1, f"Chunk {i} has invalid start_index: {start_idx}"
        assert start_idx >= 0, f"Chunk {i} has negative start_index: {start_idx}"

    # Verify ascending order
    for i in range(len(documents) - 1):
        first_start = documents[i].metadata["start_index"]
        second_start = documents[i + 1].metadata["start_index"]
        assert first_start < second_start, (
            f"Start index of chunk {i} ({first_start}) is not less than "
            f"that of chunk {i+1} ({second_start})"
        )

    # Verify that each chunk actually appears at its stated position
    for i, doc in enumerate(documents):
        start_idx = doc.metadata["start_index"]
        chunk_text = doc.page_content
        actual_text = repetitive_content[start_idx:start_idx + len(chunk_text)]
        assert actual_text == chunk_text, (
            f"Chunk {i} at start_index {start_idx} does not match. "
            f"Expected: {chunk_text[:50]}... "
            f"Got: {actual_text[:50]}..."
        )
