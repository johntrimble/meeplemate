import inspect
from collections import Counter

from pytest_unordered import unordered

from meeplemate.quote_util import expand_to_full_paragraphs, find_quote_with_gaps, find_quotes_in_text

def test_find_quotes_in_text():
    sample_answer = """
    Some unrelated content here.

    ## Heading of some kind

    > "Hey, look at me, I'm a quote!" (Rulebook Name, p. 42)

    Here is some analysis.

    > Another quote, look at that! (Another Rulebook, p. 7)

    More analysis, and then "some inline quote here," (Inline Source, p. 1) and some ongoing text.

    > Some text here that goes on and on and on.
    > It even spans multiple lines!
    >
    > With empty lines even! (Multi-line Rulebook, p. 99)
    """
    sample_answer = inspect.cleandoc(sample_answer)

    extracted_quotes = find_quotes_in_text(sample_answer)

    expected = [
        {
            'citation': {
                'end_index': 56,
                'page': '42',
                'ref_name': 'Rulebook Name',
                'start_index': 34,
                'text': '(Rulebook Name, p. 42)'
            },
            'end_index': 111,
            'quote': "Hey, look at me, I'm a quote!",
            'quote_type': 'blockquote',
            'start_index': 55,
            'text': '> "Hey, look at me, I\'m a quote!" (Rulebook Name, p. 42)'
        },
        {
            'citation': {
                'end_index': 55,
                'page': '7',
                'ref_name': 'Another Rulebook',
                'start_index': 31,
                'text': '(Another Rulebook, p. 7)'
            },
            'end_index': 192,
            'quote': 'Another quote, look at that!',
            'quote_type': 'blockquote',
            'start_index': 137,
            'text': '> Another quote, look at that! (Another Rulebook, p. 7)'
        },
        {
            'citation': {
                'end_index': 132,
                'page': '99',
                'ref_name': 'Multi-line Rulebook',
                'start_index': 104,
                'text': '(Multi-line Rulebook, p. 99)'
            },
            'end_index': 422,
            'quote': 'Some text here that goes on and on and on. It even spans multiple '
            'lines!  With empty lines even!',
            'quote_type': 'blockquote',
            'start_index': 290,
            'text': '> Some text here that goes on and on and on.\n'
                '> It even spans multiple lines!\n'
                '>\n'
                '> With empty lines even! (Multi-line Rulebook, p. 99)'
        },
        {
            'citation': {
                'end_index': 47,
                'page': '1',
                'ref_name': 'Inline Source',
                'start_index': 26,
                'text': '(Inline Source, p. 1)'
            },
            'end_index': 265,
            'quote': 'some inline quote here,',
            'quote_type': 'inline',
            'start_index': 218,
            'text': '"some inline quote here," (Inline Source, p. 1)'
        }
    ]
    assert extracted_quotes == unordered(expected), f"Extracted quotes do not match expected.\nExtracted: {extracted_quotes}\nExpected: {expected}"


def test_find_quote_with_gaps():
    # Some sample document
    sample_document_text = (
        "Here is a paragraph about text. Look at all the text we have here. So much of it. It just keeps going and going. Yay text!"
    )

    # Test exact match
    assert find_quote_with_gaps(sample_document_text, "Look at all the text we have here.")

    # Test no match
    assert find_quote_with_gaps(sample_document_text, "This quote does not exist.") is None

    # Test ellipsis handling
    result = find_quote_with_gaps(sample_document_text, "Look at all the text... here.")
    assert result is not None
    assert result.matched_text == "Look at all the text we have here."

    # Test handles spelling errors
    result = find_quote_with_gaps(sample_document_text, "Look at all the txt we have here.")
    assert result is not None
    assert result.matched_text == "Look at all the text we have here."


def test_expand_to_full_paragraphs_with_heading_blank_line():
    """Test expansion includes heading when followed by blank line."""
    doc = inspect.cleandoc("""
    Some intro text.

    ## Equipment Cards

    When you draw an Equipment card, you must decide whether to equip it or discard it.
    You can only have one of each type equipped at a time.
    """)

    quote = "you must decide whether to equip it"
    result = expand_to_full_paragraphs(doc, quote)

    # Should include the heading
    assert result.startswith("## Equipment Cards")
    assert "When you draw an Equipment card" in result
    assert "one of each type equipped" in result


def test_expand_to_full_paragraphs_with_heading_no_blank_line():
    """Test expansion includes heading even without blank line separator."""
    doc = inspect.cleandoc("""
    Some intro text.

    ### Combat Phase
    During combat, players compare their combat strength.
    The player with the higher total wins.
    """)

    quote = "compare their combat strength"
    result = expand_to_full_paragraphs(doc, quote)

    # Should include the heading even without blank line
    assert result.startswith("### Combat Phase")
    assert "During combat" in result
    assert "higher total wins" in result


def test_expand_to_full_paragraphs_without_heading():
    """Test expansion does not include regular text above paragraph."""
    doc = inspect.cleandoc("""
    This is some regular text that is not a heading.

    This is the actual paragraph we care about.
    It contains the quote we're looking for.
    """)

    quote = "the quote we're looking for"
    result = expand_to_full_paragraphs(doc, quote)

    # Should not include the regular text above
    assert not result.startswith("This is some regular text")
    assert result.startswith("This is the actual paragraph")
    assert "the quote we're looking for" in result


def test_expand_to_full_paragraphs_with_multiple_headings():
    """Test expansion includes only the immediate heading, not parent headings."""
    doc = inspect.cleandoc("""
    # Main Section

    ## Subsection

    This paragraph contains the quote we want.
    It should include the subsection heading but not the main section.
    """)

    quote = "the quote we want"
    result = expand_to_full_paragraphs(doc, quote)

    # Should include only the immediate heading (##), not the parent (#)
    assert result.startswith("## Subsection")
    assert "# Main Section" not in result
    assert "This paragraph contains" in result


def test_expand_to_full_paragraphs_heading_at_document_start():
    """Test expansion includes heading when it's at the very beginning."""
    doc = inspect.cleandoc("""
    # Introduction

    This is the first paragraph of the document.
    It starts right after the heading.
    """)

    quote = "first paragraph of the document"
    result = expand_to_full_paragraphs(doc, quote)

    # Should include the heading at document start
    assert result.startswith("# Introduction")
    assert "This is the first paragraph" in result


def test_expand_to_full_paragraphs_heading_exceeds_max_chars():
    """Test graceful fallback when heading+paragraph exceeds char limit."""
    doc = inspect.cleandoc("""
    Some intro.

    ## This Is A Very Long Heading That Takes Up Many Characters

    Short quote here.
    """)

    quote = "Short quote here"
    # Set limit so paragraph fits but paragraph+heading does not
    # The quote "Short quote here" is 17 chars
    # The heading is much longer
    result = expand_to_full_paragraphs(doc, quote, max_additional_chars=5)

    # Should use graceful fallback: paragraph without heading
    assert not result.startswith("##")
    assert result == "Short quote here."


def test_expand_to_full_paragraphs_all_exceeds_max_chars():
    """Test returns matched text when even paragraph expansion exceeds limit."""
    doc = inspect.cleandoc("""
    Some intro.

    ## Heading

    This is a much longer paragraph that contains our quote.
    It has many additional words before and after the quote we're searching for here.
    """)

    quote = "the quote we're searching"
    # Set very small limit so even paragraph expansion fails
    result = expand_to_full_paragraphs(doc, quote, max_additional_chars=5)

    # Should return just the matched text
    assert result == "the quote we're searching"
