import inspect
from collections import Counter

from pytest_unordered import unordered

from meeplemate.quote_util import (
    expand_to_full_paragraphs,
    find_quote_with_gaps,
    find_quotes_in_text,
    split_quote_parts,
    strip_blockquote_markers_and_quotes,
    strip_html_tags,
    strip_html_tags_with_map,
    strip_latex,
    strip_latex_with_map,
)

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
            'lines!\n\nWith empty lines even!',
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


def test_find_quote_with_gaps_near_end_of_long_document():
    """Quote near end of a document longer than window_size must still be found.

    Regression test: the sliding-window loop could leave up to window_step-1
    trailing characters uncovered, causing quotes at the tail to be missed.
    """
    # Build a document just over the default window_size (1200 normalised chars)
    # with the target quote sitting at the very end.
    filler = "All work and no play makes Jack a dull boy. " * 30  # ~1350 chars
    target = "The treasure is buried under the old oak tree by the river."
    doc = filler + "\n\n" + target

    result = find_quote_with_gaps(doc, target)
    assert result is not None, "Quote at the tail of a long document should be found"
    assert target in result.matched_text
    assert result.score >= 85


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


# ---------------------------------------------------------------------------
# LaTeX stripping
# ---------------------------------------------------------------------------


def test_strip_latex_with_map_prime():
    r"""\\prime inside inline math is replaced with apostrophe."""
    s = r"within \(12^{\prime \prime}\) of the"
    stripped, idx_map = strip_latex_with_map(s)
    # Space between \prime commands is preserved (collapsed later by normalize)
    assert stripped == "within 12' ' of the"
    assert len(idx_map) == len(stripped)
    assert all(0 <= i < len(s) for i in idx_map)
    # Monotonically non-decreasing
    for i in range(1, len(idx_map)):
        assert idx_map[i] >= idx_map[i - 1]


def test_strip_latex_with_map_circ():
    r"""\\circ inside inline math is replaced with degree sign."""
    s = r"within \(12^{\circ}\) of"
    stripped, idx_map = strip_latex_with_map(s)
    assert stripped == "within 12\u00b0 of"
    assert len(idx_map) == len(stripped)


def test_strip_latex_with_map_simple_math():
    """Simple arithmetic in inline math passes through."""
    s = r"gives him \(3 + 3 = 6\) points"
    stripped, idx_map = strip_latex_with_map(s)
    assert stripped == "gives him 3 + 3 = 6 points"
    assert len(idx_map) == len(stripped)


def test_strip_latex_with_map_plus_value():
    """Signed number in inline math passes through."""
    s = r"an extra \(+1\) combat bonus"
    stripped, idx_map = strip_latex_with_map(s)
    assert stripped == "an extra +1 combat bonus"


def test_strip_latex_with_map_no_latex():
    """Text without LaTeX passes through unchanged with identity map."""
    s = "plain text with no latex at all"
    stripped, idx_map = strip_latex_with_map(s)
    assert stripped == s
    assert idx_map == list(range(len(s)))


def test_strip_latex_with_map_multiple_regions():
    r"""Multiple inline math regions in one string are all processed."""
    s = r"within \(12^{\prime \prime}\) and \(8^{\circ}\) away"
    stripped, idx_map = strip_latex_with_map(s)
    assert "12' '" in stripped
    assert "8\u00b0" in stripped
    assert len(idx_map) == len(stripped)


def test_strip_latex_no_map():
    """strip_latex returns only the stripped string."""
    assert strip_latex(r"within \(12^{\prime \prime}\) of") == "within 12' ' of"


# ---------------------------------------------------------------------------
# find_quote_with_gaps – LaTeX integration
# ---------------------------------------------------------------------------


def test_find_quote_with_gaps_latex_prime():
    r"""Quote with plain quotes matches document with \\prime LaTeX notation."""
    doc = (
        r"Any unit within \(12^{\prime \prime}\) of the general model "
        r"may use the general's Leadership value instead of its own "
        r"when making a Leadership-based test."
    )
    quote = "Any unit within 12\u2019\u2019 of the general model may use the general\u2019s Leadership value"
    result = find_quote_with_gaps(doc, quote)
    assert result is not None
    assert result.score >= 85
    assert "12" in result.matched_text
    assert "Leadership value" in result.matched_text


def test_find_quote_with_gaps_latex_circ():
    r"""Quote with plain inch marks matches document with \\circ LaTeX notation."""
    doc = (
        r"each remaining unit within \(12^{\circ}\) of friendly units "
        r"which have broken or been wiped out"
    )
    quote = 'each remaining unit within 12" of friendly units which have broken'
    result = find_quote_with_gaps(doc, quote)
    assert result is not None
    assert result.score >= 85


def test_find_quote_with_gaps_latex_battle_standard_regression():
    r"""Real regression: battle standard quote was marked unfixable due to \\prime \\prime."""
    doc = (
        r"Any unit within \(12^{\prime \prime}\) of the battle standard "
        r"may retake a failed Break test. The unit is only allowed to "
        r"retake this test once."
    )
    quote = "Any unit within 12\u2019\u2019 of the battle standard may re-take a failed Break test."
    result = find_quote_with_gaps(doc, quote)
    assert result is not None, "This quote was previously marked unfixable"
    assert result.score >= 85


def test_find_quote_with_gaps_latex_matched_text_from_original():
    r"""matched_text should extract from the original LaTeX-containing document."""
    doc = r"foo bar \(12^{\prime \prime}\) baz qux"
    quote = "bar 12'' baz"
    result = find_quote_with_gaps(doc, quote)
    assert result is not None
    # matched_text must be a substring of the original doc
    assert doc[result.start : result.end] == result.matched_text
    # It should contain the original LaTeX notation
    assert r"\prime" in result.matched_text


# ---------------------------------------------------------------------------
# HTML tag stripping
# ---------------------------------------------------------------------------


def test_strip_html_tags_with_map_simple():
    """Simple HTML tags are replaced with spaces."""
    s = "Hello <b>world</b> test"
    stripped, idx_map = strip_html_tags_with_map(s)
    # Tags become spaces; extra spaces collapsed later by normalize
    assert stripped == "Hello  world  test"
    assert len(idx_map) == len(stripped)
    assert all(0 <= i < len(s) for i in idx_map)
    for i in range(1, len(idx_map)):
        assert idx_map[i] >= idx_map[i - 1]


def test_strip_html_tags_with_map_table():
    """HTML table markup is replaced with spaces, preserving word boundaries."""
    s = "<table><tr><td>Grail Knights</td><td>4</td><td>5</td></tr></table>"
    stripped, idx_map = strip_html_tags_with_map(s)
    # Tags become spaces (collapsed later by normalize)
    assert "Grail Knights" in stripped
    assert " 4 " in stripped
    assert " 5 " in stripped
    assert len(idx_map) == len(stripped)


def test_strip_html_tags_with_map_no_html():
    """Text without HTML passes through unchanged with identity map."""
    s = "plain text with no html"
    stripped, idx_map = strip_html_tags_with_map(s)
    assert stripped == s
    assert idx_map == list(range(len(s)))


def test_strip_html_tags_with_map_center():
    """<center> tags become spaces, content preserved."""
    s = "<center>Some centered text</center>"
    stripped, idx_map = strip_html_tags_with_map(s)
    assert stripped == " Some centered text "
    assert len(idx_map) == len(stripped)


def test_strip_html_tags_no_map():
    """strip_html_tags returns only the stripped string."""
    assert strip_html_tags("<td>hello</td>") == " hello "


# ---------------------------------------------------------------------------
# find_quote_with_gaps – HTML table integration
# ---------------------------------------------------------------------------


def test_find_quote_with_gaps_html_table_regression():
    """Pipe-delimited quote matches document with HTML table markup."""
    doc = (
        '<table><tr><td></td><td>M</td><td>WS</td><td>BS</td><td>S</td>'
        '<td>T</td><td>W</td><td>I</td><td>A</td><td>Ld</td></tr>'
        '<tr><td>Grail Knights</td><td>4</td><td>5</td><td>3</td>'
        '<td>4</td><td>3</td><td>1</td><td>4</td><td>1</td><td>9</td></tr></table>'
    )
    quote = "| Grail Knights | 4 | 5 | 3 | 4 | 3 | 1 | 4 | 1 | 9 |"
    result = find_quote_with_gaps(doc, quote)
    assert result is not None, "HTML table quote was previously marked unfixable"
    assert result.score >= 85


def test_find_quote_with_gaps_html_table_matched_text_from_original():
    """matched_text from HTML table match should reference original document."""
    doc = "Some text <table><tr><td>A</td><td>B</td></tr></table> more text"
    quote = "A B"
    result = find_quote_with_gaps(doc, quote)
    assert result is not None
    assert doc[result.start : result.end] == result.matched_text
    # Should contain original HTML
    assert "<td>" in result.matched_text or "A" in result.matched_text


# ---------------------------------------------------------------------------
# strip_blockquote_markers_and_quotes – paragraph breaks
# ---------------------------------------------------------------------------


def test_strip_blockquote_markers_preserves_paragraph_breaks():
    """Blank > lines become \\n\\n paragraph breaks."""
    raw = "> First paragraph.\n>\n> Second paragraph."
    result = strip_blockquote_markers_and_quotes(raw)
    assert result == "First paragraph.\n\nSecond paragraph."


def test_strip_blockquote_markers_joins_continuation_lines():
    """Consecutive > lines within a paragraph join with a single space."""
    raw = "> Line one continues\n> on line two."
    result = strip_blockquote_markers_and_quotes(raw)
    assert result == "Line one continues on line two."


def test_strip_blockquote_markers_mixed():
    """Mix of continuation lines and paragraph breaks."""
    raw = "> Para one line one\n> para one line two.\n>\n> Para two."
    result = strip_blockquote_markers_and_quotes(raw)
    assert result == "Para one line one para one line two.\n\nPara two."


# ---------------------------------------------------------------------------
# split_quote_parts
# ---------------------------------------------------------------------------


def test_split_quote_parts_ellipsis():
    """Splits on ... (ellipsis)."""
    assert split_quote_parts("hello... world") == ["hello", "world"]


def test_split_quote_parts_unicode_ellipsis():
    """Splits on \u2026 (unicode ellipsis)."""
    assert split_quote_parts("hello\u2026 world") == ["hello", "world"]


def test_split_quote_parts_newline():
    """Splits on newlines (paragraph breaks from blockquotes)."""
    assert split_quote_parts("para one.\n\npara two.") == ["para one.", "para two."]


def test_split_quote_parts_no_split():
    """No split points returns the whole string."""
    assert split_quote_parts("just one part") == ["just one part"]


# ---------------------------------------------------------------------------
# find_quote_with_gaps – multi-paragraph blockquote
# ---------------------------------------------------------------------------


def test_find_quote_with_gaps_multi_paragraph_blockquote():
    """Multi-paragraph quote where each paragraph is in a different part of the document."""
    doc = (
        "Knights who have fulfilled their quest and drunk from the grail. "
        "This makes them special above and beyond ordinary men. "
        "All Grail Knights present in the army fight together in a single unit. "
        "Some filler text here that separates the two paragraphs in the document. "
        "More filler. Even more filler to increase the gap. "
        "Grail Knights have the Grail Virtue; they have drunk from the sacred "
        "grail and are immune to psychology."
    )
    # Quote with \n\n paragraph break (as produced by strip_blockquote_markers_and_quotes)
    quote = (
        "Knights who have fulfilled their quest and drunk from the grail. "
        "This makes them special above and beyond ordinary men. "
        "All Grail Knights present in the army fight together in a single unit."
        "\n\n"
        "Grail Knights have the Grail Virtue; they have drunk from the sacred "
        "grail and are immune to psychology."
    )
    result = find_quote_with_gaps(doc, quote)
    assert result is not None, "Multi-paragraph blockquote should match with gap"
    assert result.score >= 85
