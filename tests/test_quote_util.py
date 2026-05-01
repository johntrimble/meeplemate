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

def test_find_quotes_in_text_citation_page_word():
    """Blockquote with 'page N' format citation (instead of 'p. N') is captured."""
    text = inspect.cleandoc("""\
        The rules state:

        > Roll two dice and add your combat score.

        (Battle Rulebook, page 15)
    """)
    quotes = find_quotes_in_text(text)
    assert len(quotes) == 1
    q = quotes[0]
    assert q['quote_type'] == 'blockquote'
    assert q['quote'] == 'Roll two dice and add your combat score.'
    assert q['citation'] is not None
    assert q['citation']['text'] == '(Battle Rulebook, page 15)'
    assert q['citation']['ref_name'] == 'Battle Rulebook'
    assert q['citation']['page'] == '15'
    assert q['text'].endswith('(Battle Rulebook, page 15)')


def test_find_quotes_in_text_citation_pp():
    """Blockquote with 'pp. N' format citation is captured."""
    text = inspect.cleandoc("""\
        The rules state:

        > Roll two dice and add your combat score.

        (Battle Rulebook, pp. 15)
    """)
    quotes = find_quotes_in_text(text)
    assert len(quotes) == 1
    q = quotes[0]
    assert q['quote_type'] == 'blockquote'
    assert q['quote'] == 'Roll two dice and add your combat score.'
    assert q['citation'] is not None
    assert q['citation']['text'] == '(Battle Rulebook, pp. 15)'
    assert q['citation']['ref_name'] == 'Battle Rulebook'
    assert q['citation']['page'] == '15'
    assert q['text'].endswith('(Battle Rulebook, pp. 15)')


def test_find_quotes_in_text_citation_as_blockquote():
    """Citation placed in its own blockquote line is merged with the preceding quote."""
    text = inspect.cleandoc("""\
        The rules state:

        > Roll two dice and add your combat score.

        > (Battle Rulebook, p. 15)
    """)
    quotes = find_quotes_in_text(text)
    assert len(quotes) == 1, f"Expected 1 quote, got {len(quotes)}: {quotes}"
    q = quotes[0]
    assert q['quote_type'] == 'blockquote'
    assert q['quote'] == 'Roll two dice and add your combat score.'
    assert q['citation'] is not None
    assert q['citation']['text'] == '(Battle Rulebook, p. 15)'
    assert q['citation']['ref_name'] == 'Battle Rulebook'
    assert q['citation']['page'] == '15'


def test_find_quotes_in_text_multiple_blockquote_citations():
    """Multiple blockquotes each followed by a citation-as-blockquote are all captured."""
    text = inspect.cleandoc("""\
        First rule:

        > The first rule states this.

        > (Rulebook A, p. 10)

        Second rule:

        > The second rule states that.

        > (Rulebook B, p. 20)
    """)
    quotes = find_quotes_in_text(text)
    assert len(quotes) == 2, f"Expected 2 quotes, got {len(quotes)}: {quotes}"
    quotes_by_ref = {q['citation']['ref_name']: q for q in quotes}
    assert 'Rulebook A' in quotes_by_ref
    assert quotes_by_ref['Rulebook A']['quote'] == 'The first rule states this.'
    assert quotes_by_ref['Rulebook A']['citation']['page'] == '10'
    assert 'Rulebook B' in quotes_by_ref
    assert quotes_by_ref['Rulebook B']['quote'] == 'The second rule states that.'
    assert quotes_by_ref['Rulebook B']['citation']['page'] == '20'


def test_find_quotes_in_text_unparseable_inline_citation():
    """Inline unparseable citation is still captured with empty ref_name/page."""
    text = '> Some block quote ending with an inline citation. (some citation pattern, we did not expect)'
    quotes = find_quotes_in_text(text)
    assert len(quotes) == 1
    q = quotes[0]
    assert q['quote'] == 'Some block quote ending with an inline citation.'
    assert q['citation'] is not None
    assert q['citation']['text'] == '(some citation pattern, we did not expect)'
    assert q['citation']['ref_name'] == ''
    assert q['citation']['page'] == ''
    # Citation is within the captured span
    assert q['end_index'] == len(text)


def test_find_quotes_in_text_unparseable_bare_citation():
    """Bare unparseable citation on its own line is captured with empty ref_name/page.

    A standalone (...) line immediately after a blockquote (with nothing else on
    that line) is treated as a citation regardless of its internal format.
    """
    text = '> Some block quote.\n\n(Some unexpected. Citation. Pattern)'
    quotes = find_quotes_in_text(text)
    assert len(quotes) == 1
    q = quotes[0]
    assert q['quote'] == 'Some block quote.'
    assert q['citation'] is not None
    assert q['citation']['text'] == '(Some unexpected. Citation. Pattern)'
    assert q['citation']['ref_name'] == ''
    assert q['citation']['page'] == ''
    assert q['end_index'] == len(text)


def test_find_quotes_in_text_unparseable_blockquote_citation():
    """Unparseable citation in its own blockquote line is captured with empty ref_name/page."""
    text = '> Some block quote.\n\n> (Some unexpected. Citation. Pattern)'
    quotes = find_quotes_in_text(text)
    assert len(quotes) == 1
    q = quotes[0]
    assert q['quote'] == 'Some block quote.'
    assert q['citation'] is not None
    assert q['citation']['text'] == '(Some unexpected. Citation. Pattern)'
    assert q['citation']['ref_name'] == ''
    assert q['citation']['page'] == ''
    assert q['end_index'] == len(text)


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


def test_find_quote_with_gaps_common_opening_word():
    """Verbatim quote whose opening word appears earlier in the document is still found.

    Regression for Q25: "However, some bonuses apply..." was rejected because
    SequenceMatcher anchored on an earlier "However", producing a span that
    exceeded the 3x multiplier.  The two-pass windowed approach should match
    against the dominant block (the large verbatim tail) and return a tight span.
    """
    early_filler = (
        "However, the Goblin player has 4 complete ranks in his formation, and as "
        "each extra rank adds +1 to his score this gives him 6 points. "
        "The Elves have therefore lost the combat even though they inflicted more casualties. "
        "However, the situation may be reversed in subsequent rounds of combat. "
    )
    target = (
        "However, some bonuses apply specifically to Break tests and others apply "
        "specifically to psychology tests."
    )
    doc = early_filler + target

    result = find_quote_with_gaps(doc, target)
    assert result is not None, "Verbatim quote should be found despite common opening word"
    assert result.score >= 92
    assert "some bonuses apply specifically" in result.matched_text


def test_find_quote_with_gaps_changed_opening_phrase():
    """Quote whose opening phrase differs from the document but the body is verbatim.

    Regression for Q5: "The side that loses a combat..." was rejected because
    the document says "Each unit that loses in combat..." — the changed opening
    anchored far away, exploding the span.
    """
    doc = (
        "## RESULTS\n\n"
        "Work out which side has won each combat and by how much. "
        "The losing side will have lost by 1, 2 or more points as explained later.\n\n"
        "## BREAK TESTS\n\n"
        "Each unit that loses in combat must take a Break test as explained in the rules. "
        "Any units failing their Break test are termed broken and a note is made or models "
        "are turned round to show this. Take all Break tests now."
    )
    quote = (
        "The side that loses a combat must take a Break test as explained in the rules. "
        "Any units failing their Break test are termed broken and a note is made or models "
        "are turned round to show this. Take all Break tests now."
    )

    result = find_quote_with_gaps(doc, quote)
    assert result is not None, "Near-verbatim quote with changed opening should be found"
    assert result.score >= 92
    assert "Break test as explained in the rules" in result.matched_text


def test_find_quote_with_gaps_paraphrase_rejected():
    """Quote with multiple word substitutions spread across the text is rejected.

    Regression for Q26: "Break tests are not psychological tests. The two are
    separate and distinct." — multiple changes from the real text mean no single
    block is large enough to anchor a clean windowed match above threshold.
    """
    doc = (
        "Players will immediately realise that a psychology test is taken in the same "
        "way as a Break test in hand-to-hand combat and uses the same characteristic, "
        "namely Leadership. However, a Break test is not a psychology test. "
        "The two tests are quite separate. This is important because some bonuses apply "
        "specifically to Break tests and others apply specifically to psychology tests."
    )
    quote = "Break tests are not psychological tests. The two are separate and distinct."

    result = find_quote_with_gaps(doc, quote)
    assert result is None, "Multi-word paraphrase should not be accepted"


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


def test_multiple_part_match_candidates():
    # The key property exercised here is that one sentence — "Roll 2 dice and
    # consult the Resolve table." — appears verbatim in multiple paragraphs of
    # the document, giving the matcher several candidate positions for that
    # quote part and forcing it to pick the correct one.
    quote = (
        "A fortress that fails a Garrison check must immediately discard one Support token. "
        "If no Support tokens remain the fortress is considered Broken and its garrison surrenders. "
        "Roll 2 dice and consult the Resolve table. "
        "Each defending player with units inside the fortress walls must also check morale separately.\n\n"
        "To resolve a Garrison check, the active player first counts the number of Siege tokens on the fortress card. "
        "Each Siege token reduces the fortress Resolve score by one. "
        "If the modified Resolve score falls below the current Threat level, the check fails. "
        "Roll 2 dice and consult the Resolve table. "
        "Apply any terrain or weather modifiers shown on the active Condition card before comparing the result.\n\n"
        "## REINFORCEMENTS\n\n"
        "When a Garrison check is failed and the fortress becomes Broken, the besieging player may immediately "
        "move up to two of their units from an adjacent territory into the fortress hex at no movement cost. "
        "These units are treated as having entered through the breach and do not trigger Gate checks.\n\n"
        "## SURRENDER TERMS\n\n"
        "A Broken fortress must accept Surrender Terms at the start of the next round unless it is relieved. "
        "The controlling player draws one Term card and applies its effect immediately. "
        "Surrender Terms may include resource penalties, unit losses, or forced allegiance shifts. "
        "Once Surrender Terms are accepted the fortress token is flipped to its Occupied side.\n\n"
        "## RELIEF ATTEMPTS\n\n"
        "A Broken fortress may be saved if a friendly army enters its hex before Surrender Terms are applied. "
        "The relieving army must win a field battle against any besieging units present. "
        "If the battle is won, remove all Siege tokens from the fortress and flip it back to its Garrisoned side. "
        "The fortress is no longer considered Broken and resumes normal Garrison checks next round.\n\n"
        "<center>The northern fortress falls — the eastern road lies open! </center>"
    )
    document = (
        "## GARRISON CHECKS\n\n"
        "A fortress that fails a Garrison check must immediately discard one Support token. "
        "If no Support tokens remain the fortress is considered Broken and its garrison surrenders. "
        "Roll 2 dice and consult the Resolve table. "
        "Each defending player with units inside the fortress walls must also check morale separately.\n\n"
        "To resolve a Garrison check, the active player first counts the number of Siege tokens on the fortress card. "
        "Each Siege token reduces the fortress Resolve score by one. "
        "If the modified Resolve score falls below the current Threat level, the check fails. "
        "Roll 2 dice and consult the Resolve table. "
        "Apply any terrain or weather modifiers shown on the active Condition card before comparing the result.\n\n"
        "## REINFORCEMENTS\n\n"
        "When a Garrison check is failed and the fortress becomes Broken, the besieging player may immediately "
        "move up to two of their units from an adjacent territory into the fortress hex at no movement cost. "
        "These units are treated as having entered through the breach and do not trigger Gate checks.\n\n"
        "## SURRENDER TERMS\n\n"
        "A Broken fortress must accept Surrender Terms at the start of the next round unless it is relieved. "
        "The controlling player draws one Term card and applies its effect immediately. "
        "Surrender Terms may include resource penalties, unit losses, or forced allegiance shifts. "
        "Once Surrender Terms are accepted the fortress token is flipped to its Occupied side.\n\n"
        "## RELIEF ATTEMPTS\n\n"
        "A Broken fortress may be saved if a friendly army enters its hex before Surrender Terms are applied. "
        "The relieving army must win a field battle against any besieging units present. "
        "If the battle is won, remove all Siege tokens from the fortress and flip it back to its Garrisoned side. "
        "The fortress is no longer considered Broken and resumes normal Garrison checks next round.\n\n"
        "<center>The northern fortress falls — the eastern road lies open! </center>"
    )
    result = find_quote_with_gaps(document, quote)
    assert result is not None


def test_find_quote_with_html_escapes():
    # Key property: the quote contains HTML-escaped table markup (&lt;table&gt;, &lt;tr&gt;, &lt;td&gt;)
    # while the document contains the same markup unescaped (<table>, <tr>, <td>).
    # The matcher must unescape the quote before comparing so it can find the match.
    quote = (
        '## TURN SEQUENCE\n\n'
        '&lt;table&gt;&lt;tr&gt;&lt;td colspan="2"&gt;TURN SEQUENCE&lt;/td&gt;&lt;/tr&gt;'
        '&lt;tr&gt;&lt;td&gt;1&lt;/td&gt;&lt;td&gt;Draw Phase Each player draws two cards from the top of the deck and adds them to their hand.&lt;/td&gt;&lt;/tr&gt;'
        '&lt;tr&gt;&lt;td&gt;2&lt;/td&gt;&lt;td&gt;Play Phase Each player may play any number of cards from their hand by paying their cost in resources.&lt;/td&gt;&lt;/tr&gt;'
        '&lt;tr&gt;&lt;td&gt;3&lt;/td&gt;&lt;td&gt;Attack Phase The active player may declare attacks with any ready units they control.&lt;/td&gt;&lt;/tr&gt;'
        '&lt;tr&gt;&lt;td&gt;4&lt;/td&gt;&lt;td&gt;Block Phase The defending player may assign any number of ready units to block each incoming attacker.&lt;/td&gt;&lt;/tr&gt;'
        '&lt;tr&gt;&lt;td&gt;5&lt;/td&gt;&lt;td&gt;Resolve Phase Deal damage equal to each unit\'s power and remove units that have taken lethal damage.&lt;/td&gt;&lt;/tr&gt;'
        '&lt;tr&gt;&lt;td&gt;6&lt;/td&gt;&lt;td&gt;End Phase Discard down to the hand limit and pass the turn to the next player.&lt;/td&gt;&lt;/tr&gt;'
        '&lt;/table&gt;'
    )
    document = (
        'TURN SEQUENCE</td></tr>'
        '<tr><td>1</td><td>Draw Phase Each player draws two cards from the top of the deck and adds them to their hand.</td></tr>'
        '<tr><td>2</td><td>Play Phase Each player may play any number of cards from their hand by paying their cost in resources.</td></tr>'
        '<tr><td>3</td><td>Attack Phase The active player may declare attacks with any ready units they control.</td></tr>'
        '<tr><td>4</td><td>Block Phase The defending player may assign any number of ready units to block each incoming attacker.</td></tr>'
        '<tr><td>5</td><td>Resolve Phase Deal damage equal to each unit\'s power and remove units that have taken lethal damage.</td></tr>'
        '<tr><td>6</td><td>End Phase Discard down to the hand limit and pass the turn to the next player.</td></tr>'
        '</table>'
    )
    result = find_quote_with_gaps(document, quote)
    assert result is not None

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


# ---------------------------------------------------------------------------
# find_quotes_in_text – indented blockquotes
# ---------------------------------------------------------------------------


def test_find_quotes_in_text_indented_blockquote_spaces():
    """Blockquote indented with spaces (as in a numbered list) is detected."""
    text = inspect.cleandoc("""\
        2. **Locate the definition**:

           > The side that loses a combat must take a Break test.

           > (Warhammer Rulebook, p. 41)
    """)
    quotes = find_quotes_in_text(text)
    assert len(quotes) == 1
    q = quotes[0]
    assert q['quote_type'] == 'blockquote'
    assert q['quote'] == 'The side that loses a combat must take a Break test.'
    assert q['citation'] is not None
    assert q['citation']['ref_name'] == 'Warhammer Rulebook'
    assert q['citation']['page'] == '41'


def test_find_quotes_in_text_indented_blockquote_multi_line():
    """Multi-line blockquote indented with spaces is detected and joined correctly."""
    text = "2. Step:\n\n   > First sentence of the rule.\n   > Second sentence continues.\n   >\n   > (Rulebook, p. 10)"
    quotes = find_quotes_in_text(text)
    assert len(quotes) == 1
    q = quotes[0]
    assert q['quote_type'] == 'blockquote'
    assert 'First sentence' in q['quote']
    assert 'Second sentence' in q['quote']
    assert q['citation'] is not None
    assert q['citation']['page'] == '10'


def test_find_quotes_in_text_indented_blockquote_mixed():
    """Mix of a properly-formatted blockquote and an indented blockquote — both detected."""
    text = inspect.cleandoc("""\
        Normal blockquote:

        > Normal rule text.

        (Rulebook A, p. 1)

        Numbered step:

           > Indented rule text.

           > (Rulebook B, p. 2)
    """)
    quotes = find_quotes_in_text(text)
    assert len(quotes) == 2
    quote_types = {q['quote'] for q in quotes}
    assert 'Normal rule text.' in quote_types
    assert 'Indented rule text.' in quote_types


def test_find_quotes_in_text_numbered_list_with_blockquotes():
    """Realistic LLM-style numbered list with indented blockquotes as sub-items."""
    text = inspect.cleandoc("""\
        ### Step-by-step reasoning:

        1. **Identify the relevant mechanics**:
           The question is about Break tests and Grail Knights.

        2. **Locate the definition of Break tests**:
           > The side that loses a combat must take a test to determine whether it stands and fights.
           >
           > (Warhammer Rulebook, p. 41)

        3. **Check the Grail Virtue**:
           > Grail Knights are unaffected by any of the psychology rules.
           >
           > (Bretonnia Army Book, p. 43)
    """)
    quotes = find_quotes_in_text(text)
    assert len(quotes) == 2
    quotes_by_ref = {q['citation']['ref_name']: q for q in quotes if q['citation']}
    assert 'Warhammer Rulebook' in quotes_by_ref
    assert 'Bretonnia Army Book' in quotes_by_ref
    assert quotes_by_ref['Warhammer Rulebook']['citation']['page'] == '41'
    assert quotes_by_ref['Bretonnia Army Book']['citation']['page'] == '43'


def test_find_quotes_in_text_lazy_continuation_non_indented():
    """Non-indented blockquote followed immediately by a non-'>' line uses lazy continuation."""
    text = (
        "> The side that loses a combat must take a Break test to determine\n"
        "whether it stands and fights.\n"
        "\n"
        "(Warhammer Rulebook, p. 41)"
    )
    quotes = find_quotes_in_text(text)
    assert len(quotes) == 1
    q = quotes[0]
    assert 'whether it stands and fights' in q['quote']
    assert q['citation'] is not None
    assert q['citation']['page'] == '41'


def test_find_quotes_in_text_lazy_continuation_not_applied_to_indented():
    """Indented blockquote followed by a non-'>' line does NOT include the continuation line."""
    # Mirrors the real failure: indented blockquote inside a numbered list where
    # the next line is a list bullet that must NOT be absorbed into the quote.
    text = (
        "1. Apply the separation rule:\n\n"
        "     > However, a Break test is not a psychology test. The two tests are quite separate.\n"
        "     >\n"
        "     > (Warhammer Rulebook, p. 46)\n"
        "   - Therefore, Break tests are **not** considered part of the psychology rules."
    )
    quotes = find_quotes_in_text(text)
    assert len(quotes) == 1
    q = quotes[0]
    # The quote text must not include the list bullet
    assert 'Therefore' not in q['text']
    assert 'Therefore' not in q['quote']
    # The real rule text and citation must be present
    assert 'Break test is not a psychology test' in q['quote']
    assert q['citation'] is not None
    assert q['citation']['ref_name'] == 'Warhammer Rulebook'
    assert q['citation']['page'] == '46'


def test_find_quotes_in_text_prose_then_citation_blockquote():
    """Prose line immediately before a citation-only blockquote is absorbed as the quote text."""
    text = (
        "Grail Knights have the Grail Virtue; they have drunk from the sacred grail and are immune to psychology.\n"
        "> (Bretonnia Army Book, p. 62)"
    )
    quotes = find_quotes_in_text(text)
    assert len(quotes) == 1
    q = quotes[0]
    assert q['quote'] == 'Grail Knights have the Grail Virtue; they have drunk from the sacred grail and are immune to psychology.'
    assert q['citation'] is not None
    assert q['citation']['ref_name'] == 'Bretonnia Army Book'
    assert q['citation']['page'] == '62'


def test_find_quotes_in_text_prose_then_blank_bq_then_citation_blockquote():
    """Prose line followed by a blank > line then citation-only blockquote is absorbed."""
    text = (
        "Grail Knights have the Grail Virtue; they have drunk from the sacred grail and are immune to psychology.\n"
        "> \n"
        "> (Bretonnia Army Book, p. 62)"
    )
    quotes = find_quotes_in_text(text)
    assert len(quotes) == 1
    q = quotes[0]
    assert q['quote'] == 'Grail Knights have the Grail Virtue; they have drunk from the sacred grail and are immune to psychology.'
    assert q['citation'] is not None
    assert q['citation']['ref_name'] == 'Bretonnia Army Book'
    assert q['citation']['page'] == '62'


def test_find_quotes_in_text_prose_blank_line_then_citation_blockquote():
    """Prose line separated by a blank line from a citation-only blockquote is absorbed."""
    text = (
        "Grail Knights have the Grail Virtue; they have drunk from the sacred grail and are immune to psychology.\n"
        "\n"
        "> (Bretonnia Army Book, p. 62)"
    )
    quotes = find_quotes_in_text(text)
    assert len(quotes) == 1
    q = quotes[0]
    assert q['quote'] == 'Grail Knights have the Grail Virtue; they have drunk from the sacred grail and are immune to psychology.'
    assert q['citation'] is not None
    assert q['citation']['ref_name'] == 'Bretonnia Army Book'
    assert q['citation']['page'] == '62'


def test_find_quotes_in_text_citation_blockquote_no_preceding_prose():
    """A citation-only blockquote with no preceding prose line is left as-is (empty quote)."""
    text = "> (Bretonnia Army Book, p. 62)"
    quotes = find_quotes_in_text(text)
    assert len(quotes) == 1
    q = quotes[0]
    assert q['quote'] == ''
    assert q['citation'] is not None
    assert q['citation']['ref_name'] == 'Bretonnia Army Book'
    assert q['citation']['page'] == '62'
