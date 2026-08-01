from langchain.messages import ToolMessage
from langchain_core.messages.content import ToolCall
from langchain_core.runnables import chain
import pytest
from meeplemate.qa_graph import (
    Chunk,
    ChunkSearchResult,
    FixQuoteInput,
    FixQuotesResult,
    LocatedQuote,
    PlainText,
    QuoteMatch,
    QuoteReplacement,
    QuoteSegment,
    QuoteEntry,
    ValidateAndFixResponseOutput,
    ValidateAndFixResponseInput,
    _strip_invalid_quotes,
    apply_replacements,
    are_segments_adjacent,
    build_segments,
    dedupe_chunks,
    dedupe_chunks_in_message_history,
    extracted_quote_to_quote_entry,
    fix_quote_citations_in_text,
    format_blockquote_with_inline_citation,
    format_quote,
    get_chunk_id_tuple,
    locate_quotes,
    materialize,
    remove_quote_segments,
    sort_chunks,
    unescape_table_html,
    validate_and_fix_response,
)
from langchain_core.messages import AIMessage
from typing import List
import json
import inspect

from meeplemate.quote_util import find_quotes_in_text

def test_dedupe_chunks_in_message_history():
    result1 = ChunkSearchResult(
        chunk=Chunk(
            rulebook_name="Rulebook 1",
            page="1",
            start_index=3,
            end_index=-1,
            content="This is chunk 1",
        ),
        relevance_reason="Relevant to the query",
    )
    result2 = ChunkSearchResult(
        chunk=Chunk(
            rulebook_name="Rulebook 1",
            page="2",
            start_index=5,
            end_index=-1,
            content="This is chunk 2",
        ),
        relevance_reason="Also relevant to the query",
    )
    result3 = ChunkSearchResult(
        chunk=Chunk(
            rulebook_name="Rulebook 2",
            page="1",
            start_index=0,
            end_index=-1,
            content="This is chunk 3",
        ),
        relevance_reason="Somewhat relevant to the query",
    )

    tool_messages = [
        ToolMessage(
            content=json.dumps([result1, result2]),
            tool_call_id="7",
            status="success",
            id="tool_msg_1"
        ),
        ToolMessage(
            content=json.dumps([result2, result3]),
            tool_call_id="8",
            status="success",
            id="tool_msg_2"
        ),
    ]

    ai_message = AIMessage(
        content="",
        tool_calls=[
            ToolCall(name="search_chunks", args={}, id="7", type="tool_call"),
            ToolCall(name="search_chunks", args={}, id="8", type="tool_call")
        ],
        metadata={"related_tool_messages": tool_messages},
    )

    message_edits = dedupe_chunks_in_message_history([ai_message] + tool_messages)
    assert len(message_edits) == 1
    message_edit = message_edits[0]
    assert message_edit.id == "tool_msg_1"
    deduped_results = json.loads(message_edit.content)
    assert len(deduped_results) == 1
    assert get_chunk_id_tuple(deduped_results[0]["chunk"]) == get_chunk_id_tuple(result1["chunk"])



def test_validation_blockquote_citation_already_inline():
    """Test blockquotes with inline citations remain unchanged"""
    chunks: List[Chunk] = [
        {
            'content': 'Items can be sold for levels.',
            'start_index': 0,
            'end_index': -1,
            'page': "1",
            'rulebook_name': 'Rules'
        }
    ]

    final_answer = '> "Items can be sold for levels." (Rules, p. 1)'

    result = fix_quote_citations_in_text(final_answer, chunks)

    # Should have no invalid quotes
    assert len(result.unfixable_quotes) == 0

    # Should be reformatted with citation on its own blockquote line, preceded by verified div marker
    assert materialize(result.segments, wrap_verified=True) == '<div data-quote-status="verified"></div>\n\n> "Items can be sold for levels."\n> \n> (Rules, p. 1)'


def test_validation_mixed_quote_types():
    """Test blockquotes and inline quotes together"""
    chunks: List[Chunk] = [
        {
            'content': 'Blockquote text here.',
            'start_index': 0,
            'end_index': -1,
            'page': "1",
            'rulebook_name': 'Book A'
        },
        {
            'content': 'Inline quote text here.',
            'start_index': 100,
            'end_index': -1,
            'page': "2",
            'rulebook_name': 'Book B'
        }
    ]

    final_answer = '> "Blockquote text here."\n\n(Book A, p. 1)\n\nAdditionally, "Inline quote text here." (Book B, p. 2)'

    result = fix_quote_citations_in_text(final_answer, chunks)

    # Should have no invalid quotes
    assert len(result.unfixable_quotes) == 0

    # Blockquote should have citation on its own blockquote line
    assert '> "Blockquote text here."\n> \n> (Book A, p. 1)' in result.fixed_text
    # Inline quote should remain unchanged
    assert '"Inline quote text here." (Book B, p. 2)' in result.fixed_text


def test_strip_invalid_quotes_preserves_verified_quotes_and_prose():
    """The exhausted-retry fallback must fail closed without losing the answer."""
    chunks: List[Chunk] = [
        {
            "content": "Items can be sold for levels.",
            "start_index": 0,
            "end_index": 29,
            "page": "1",
            "rulebook_name": "Rules",
        }
    ]
    response = inspect.cleandoc(
        '''\
        You may sell items during your turn.

        > "Items can be sold for levels."
        >
        > (Rules, p. 1)

        This second claim has no supporting source.

        > "Invented rulebook text."
        >
        > (Rules, p. 99)
        '''
    )

    fix_result = fix_quote_citations_in_text(response, chunks)
    result = _strip_invalid_quotes(fix_result, chunks)

    assert "Items can be sold for levels." in result["response"]
    assert "This second claim has no supporting source." in result["response"]
    assert "Invented rulebook text." not in result["response"]
    assert result["invalid_quotes"] == []


def test_validation_preserves_standalone_citation():
    """A bare citation in prose with no preceding blockquote is left unchanged.

    The standalone-citation cleanup was removed. A citation like (Book, p. 99)
    that appears in prose text without any associated blockquote is not
    recognised as a quote and is therefore left as-is. The retry mechanism
    (unfixable_quotes) handles genuine hallucinated quotes that do have
    blockquotes; bare citations in prose are simply preserved.
    """
    chunks: List[Chunk] = [
        {
            'content': 'Some rule text.',
            'start_index': 0,
            'end_index': -1,
            'page': "1",
            'rulebook_name': 'Book'
        }
    ]

    final_answer = 'Some text here.\n\n(Book, p. 99)\n\nMore text.'

    result = fix_quote_citations_in_text(final_answer, chunks)

    # No blockquote precedes this citation, so it is not captured as a quote
    # and the text is left unchanged.
    assert result.fixed_text == final_answer


def test_multiline_blockquote_citation_next_line():
    warpstorm_scroll = inspect.cleandoc(
        '''\
        ## WARPSTORM SCROLL
        Bearer can cast spell in his magic phase. All creatures flying high's suffer D6 56 hits, and are forced down to earth, re- entering the table on their own side's table edge in their following turn.

        One use only. Skaven wizards only
        '''
    )

    flying_high = inspect.cleandoc(
        '''\
        ## FLYING HIGH
        A flyer may choose to fly high during his turn instead of making a normal flying move. This represents a flyer ascending far into the air above the battlefield. All that observers on the ground can see is a tiny black dot amongst the clouds. These rules represent models flying high, diving down onto the battlefield, and engaging in combat against each other above the clouds.
        '''
    )

    chunks: list[Chunk] = [
        {
            "content": warpstorm_scroll,
            "rulebook_name": "Warhammer Magic",
            "page": "44",
            "start_index": 0,
            "end_index": -1,
        },
        {
            "content": flying_high,
            "rulebook_name": "Warhammer Rulebook",
            "page": "74",
            "start_index": 0,
            "end_index": -1
        }
    ]

    before = inspect.cleandoc(
        '''\
        Yes, you can use the Warpstorm Scroll against a model that is flying high. According to the definition of the Warpstorm Scroll:
        
        > "## WARPSTORM SCROLL
        Bearer can cast spell in his magic phase. All creatures flying high's suffer D6 56 hits, and are forced down to earth, re- entering the table on their own side's table edge in their following turn."
        
        (Warhammer Magic, p. 44)
        
        The rule explicitly states that the scroll affects "creatures flying high," which aligns with the definition of flying high:
        
        > "## FLYING HIGH
        A flyer may choose to fly high during his turn instead of making a normal flying move. This represents a flyer ascending far into the air above the battlefield..."
        
        (Warhammer Rulebook, p. 74)
        
        Thus, the effect of the Warpstorm Scroll applies to models in the flying high status.
        '''
    )

    expected = inspect.cleandoc(
        '''\
        Yes, you can use the Warpstorm Scroll against a model that is flying high. According to the definition of the Warpstorm Scroll:

        <div data-quote-status="verified"></div>

        > "## WARPSTORM SCROLL
        > Bearer can cast spell in his magic phase. All creatures flying high's suffer D6 56 hits, and are forced down to earth, re- entering the table on their own side's table edge in their following turn."
        >\x20
        > (Warhammer Magic, p. 44)

        The rule explicitly states that the scroll affects "creatures flying high," which aligns with the definition of flying high:

        <div data-quote-status="verified"></div>

        > "## FLYING HIGH
        > A flyer may choose to fly high during his turn instead of making a normal flying move. This represents a flyer ascending far into the air above the battlefield..."
        >\x20
        > (Warhammer Rulebook, p. 74)

        Thus, the effect of the Warpstorm Scroll applies to models in the flying high status.
        '''
    )

    result = fix_quote_citations_in_text(before, chunks)

    assert materialize(result.segments, wrap_verified=True) == expected


def test_fix_quote_citations_in_text():
    chunk_content = inspect.cleandoc(
        '''\
        Some rulebook text. It goes on and on. For some sentences.

        Look, it has paragraphs too!

        Here is another one too!
        '''
    )

    chunks: list[Chunk] = [
        {
            "content": chunk_content,
            "rulebook_name": "Some Rulebook",
            "page": "44",
            "start_index": 0,
            "end_index": -1,
        },
    ]

    before = inspect.cleandoc(
        '''\
        According to the rules:

        > "Some rulebook text. It goes on and on. For some sentences."

        (Some Rulebook, p. 44)

        This indicates that the rule has multiple paragraphs.

        > "Look, it has paragraphs too!"
        > (Some Rulebook, p. 44)

        Finally, we see:

        > "Here is another one too!"
        >
        > (Some Rulebook, p. 44)

        Some concluding text.
        '''
    )
    expected = inspect.cleandoc(
        '''\
        According to the rules:

        > "Some rulebook text. It goes on and on. For some sentences."
        >\x20
        > (Some Rulebook, p. 44)

        This indicates that the rule has multiple paragraphs.

        > "Look, it has paragraphs too!"
        >\x20
        > (Some Rulebook, p. 44)

        Finally, we see:

        > "Here is another one too!"
        >\x20
        > (Some Rulebook, p. 44)

        Some concluding text.
        '''
    )

    result = fix_quote_citations_in_text(before, chunks)
    assert result.fixed_text == expected
    # Segment model: three verified blockquotes → three QuoteSegments
    quote_segs = [s for s in result.segments if isinstance(s, QuoteSegment)]
    assert len(quote_segs) == 3
    assert all(s.located.is_verified for s in quote_segs)
    # materialize with wrap_verified=True produces div wrappers
    wrapped = materialize(result.segments, wrap_verified=True)
    assert wrapped.count('<div data-quote-status="verified"></div>') == 3


def test_fix_quote_citations_in_text_2():

    chunks: list[Chunk] = [
        {
            "content": "The side that loses a combat must take a test to determine whether it stands and fights or turns tail and runs away This is called a Break test.",
            "rulebook_name": "Warhammer Rulebook",
            "page": "42",
            "start_index": 0,
            "end_index": -1,
        },
        {
            "content": "The unit never needs test for any of the psychology rules, whether panic, fear, terror or whatever. The Knights are unaffected by any psychology.",
            "rulebook_name": "Bretonnia Army Book",
            "page": "49",
            "start_index": 0,
            "end_index": -1,
        },
        {
            "content": "However, a Break test is not a psychology test. The two tests are quite separate. This is important because some bonuses apply specifically to Break tests and others apply specifically to psychology tests.",
            "rulebook_name": "Warhammer Rulebook",
            "page": "47",
            "start_index": 0,
            "end_index": -1,
        },
    ]

    before = inspect.cleandoc(
        '''\
        **Yes, the Grail Knights must take a Break test when they lose combat against a Green Dragon.**

        The rulebook clearly states that losing combat triggers a Break test:

        > The side that loses a combat must take a test to determine whether it stands and fights or turns tail and runs away This is called a Break test.

        (Warhammer Rulebook, p. 42)

        This means that regardless of the enemy type or unit abilities, any unit that loses a combat must perform a Break test.

        However, the Grail Knights have a special immunity that protects them from psychological effects:

        > The unit never needs test for any of the psychology rules, whether panic, fear, terror or whatever. The Knights are unaffected by any psychology.

        (Bretonnia Army Book, p. 49)

        This protection applies specifically to psychology-related tests, including those caused by Fear, Panic, or similar effects.

        But the Break test is explicitly not a psychology test:

        > However, a Break test is not a psychology test. The two tests are quite separate. This is important because some bonuses apply specifically to Break tests and others apply specifically to psychology tests.

        (Warhammer Rulebook, p. 47)

        Since the Break test is functionally distinct from psychology tests, and the Grail Knights’ immunity only applies to psychology rules, their protection does not extend to Break tests.

        Therefore, even though the Grail Knights are immune to the Green Dragon’s corrosive fumes (which trigger a Leadership test “in the same way as for a fear or other psychology test”), this does not affect the outcome of the Break test after combat loss.

        In conclusion: Losing combat triggers a Break test, and since the Break test is not a psychology test, the Grail Knights’ immunity does not prevent them from taking it.
        '''
    )

    expected = inspect.cleandoc(
        '''\
        **Yes, the Grail Knights must take a Break test when they lose combat against a Green Dragon.**

        The rulebook clearly states that losing combat triggers a Break test:

        > The side that loses a combat must take a test to determine whether it stands and fights or turns tail and runs away This is called a Break test.
        >\x20
        > (Warhammer Rulebook, p. 42)

        This means that regardless of the enemy type or unit abilities, any unit that loses a combat must perform a Break test.

        However, the Grail Knights have a special immunity that protects them from psychological effects:

        > The unit never needs test for any of the psychology rules, whether panic, fear, terror or whatever. The Knights are unaffected by any psychology.
        >\x20
        > (Bretonnia Army Book, p. 49)

        This protection applies specifically to psychology-related tests, including those caused by Fear, Panic, or similar effects.

        But the Break test is explicitly not a psychology test:

        > However, a Break test is not a psychology test. The two tests are quite separate. This is important because some bonuses apply specifically to Break tests and others apply specifically to psychology tests.
        >\x20
        > (Warhammer Rulebook, p. 47)

        Since the Break test is functionally distinct from psychology tests, and the Grail Knights’ immunity only applies to psychology rules, their protection does not extend to Break tests.

        Therefore, even though the Grail Knights are immune to the Green Dragon’s corrosive fumes (which trigger a Leadership test “in the same way as for a fear or other psychology test”), this does not affect the outcome of the Break test after combat loss.

        In conclusion: Losing combat triggers a Break test, and since the Break test is not a psychology test, the Grail Knights’ immunity does not prevent them from taking it.
        '''
    )

    result = fix_quote_citations_in_text(before, chunks)
    assert result.fixed_text == expected

def test_fix_quote_citations_in_text_3():
    chunks: list[Chunk] = [
        {
            "content": "The side that loses a combat must take a test to determine whether it stands and fights or turns tail and runs away. This is called a Break test.",
            "rulebook_name": "Warhammer Rulebook",
            "page": "42",
            "start_index": 0,
            "end_index": -1,
        },
        {
            "content": "Grail Knights have the most noble chivalric virtue of all – the Grail Virtue. This means that they are unaffected by any of the psychology rules; any such tests they are called upon to take are disregarded with a cool and steely countenance. The Knight knows neither fear nor terror, nor will he panic, for the grail sustains his noble will better than any magic trickery.",
            "rulebook_name": "Bretonnia Army Book",
            "page": "44",
            "start_index": 0,
            "end_index": -1,
        },
        {
            "content": "However, a Break test is not a psychology test. The two tests are quite separate. This is important because some bonuses apply specifically to Break tests and others apply specifically to psychology tests.",
            "rulebook_name": "Warhammer Rulebook",
            "page": "47",
            "start_index": 0,
            "end_index": -1,
        },
    ]

    before = '**Yes, Grail Knights must take a Break test when they lose combat, despite their immunity to psychological effects.**\n\nThe general rule for losing combat requires a Break test:\n\n> The side that loses a combat must take a test to determine whether it stands and fights or turns tail and runs away. This is called a Break test. You need to take a separate Break test for every unit involved in the combat.\n\n(Warhammer Rulebook, p. 42)\n\nThis means that any unit which loses a combat must attempt a Break test, regardless of other traits.\n\nHowever, Grail Knights possess the Grail Virtue, which grants immunity to psychological effects:\n\n> Grail Knights have the most noble chivalric virtue of all – the Grail Virtue. This means that they are unaffected by any of the psychology rules; any such tests they are called upon to take are disregarded with a cool and steely countenance. The Knight knows neither fear nor terror, nor will he panic, for the grail sustains his noble will better than any magic trickery.\n\n(Bretonnia Army Book, p. 44)\n\nThe key distinction lies in the categorization of Break tests:\n\n> However, a Break test is not a psychology test. The two tests are quite separate. This is important because some bonuses apply specifically to Break tests and others apply specifically to psychology tests.\n\n(Warhammer Rulebook, p. 47)\n\nSince Break tests are explicitly stated to be *not* psychology tests, and the Grail Virtue only applies to "psychology rules" and "such tests" — which refer exclusively to Panic, Fear, Terror, and Stupidity — the immunity does not extend to Break tests.\n\nTherefore, even though Grail Knights are immune to psychological effects, they are still required to take a Break test when they lose combat, as the rule for Break tests is not overridden by the Grail Virtue.'

    # The first quote contains an extra fabricated sentence not present in the chunk
    # ("You need to take a separate Break test..."), so it cannot be verified and its
    # citation stays as-is (bare paragraph). The other two quotes match exactly and
    # have their citations reformatted inline.
    fixed = '**Yes, Grail Knights must take a Break test when they lose combat, despite their immunity to psychological effects.**\n\nThe general rule for losing combat requires a Break test:\n\n> The side that loses a combat must take a test to determine whether it stands and fights or turns tail and runs away. This is called a Break test. You need to take a separate Break test for every unit involved in the combat.\n\n(Warhammer Rulebook, p. 42)\n\nThis means that any unit which loses a combat must attempt a Break test, regardless of other traits.\n\nHowever, Grail Knights possess the Grail Virtue, which grants immunity to psychological effects:\n\n> Grail Knights have the most noble chivalric virtue of all – the Grail Virtue. This means that they are unaffected by any of the psychology rules; any such tests they are called upon to take are disregarded with a cool and steely countenance. The Knight knows neither fear nor terror, nor will he panic, for the grail sustains his noble will better than any magic trickery.\n> \n> (Bretonnia Army Book, p. 44)\n\nThe key distinction lies in the categorization of Break tests:\n\n> However, a Break test is not a psychology test. The two tests are quite separate. This is important because some bonuses apply specifically to Break tests and others apply specifically to psychology tests.\n> \n> (Warhammer Rulebook, p. 47)\n\nSince Break tests are explicitly stated to be *not* psychology tests, and the Grail Virtue only applies to "psychology rules" and "such tests" — which refer exclusively to Panic, Fear, Terror, and Stupidity — the immunity does not extend to Break tests.\n\nTherefore, even though Grail Knights are immune to psychological effects, they are still required to take a Break test when they lose combat, as the rule for Break tests is not overridden by the Grail Virtue.'

    result = fix_quote_citations_in_text(before, chunks)
    assert result.fixed_text == fixed
    assert len(result.unfixable_quotes) == 1


def test_fix_quote_citations_duplicate_chunks():
    """Regression test: duplicate chunks for same page must not corrupt text after blockquotes."""
    chunk_content = "Some quote text here matching chunk content."

    # Multiple chunks for the same (rulebook, page) key
    chunks: list[Chunk] = [
        {"content": chunk_content, "rulebook_name": "Some Rulebook", "page": "42", "start_index": 0, "end_index": -1},
        {"content": chunk_content, "rulebook_name": "Some Rulebook", "page": "42", "start_index": -1, "end_index": -1},
        {"content": chunk_content, "rulebook_name": "Some Rulebook", "page": "42", "start_index": -2, "end_index": -1},
    ]

    before = inspect.cleandoc(
        '''\
        According to the rules:

        > Some quote text here matching chunk content.

        (Some Rulebook, p. 42)

        This is the text after the quote.
        '''
    )
    expected = inspect.cleandoc(
        '''\
        According to the rules:

        > Some quote text here matching chunk content.
        >\x20
        > (Some Rulebook, p. 42)

        This is the text after the quote.
        '''
    )

    result = fix_quote_citations_in_text(before, chunks)
    assert result.fixed_text == expected


def test_fix_quote_separated_citation():
    chunks: list[Chunk] = [
        {
            "content": "A. No, the encounter ends immediately and you do not have to kill all the minions either.",
            "rulebook_name": "Encounter Rule Book",
            "page": "35",
            "start_index": 0,
            "end_index": -1,
        },
    ]

    before = inspect.cleandoc(
        '''\
        Yes, the encounter ends immediately when the last HP die is removed, even if other reactions were supposed to be drawn first.

        > A. No, the encounter ends immediately and you do not have to kill all the minions either.

        > (Encounter Rule Book, p. 33)

        This rule explicitly states that the encounter ends immediately upon removing the last HP die, with no requirement to resolve further actions such as drawing reactions or killing remaining minions. The rulebook clarifies that no additional steps are needed after the last die is broken.
        '''
    )

    expected = inspect.cleandoc(
        '''\
        Yes, the encounter ends immediately when the last HP die is removed, even if other reactions were supposed to be drawn first.

        > A. No, the encounter ends immediately and you do not have to kill all the minions either.
        >\x20
        > (Encounter Rule Book, p. 35)

        This rule explicitly states that the encounter ends immediately upon removing the last HP die, with no requirement to resolve further actions such as drawing reactions or killing remaining minions. The rulebook clarifies that no additional steps are needed after the last die is broken.
        '''
    )

    result = fix_quote_citations_in_text(before, chunks)
    assert result.fixed_text == expected


def test_fix_quote_citations_unverifiable_blockquote_citation():
    """Quote not found in any document goes to unfixable_quotes and text is unchanged.

    Previously, an unverifiable blockquote whose citation appeared as a separate
    > (citation) blockquote was silently moved to valid_or_fixed as a defensive
    measure against the standalone-citation cleanup. That edge case is removed:
    unverifiable quotes now always land in unfixable_quotes.
    """
    chunks: list[Chunk] = [
        {
            "content": "Some completely unrelated rulebook content about goblins.",
            "rulebook_name": "Goblin Rulebook",
            "page": "7",
            "start_index": 0,
            "end_index": -1,
        }
    ]
    text = inspect.cleandoc("""\
        Here is the answer.

        > This rule text does not appear anywhere in our documents whatsoever.

        > (Goblin Rulebook, p. 7)

        Some follow-up text.
    """)
    result = fix_quote_citations_in_text(text, chunks)
    assert len(result.unfixable_quotes) == 1
    assert result.unfixable_quotes[0]["quote"] == "This rule text does not appear anywhere in our documents whatsoever."
    assert len(result.valid_quotes) == 0
    assert result.fixed_text == text


def test_fix_quote_citations_no_dangling_citation_created():
    """Correcting a wrong citation does not leave a dangling standalone citation.

    Previously, find_quotes_in_text could miss a citation that appeared on a
    separate paragraph, causing fix_quote_citations_in_text to insert a second
    citation and leave the original as a dangling standalone (which then required
    its own cleanup pass). With citations now captured within the quote span,
    the correction is made in-place and nothing is left dangling.
    """
    chunks: list[Chunk] = [
        {
            "content": "You may move up to three spaces on your turn.",
            "rulebook_name": "Movement Rules",
            "page": "12",
            "start_index": 0,
            "end_index": -1,
        }
    ]
    text = inspect.cleandoc("""\
        The movement rule states:

        > You may move up to three spaces on your turn.

        (Movement Rules, p. 99)

        End of answer.
    """)
    result = fix_quote_citations_in_text(text, chunks)
    # Citation should be corrected to the real page
    assert '(Movement Rules, p. 12)' in result.fixed_text
    # The wrong citation must not remain anywhere in the text
    assert '(Movement Rules, p. 99)' not in result.fixed_text
    # And there should be no orphaned bare citation paragraph anywhere
    assert '\n\n(' not in result.fixed_text


# ── locate_quotes ────────────────────────────────────────────────────────────

def test_locate_quotes_found():
    chunks: list[Chunk] = [{
        "content": "You may move up to three spaces on your turn.",
        "rulebook_name": "Movement Rules",
        "page": "12",
        "start_index": 0,
        "end_index": -1,
    }]
    text = "> You may move up to three spaces on your turn.\n\n(Movement Rules, p. 12)"
    from meeplemate import quote_util
    quotes = quote_util.find_quotes_in_text(text)
    located = locate_quotes(quotes, chunks)
    assert len(located) == 1
    lq = located[0]
    assert lq.is_verified
    assert lq.match is not None
    assert lq.match.source_chunk["rulebook_name"] == "Movement Rules"
    assert lq.match.source_chunk["page"] == "12"


def test_locate_quotes_not_found():
    chunks: list[Chunk] = [{
        "content": "Completely unrelated content.",
        "rulebook_name": "Some Book",
        "page": "1",
        "start_index": 0,
        "end_index": -1,
    }]
    text = "> This text does not appear in any document.\n\n(Some Book, p. 1)"
    from meeplemate import quote_util
    quotes = quote_util.find_quotes_in_text(text)
    located = locate_quotes(quotes, chunks)
    assert len(located) == 1
    assert not located[0].is_verified
    assert located[0].match is None


def test_locate_quotes_matched_text():
    """match.matched_text is the literal document text, not the LLM's wording."""
    chunk_content = "You may move up to three spaces on your turn."
    chunks: list[Chunk] = [{
        "content": chunk_content,
        "rulebook_name": "Movement Rules",
        "page": "12",
        "start_index": 0,
        "end_index": -1,
    }]
    text = "> You may move up to three spaces on your turn.\n\n(Movement Rules, p. 12)"
    from meeplemate import quote_util
    quotes = quote_util.find_quotes_in_text(text)
    located = locate_quotes(quotes, chunks)
    assert located[0].is_verified
    assert located[0].match.matched_text == chunk_content


def test_locate_quotes_referenced_chunks():
    """match.referenced_chunks contains every chunk overlapping the match span."""
    chunks: list[Chunk] = [
        {"content": "First sentence here.", "rulebook_name": "Book", "page": "1", "start_index": 0, "end_index": 20},
        {"content": "Second sentence here.", "rulebook_name": "Book", "page": "2", "start_index": 21, "end_index": 42},
    ]
    # Quote that spans both chunks
    text = "> First sentence here. Second sentence here.\n\n(Book, p. 1)"
    from meeplemate import quote_util
    quotes = quote_util.find_quotes_in_text(text)
    located = locate_quotes(quotes, chunks)
    assert located[0].is_verified
    assert len(located[0].match.referenced_chunks) == 2, (
        "Both chunks should be in referenced_chunks when the quote spans two chunks"
    )


def test_fix_quote_citations_referenced_chunks_multi_chunk():
    """fix_quote_citations_in_text.referenced_chunks includes all chunks overlapping a quote."""
    chunks: list[Chunk] = [
        {"content": "First sentence here.", "rulebook_name": "Book", "page": "1", "start_index": 0, "end_index": 20},
        {"content": "Second sentence here.", "rulebook_name": "Book", "page": "1", "start_index": 21, "end_index": 42},
        {"content": "Unrelated content.", "rulebook_name": "Book", "page": "1", "start_index": 43, "end_index": 61},
    ]
    # Quote spans chunk 0 and chunk 1; chunk 2 is not touched
    text = "> First sentence here. Second sentence here.\n\n(Book, p. 1)"
    result = fix_quote_citations_in_text(text, chunks)
    assert len(result.unfixable_quotes) == 0
    ref_contents = {c["content"] for c in result.referenced_chunks}
    assert "First sentence here." in ref_contents, "First chunk should be in referenced_chunks"
    assert "Second sentence here." in ref_contents, "Second chunk should be in referenced_chunks"
    assert "Unrelated content." not in ref_contents, "Unrelated chunk should not be in referenced_chunks"


def test_locate_quotes_match_ratio():
    """match_ratio is between 0 and 1; exact match gives a ratio near 1."""
    chunks: list[Chunk] = [{
        "content": "You may move up to three spaces on your turn.",
        "rulebook_name": "Movement Rules",
        "page": "12",
        "start_index": 0,
        "end_index": -1,
    }]
    text = "> You may move up to three spaces on your turn.\n\n(Movement Rules, p. 12)"
    from meeplemate import quote_util
    quotes = quote_util.find_quotes_in_text(text)
    located = locate_quotes(quotes, chunks)
    assert located[0].is_verified
    ratio = located[0].match.match_ratio
    assert 0.0 <= ratio <= 1.0
    assert ratio > 0.9  # near-exact match


def test_locate_quotes_ellipsis_span():
    """Quotes with ellipsis are matched despite a larger span in the combined text."""
    chunks: list[Chunk] = [
        {"content": "First part of the rule.", "rulebook_name": "Book", "page": "1", "start_index": 0, "end_index": 23},
        {"content": "Some intervening text that is not quoted.", "rulebook_name": "Book", "page": "2", "start_index": 24, "end_index": 64},
        {"content": "Last part of the rule.", "rulebook_name": "Book", "page": "3", "start_index": 65, "end_index": 87},
    ]
    text = "> First part of the rule...Last part of the rule.\n\n(Book, p. 1)"
    from meeplemate import quote_util
    quotes = quote_util.find_quotes_in_text(text)
    located = locate_quotes(quotes, chunks)
    assert located[0].is_verified


# ── format_quote ─────────────────────────────────────────────────────────────

def _make_chunk(rulebook: str, page: str, content: str) -> Chunk:
    return {"content": content, "rulebook_name": rulebook, "page": page, "start_index": 0, "end_index": -1}


def _make_located(text: str, chunk: Chunk | None) -> LocatedQuote:
    from meeplemate import quote_util
    quotes = quote_util.find_quotes_in_text(text)
    assert len(quotes) == 1
    match = None
    if chunk is not None:
        match = QuoteMatch(
            matched_text=chunk["content"],
            source_chunk=chunk,
            referenced_chunks=[chunk],
            match_ratio=1.0,
            matched_span=chunk,
        )
    return LocatedQuote(quote=quotes[0], match=match)


def test_format_quote_unverified():
    chunk = _make_chunk("Book", "1", "Some rule text.")
    text = "> Some rule text.\n\n(Book, p. 99)"
    lq = _make_located(text, None)  # not verified
    r = format_quote(lq)
    assert not r.is_verified
    assert r.replacement == lq.quote["text"]


def test_format_quote_correct_citation_inline():
    """Already-correct citation retains correct citation text in the replacement."""
    chunk = _make_chunk("Book", "1", "Some rule text.")
    text = "> Some rule text.\n> \n> (Book, p. 1)"
    lq = _make_located(text, chunk)
    r = format_quote(lq)
    assert r.is_verified
    assert '(Book, p. 1)' in r.replacement
    assert '(Book, p. 2)' not in r.replacement


def test_format_quote_correct_citation_reformat():
    """Correct citation on a separate paragraph → reformatted inline in blockquote."""
    chunk = _make_chunk("Book", "1", "Some rule text.")
    text = "> Some rule text.\n\n(Book, p. 1)"
    lq = _make_located(text, chunk)
    r = format_quote(lq)
    assert r.is_verified
    assert '> (Book, p. 1)' in r.replacement
    assert '\n\n(Book, p. 1)' not in r.replacement


def test_format_quote_wrong_citation():
    """Wrong page number → corrected to the chunk's actual page."""
    chunk = _make_chunk("Book", "42", "Some rule text.")
    text = "> Some rule text.\n\n(Book, p. 99)"
    lq = _make_located(text, chunk)
    r = format_quote(lq)
    assert r.is_verified
    assert '(Book, p. 42)' in r.replacement
    assert '(Book, p. 99)' not in r.replacement


def test_format_quote_missing_citation():
    """No citation at all → correct citation is inserted."""
    chunk = _make_chunk("Book", "7", "Some rule text.")
    text = "> Some rule text."
    lq = _make_located(text, chunk)
    r = format_quote(lq)
    assert r.is_verified
    assert '(Book, p. 7)' in r.replacement


def test_format_quote_no_extra_blank_line_added():
    """Citation already in blockquote with blank-line separator — no extra > line inserted.

    Regression: format_blockquote_with_inline_citation was producing
    `\\n> \\n>\\n> \\n> (citation)` instead of `\\n> \\n> (citation)` because
    the blank-line cleanup treated `> ` (blockquote blank) as non-blank (its
    .strip() is '>' not '').
    """
    chunk = _make_chunk("Book", "1", "Some rule text.")
    text = "> Some rule text.\n> \n> (Book, p. 1)"
    lq = _make_located(text, chunk)
    r = format_quote(lq)
    assert r.is_verified
    assert r.replacement == "> Some rule text.\n> \n> (Book, p. 1)", (
        f"Expected clean citation format but got: {r.replacement!r}"
    )


def test_fix_quote_citations_em_dash_for_hyphen():
    """LLM uses em-dash (–) where the document has a hyphen (-).

    Regression: both characters normalize to a word-boundary space, so the
    quote should validate successfully despite the substitution.
    """
    chunk: Chunk = {
        "rulebook_name": "Bretonnia Army Book",
        "page": "43",
        "start_index": 132609,
        "end_index": 132998,
        "content": (
            "## Grail Virtue\n\n"
            "Grail Knights have the most noble chivalric virtue of all - the Grail Virtue. "
            "This means that they are unaffected by any of the psychology rules; any such "
            "tests they are called upon to take are disregarded with a cool and steely "
            "countenance. The Knight knows neither fear nor terror, nor will he panic, for "
            "the grail sustains his noble will better than any magic trickery."
        ),
    }
    # LLM used an en-dash instead of the document's hyphen
    response = inspect.cleandoc("""\
        Grail Knights are immune to psychology tests but not Break tests.

        > Grail Knights have the most noble chivalric virtue of all \u2013 the Grail Virtue. This means that they are unaffected by any of the psychology rules; any such tests they are called upon to take are disregarded with a cool and steely countenance. The Knight knows neither fear nor terror, nor will he panic, for the grail sustains his noble will better than any magic trickery.
        >
        > (Bretonnia Army Book, p. 43)
    """)
    result = fix_quote_citations_in_text(response, [chunk])
    assert len(result.unfixable_quotes) == 0, (
        f"Em-dash quote should be valid but was flagged unfixable: {result.unfixable_quotes}"
    )


def test_fix_quote_citations_truncated_leading_context():
    """LLM quotes a sentence fragment that drops leading context ('However, ').

    Regression: the LLM quoted 'A Break test is not a psychology test.' starting
    mid-sentence, capitalising the first word. After normalisation the text is a
    verbatim substring of the document, so the match should succeed.
    """
    chunk: Chunk = {
        "rulebook_name": "Warhammer Rulebook",
        "page": "46",
        "start_index": 123428,
        "end_index": 123808,
        "content": (
            "Players will immediately realise that a psychology test is taken in the same "
            "way as a Break test in hand- to- hand combat and uses the same characteristic, "
            "namely Leadership. However, a Break test is not a psychology test. The two "
            "tests are quite separate. This is important because some bonuses apply "
            "specifically to Break tests and others apply specifically to psychology tests."
        ),
    }
    # LLM dropped "However, " and capitalised the first word
    response = inspect.cleandoc("""\
        Break tests are distinct from psychology tests.

        > A Break test is not a psychology test. The two tests are quite separate.
        >
        > (Warhammer Rulebook, p. 46)
    """)
    result = fix_quote_citations_in_text(response, [chunk])
    assert len(result.unfixable_quotes) == 0, (
        f"Truncated-prefix quote should be valid but was flagged unfixable: {result.unfixable_quotes}"
    )


# ── apply_replacements ───────────────────────────────────────────────────────

def test_apply_replacements_single():
    text = "Hello world!"
    replacements = [QuoteReplacement(start_index=6, end_index=11, replacement="Python", is_verified=True, quote_type="inline")]
    assert apply_replacements(text, replacements) == "Hello Python!"


def test_apply_replacements_multiple():
    text = "aaa bbb ccc"
    replacements = [
        QuoteReplacement(start_index=0, end_index=3, replacement="AAA", is_verified=True, quote_type="inline"),
        QuoteReplacement(start_index=8, end_index=11, replacement="CCC", is_verified=True, quote_type="inline"),
    ]
    assert apply_replacements(text, replacements) == "AAA bbb CCC"


def test_apply_replacements_empty():
    text = "unchanged text"
    assert apply_replacements(text, []) == text


# ── materialize ──────────────────────────────────────────────────────────────

def _make_verified_bq_seg(formatted_text: str) -> QuoteSegment:
    from meeplemate import quote_util
    mock_chunk: Chunk = {"content": "x", "rulebook_name": "B", "page": "1", "start_index": 0, "end_index": 1}
    mock_match = QuoteMatch(matched_text="x", source_chunk=mock_chunk, referenced_chunks=[mock_chunk], match_ratio=1.0)
    mock_lq = LocatedQuote(
        quote={"text": formatted_text, "quote": "x", "quote_type": "blockquote", "citation": None, "start_index": 0, "end_index": len(formatted_text)},
        match=mock_match,
    )
    return QuoteSegment(original_text=formatted_text, located=mock_lq, formatted_text=formatted_text)


def _make_unverified_bq_seg(formatted_text: str) -> QuoteSegment:
    mock_lq = LocatedQuote(
        quote={"text": formatted_text, "quote": "x", "quote_type": "blockquote", "citation": None, "start_index": 0, "end_index": len(formatted_text)},
        match=None,
    )
    return QuoteSegment(original_text=formatted_text, located=mock_lq, formatted_text=formatted_text)


def _make_inline_seg(formatted_text: str) -> QuoteSegment:
    from meeplemate import quote_util
    mock_chunk: Chunk = {"content": "x", "rulebook_name": "B", "page": "1", "start_index": 0, "end_index": 1}
    mock_match = QuoteMatch(matched_text="x", source_chunk=mock_chunk, referenced_chunks=[mock_chunk], match_ratio=1.0)
    mock_lq = LocatedQuote(
        quote={"text": formatted_text, "quote": "x", "quote_type": "inline", "citation": None, "start_index": 0, "end_index": len(formatted_text)},
        match=mock_match,
    )
    return QuoteSegment(original_text=formatted_text, located=mock_lq, formatted_text=formatted_text)


def test_materialize_plain_only():
    segs = [PlainText("Hello world")]
    assert materialize(segs) == "Hello world"


def test_materialize_verified_blockquote_wrapped():
    seg = _make_verified_bq_seg("> Some rule.\n> \n> (Book, p. 1)")
    result = materialize([seg], wrap_verified=True)
    assert result.startswith('<div data-quote-status="verified"></div>')
    assert '> Some rule.' in result


def test_materialize_unverified_not_wrapped():
    seg = _make_unverified_bq_seg("> Some rule.")
    result = materialize([seg], wrap_verified=True)
    assert '<div data-quote-status="verified">' not in result
    assert result == "> Some rule."


def test_materialize_inline_not_wrapped():
    seg = _make_inline_seg('"Some rule." (Book, p. 1)')
    result = materialize([seg], wrap_verified=True)
    assert '<div data-quote-status="verified">' not in result


def test_materialize_wrap_verified_false():
    seg = _make_verified_bq_seg("> Rule.\n> \n> (B, p. 1)")
    result = materialize([seg], wrap_verified=False)
    assert '<div data-quote-status="verified">' not in result
    assert result == "> Rule.\n> \n> (B, p. 1)"


def test_materialize_mixed_segments():
    plain = PlainText("Before.\n\n")
    verified_bq = _make_verified_bq_seg("> A rule.\n> \n> (B, p. 1)")
    unverified_bq = _make_unverified_bq_seg("> Unknown.")
    result = materialize([plain, verified_bq, PlainText("\n\n"), unverified_bq])
    assert result.startswith("Before.\n\n")
    assert '<div data-quote-status="verified">' in result
    assert "> Unknown." in result
    # unverified not wrapped
    idx = result.index("> Unknown.")
    assert '<div data-quote-status="verified">' not in result[idx:]


# ── build_segments ────────────────────────────────────────────────────────────

def test_build_segments_no_quotes():
    text = "No quotes here at all."
    segs = build_segments(text, [])
    assert len(segs) == 1
    assert isinstance(segs[0], PlainText)
    assert segs[0].text == text


def test_build_segments_single_quote():
    chunk = _make_chunk("Book", "1", "Some rule text.")
    text = "Before.\n\n> Some rule text.\n> \n> (Book, p. 1)\n\nAfter."
    located = locate_quotes(find_quotes_in_text(text), [chunk])
    segs = build_segments(text, located)
    types = [type(s).__name__ for s in segs]
    assert types == ["PlainText", "QuoteSegment", "PlainText"]
    assert segs[0].text == "Before.\n\n"
    assert isinstance(segs[1], QuoteSegment)
    assert segs[2].text == "\n\nAfter."


def test_build_segments_quote_at_start():
    chunk = _make_chunk("Book", "1", "Some rule text.")
    text = "> Some rule text.\n> \n> (Book, p. 1)\n\nAfter."
    located = locate_quotes(find_quotes_in_text(text), [chunk])
    segs = build_segments(text, located)
    assert isinstance(segs[0], QuoteSegment)


def test_build_segments_quote_at_end():
    chunk = _make_chunk("Book", "1", "Some rule text.")
    text = "Before.\n\n> Some rule text.\n> \n> (Book, p. 1)"
    located = locate_quotes(find_quotes_in_text(text), [chunk])
    segs = build_segments(text, located)
    assert isinstance(segs[-1], QuoteSegment)


def test_build_segments_plain_text_matches_gaps():
    chunk = _make_chunk("Book", "1", "Some rule text.")
    text = "AAA\n\n> Some rule text.\n> \n> (Book, p. 1)\n\nBBB"
    located = locate_quotes(find_quotes_in_text(text), [chunk])
    segs = build_segments(text, located)
    plain_texts = [s.text for s in segs if isinstance(s, PlainText)]
    assert plain_texts[0] == "AAA\n\n"
    assert plain_texts[1] == "\n\nBBB"


def test_build_segments_formatted_text_from_format_quote():
    """QuoteSegment.formatted_text equals format_quote(lq).replacement."""
    chunk = _make_chunk("Book", "1", "Some rule text.")
    text = "> Some rule text.\n\n(Book, p. 99)"  # wrong page → will be fixed
    located = locate_quotes(find_quotes_in_text(text), [chunk])
    segs = build_segments(text, located)
    qs = [s for s in segs if isinstance(s, QuoteSegment)]
    assert len(qs) == 1
    assert qs[0].formatted_text == format_quote(qs[0].located).replacement


# ── are_segments_adjacent ─────────────────────────────────────────────────────

def _make_simple_quote_seg(text: str, quote_type: str = "blockquote", verified: bool = True) -> QuoteSegment:
    chunk: Chunk = {"content": text, "rulebook_name": "B", "page": "1", "start_index": 0, "end_index": len(text)}
    match = QuoteMatch(matched_text=text, source_chunk=chunk, referenced_chunks=[chunk], match_ratio=1.0) if verified else None
    lq = LocatedQuote(
        quote={"text": text, "quote": text, "quote_type": quote_type, "citation": None, "start_index": 0, "end_index": len(text)},
        match=match,
    )
    return QuoteSegment(original_text=text, located=lq, formatted_text=text)


def test_are_segments_adjacent_whitespace_only():
    seg_a = _make_simple_quote_seg("> Rule A.")
    seg_b = _make_simple_quote_seg("> Rule B.")
    segs = [seg_a, PlainText("\n\n"), seg_b]
    assert are_segments_adjacent(segs, seg_a, seg_b) is True


def test_are_segments_adjacent_content_between():
    seg_a = _make_simple_quote_seg("> Rule A.")
    seg_b = _make_simple_quote_seg("> Rule B.")
    segs = [seg_a, PlainText("Some content here."), seg_b]
    assert are_segments_adjacent(segs, seg_a, seg_b) is False


def test_are_segments_adjacent_quote_between():
    seg_a = _make_simple_quote_seg("> Rule A.")
    seg_mid = _make_simple_quote_seg("> Rule M.")
    seg_b = _make_simple_quote_seg("> Rule B.")
    segs = [seg_a, PlainText("\n\n"), seg_mid, PlainText("\n\n"), seg_b]
    assert are_segments_adjacent(segs, seg_a, seg_b) is False


def test_are_segments_adjacent_order_independent():
    seg_a = _make_simple_quote_seg("> Rule A.")
    seg_b = _make_simple_quote_seg("> Rule B.")
    segs = [seg_a, PlainText("\n\n"), seg_b]
    assert are_segments_adjacent(segs, seg_a, seg_b) == are_segments_adjacent(segs, seg_b, seg_a)


# ── remove_quote_segments ─────────────────────────────────────────────────────

def test_remove_quote_segments_basic():
    seg_a = _make_simple_quote_seg("> Rule A.")
    seg_b = _make_simple_quote_seg("> Rule B.")
    segs = [PlainText("x"), seg_a, PlainText("\n\n"), seg_b, PlainText("y")]
    result = remove_quote_segments(segs, {seg_a})
    assert seg_a not in result
    assert seg_b in result


def test_remove_quote_segments_merges_plain_text():
    seg = _make_simple_quote_seg("> Rule.")
    segs = [PlainText("AAA"), seg, PlainText("BBB")]
    result = remove_quote_segments(segs, {seg})
    assert len(result) == 1
    assert isinstance(result[0], PlainText)
    assert result[0].text == "AAABBB"


def test_remove_quote_segments_empty_set():
    seg = _make_simple_quote_seg("> Rule.")
    segs = [PlainText("x"), seg, PlainText("y")]
    result = remove_quote_segments(segs, set())
    assert result == segs


# ── dedupe_chunks ─────────────────────────────────────────────────────────────

def test_dedupe_chunks_overlapping():
    """Overlapping indexed chunks are merged with correct content and indices."""
    chunks: list[Chunk] = [
        {"rulebook_name": "Rules", "page": "1", "start_index": 0, "end_index": 20, "content": "AAAAAAAAAABBBBBBBBBB"},
        {"rulebook_name": "Rules", "page": "1", "start_index": 10, "end_index": 30, "content": "BBBBBBBBBBCCCCCCCCCC"},
    ]
    result = dedupe_chunks(chunks)
    assert len(result) == 1
    assert result[0]["start_index"] == 0
    assert result[0]["end_index"] == 30
    assert result[0]["content"] == "AAAAAAAAAABBBBBBBBBBCCCCCCCCCC"


def test_dedupe_chunks_adjacent():
    """Adjacent indexed chunks are merged."""
    chunks: list[Chunk] = [
        {"rulebook_name": "Rules", "page": "1", "start_index": 0, "end_index": 10, "content": "AAAAAAAAAA"},
        {"rulebook_name": "Rules", "page": "1", "start_index": 10, "end_index": 20, "content": "BBBBBBBBBB"},
    ]
    result = dedupe_chunks(chunks)
    assert len(result) == 1
    assert result[0]["start_index"] == 0
    assert result[0]["end_index"] == 20
    assert result[0]["content"] == "AAAAAAAAAABBBBBBBBBB"


def test_dedupe_chunks_fully_contained():
    """A chunk fully contained within another is absorbed."""
    chunks: list[Chunk] = [
        {"rulebook_name": "Rules", "page": "1", "start_index": 0, "end_index": 30, "content": "AAAAAAAAAABBBBBBBBBBCCCCCCCCCC"},
        {"rulebook_name": "Rules", "page": "1", "start_index": 5, "end_index": 15, "content": "AAAAABBBBB"},
    ]
    result = dedupe_chunks(chunks)
    assert len(result) == 1
    assert result[0]["start_index"] == 0
    assert result[0]["end_index"] == 30
    assert result[0]["content"] == "AAAAAAAAAABBBBBBBBBBCCCCCCCCCC"


def test_dedupe_chunks_no_overlap():
    """Non-overlapping indexed chunks are kept separate."""
    chunks: list[Chunk] = [
        {"rulebook_name": "Rules", "page": "1", "start_index": 0, "end_index": 10, "content": "AAAAAAAAAA"},
        {"rulebook_name": "Rules", "page": "1", "start_index": 20, "end_index": 30, "content": "CCCCCCCCCC"},
    ]
    result = dedupe_chunks(chunks)
    assert len(result) == 2
    assert result[0]["content"] == "AAAAAAAAAA"
    assert result[1]["content"] == "CCCCCCCCCC"


def test_dedupe_chunks_unindexed():
    """Unindexed chunks dedup by content as before."""
    chunks: list[Chunk] = [
        {"rulebook_name": "Rules", "page": "1", "start_index": -1, "end_index": -1, "content": "same text"},
        {"rulebook_name": "Rules", "page": "1", "start_index": -1, "end_index": -1, "content": "same text"},
        {"rulebook_name": "Rules", "page": "1", "start_index": -1, "end_index": -1, "content": "different text"},
    ]
    result = dedupe_chunks(chunks)
    assert len(result) == 2
    assert result[0]["content"] == "same text"
    assert result[1]["content"] == "different text"


def test_dedupe_chunks_mixed_indexed_and_unindexed():
    """Indexed and unindexed chunks in the same group are handled separately."""
    chunks: list[Chunk] = [
        {"rulebook_name": "Rules", "page": "1", "start_index": 0, "end_index": 10, "content": "AAAAAAAAAA"},
        {"rulebook_name": "Rules", "page": "1", "start_index": -1, "end_index": -1, "content": "unindexed chunk"},
        {"rulebook_name": "Rules", "page": "1", "start_index": 5, "end_index": 15, "content": "AAAAABBBBB"},
    ]
    result = dedupe_chunks(chunks)
    # Indexed chunks merge into one, unindexed kept separately
    assert len(result) == 2
    indexed = [c for c in result if c["start_index"] >= 0]
    unindexed = [c for c in result if c["start_index"] < 0]
    assert len(indexed) == 1
    assert indexed[0]["start_index"] == 0
    assert indexed[0]["end_index"] == 15
    assert indexed[0]["content"] == "AAAAAAAAAA" + "BBBBB"
    assert len(unindexed) == 1
    assert unindexed[0]["content"] == "unindexed chunk"


def test_dedupe_chunks_different_pages():
    """Chunks from different pages are never merged."""
    chunks: list[Chunk] = [
        {"rulebook_name": "Rules", "page": "1", "start_index": 0, "end_index": 10, "content": "page 1 text"},
        {"rulebook_name": "Rules", "page": "2", "start_index": 0, "end_index": 10, "content": "page 2 text"},
    ]
    result = dedupe_chunks(chunks)
    assert len(result) == 2


def test_dedupe_chunks_non_numeric_page():
    """Verify dedupe works with non-numeric page values like roman numerals."""
    chunks: list[Chunk] = [
        {"rulebook_name": "Rules", "page": "iii", "start_index": 0, "end_index": 20, "content": "AAAAAAAAAABBBBBBBBBB"},
        {"rulebook_name": "Rules", "page": "iii", "start_index": 10, "end_index": 30, "content": "BBBBBBBBBBCCCCCCCCCC"},
    ]
    result = dedupe_chunks(chunks)
    assert len(result) == 1
    assert result[0]["start_index"] == 0
    assert result[0]["end_index"] == 30
    assert result[0]["content"] == "AAAAAAAAAABBBBBBBBBBCCCCCCCCCC"


def test_fix_quote_citations_string_page():
    """Verify quote citation matching works when chunk page is a string."""
    chunks: list[Chunk] = [
        {
            "content": "Some important rule text for testing.",
            "rulebook_name": "Game Rules",
            "page": "42",
            "start_index": 0,
            "end_index": -1,
        },
    ]

    text = '> "Some important rule text for testing."\n\n(Game Rules, p. 42)'
    result = fix_quote_citations_in_text(text, chunks)
    assert len(result.unfixable_quotes) == 0
    assert len(result.valid_quotes) == 1


def test_sort_chunks_by_start_index():
    """Verify chunks sort by document position (start_index), not by page string."""
    manifest = {
        "game_id": "test",
        "game_version": "1.0",
        "rulebooks": [{"name": "Rules", "document_key": "rules"}],
    }
    chunks: list[Chunk] = [
        {"rulebook_name": "Rules", "page": "10", "start_index": 500, "end_index": 600, "content": "later in doc"},
        {"rulebook_name": "Rules", "page": "2", "start_index": 100, "end_index": 200, "content": "earlier in doc"},
    ]
    sorted_result = sort_chunks(chunks, manifest)
    # start_index=100 should come before start_index=500 regardless of page strings
    assert sorted_result[0]["start_index"] == 100
    assert sorted_result[1]["start_index"] == 500


@pytest.mark.asyncio
async def test_validate_and_fix_response_fix_via_llm():
    response = """
    Some summary of the response.

    > Some quoted text that is okay. (Rulebook, p. 5)

    > Wrong bit of some quoted text. (Rulebook, p. 6)

    > Another bit of not so right text. (Rulebook, p. 6)

    Some other text about stuff and such.

    > Another quote that is good. (Rulebook, p. 4)
    
    > A multi-line bit of text.
    >
    > Another paragraph. (Rulebook, p. 4)
    """

    response = inspect.cleandoc(response)

    page_4_content = """
    Another quote that is good. A multi-line bit of text.

    Another paragraph.
    """

    page_4_content = inspect.cleandoc(page_4_content)

    page_5_content = """
    Some amount of text here. There is some amount of it. Yes there is. Some quoted text that is okay.
    """

    page_5_content = inspect.cleandoc(page_5_content)

    page_6_content = """
    This quoted text is a bit wrong.  Another bit of maybe not right text.
    """

    page_6_content = inspect.cleandoc(page_6_content)

    chunks: list[Chunk] = [
        Chunk(
            rulebook_name="Rulebook",
            page="4",
            start_index=50,
            end_index=50 + len(page_4_content),
            content=page_4_content
        ),
        Chunk(
            rulebook_name="Rulebook",
            page="5",
            start_index=60,
            end_index=60+len(page_5_content),
            content=page_5_content
        ),
        Chunk(
            rulebook_name="Rulebook",
            page="6",
            start_index=0,
            end_index=0 + len(page_6_content),
            content=page_6_content
        )
    ]

    input = ValidateAndFixResponseInput(
        response=response,
        evidence=chunks,
        messages=[],
        validation_attempts=0
    )

    fix_quote_response_idx = 0
    fix_quote_responses = [
        FixQuotesResult(
            reasoning="Some fancy pants reasoning",
            fixable=True,
            fixed_quote=QuoteEntry(
                text="This quoted text is a bit wrong.",
                rulebook_name="Rulebook",
                page="6"
            )
        ),
        FixQuotesResult(
            reasoning="Some fancy pants reasoning",
            fixable=True,
            fixed_quote=QuoteEntry(
                text="This quoted text is a bit wrong.",
                rulebook_name="Rulebook",
                page="6"
            )
        )        
    ]

    @chain
    def fix_quote_stub(input: FixQuoteInput) -> FixQuotesResult:
        nonlocal fix_quote_response_idx
        return_value = fix_quote_responses[fix_quote_response_idx]
        fix_quote_response_idx += 1
        return return_value

    # Mock out meeplemate.qa_graph.build_fix_quote_chain to return our stub chain
    from unittest.mock import patch, MagicMock

    runtime = MagicMock()
    runtime.context.manifest = {
        "game_id": "test_game",
        "game_version": "1.0",
        "rulebooks": [
            {"name": "Rulebook", "document_key": "rulebook"}
        ]
    }

    with patch("meeplemate.qa_graph.build_fix_quote_chain", return_value=fix_quote_stub):
        result: ValidateAndFixResponseOutput = await validate_and_fix_response(input, runtime=runtime, config=None)

    quotes = find_quotes_in_text(result['response'])
    quote_entries: List[QuoteEntry] = [extracted_quote_to_quote_entry(q) for q in quotes]
    expected_quote_entries = [
        QuoteEntry(
            text="Some quoted text that is okay.",
            rulebook_name="Rulebook",
            page="5"
        ),
        QuoteEntry(
            text="This quoted text is a bit wrong.",
            rulebook_name="Rulebook",
            page="6"
        ),
        QuoteEntry(
            text="Another quote that is good.",
            rulebook_name="Rulebook",
            page="4"
        ),
        QuoteEntry(
            text="A multi-line bit of text.\n\nAnother paragraph.",
            rulebook_name="Rulebook",
            page="4"
        )
    ]

    assert result["invalid_quotes"] == [], "All quotes should be fixable in this test case, so no invalid quotes should remain."

    # Lets check the expected quote entries are correct ignoring the order they
    # appear in
    for expected_entry in expected_quote_entries:
        assert expected_entry in quote_entries, f"Expected quote entry not found in result: {expected_entry}"
    assert len(quote_entries) == len(expected_quote_entries), f"Expected {len(expected_quote_entries)} quote entries, but found {len(quote_entries)}. Quote entries found: {quote_entries}"


def test_quote_spanning_multiple_chunks():
    # Tests that a valid quote is recognised when its text spans two adjacent
    # chunks (by start_index).  Adjacent chunks can cross page boundaries, so
    # chunk 5 is on page 5 and chunk 6 is on page 6.  The "Battle Rounds"
    # quote in the response covers the tail of chunk 5 and the head of chunk 6.
    documents: list[Chunk] = [
        {'rulebook_name': 'Crystal Knights Rule Book', 'page': '5', 'start_index': 1000, 'end_index': 1120,  'content': '## ACTION PHASE\n\nThe Action Phase consists of two steps performed in the following order: 1. Draw Event Card 2. Move Enemies'},
        {'rulebook_name': 'Crystal Knights Rule Book', 'page': '5', 'start_index': 1122, 'end_index': 1480,  'content': '## Draw Event Card\n\nDraw the top card of the Event Deck and resolve its effects. Event Cards have numbered steps that are performed in order. If for any reason a step cannot be completed, skip it. If the card has a lasting effect, keep it visible to all players. Otherwise, place it face up in the Event Card discard pile.'},
        {'rulebook_name': 'Crystal Knights Rule Book', 'page': '5', 'start_index': 1900, 'end_index': 2180,  'content': '## Special Triggers\n\nSome Event Cards have Special Triggers printed in bold (such as Stun or Sweep). Each Special Trigger is described alphabetically in Appendix A.\n\n## Special Powers\n\nSome Event Cards have a Special Power listed in italics. When you see this, consult the Power Reference Sheet and follow the listed instructions.'},
        {'rulebook_name': 'Crystal Knights Rule Book', 'page': '5', 'start_index': 2182, 'end_index': 2440,  'content': 'After resolving an Event Card, reveal the top card of the Event Deck and place it face up. If the Event Deck is empty, reshuffle the discard pile to form a new Event Deck.'},
        {'rulebook_name': 'Crystal Knights Rule Book', 'page': '5', 'start_index': 2442, 'end_index': 2560,  'content': '## Battle Rounds\n\nA game is divided into 3 Battle Rounds, representing escalating enemy aggression as the battle progresses.'},
        {'rulebook_name': 'Crystal Knights Rule Book', 'page': '6', 'start_index': 2562, 'end_index': 3100,  'content': "Round transitions occur in two ways.\n\nFirst, a new Round begins automatically after 4 Event Cards have been drawn from the current Round.\n\nSecondly, the Scenario Sheet describes an alternate way to trigger a Round transition. If a Round changes and there are still Event Cards remaining from the old Round, discard those cards until the new Round's cards are on top of the deck. Reveal the top card and place it face up on the Event Deck."},
        {'rulebook_name': 'Crystal Knights Rule Book', 'page': '2', 'start_index':  200, 'end_index':  520,  'content': 'Setup: Take the Event Cards listed in the Scenario Sheet. Shuffle the Round 3 cards and place them face down. Shuffle the Round 2 cards and place them face down on top. Shuffle the Round 1 cards and place them face down on top of those. Turn the top Round 1 card face up.'},
    ]
    response = (
        'Yes, enemy actions are resolved through the Event Card system during the Action Phase.\n\n'
        'The Action Phase consists of two steps:\n\n'
        '> ## ACTION PHASE\n'
        '> The Action Phase consists of two steps performed in the following order: 1. Draw Event Card 2. Move Enemies\n'
        '>\n'
        '> (Crystal Knights Rule Book, p. 5)\n\n'
        "Enemy actions are defined by each card's numbered steps:\n\n"
        '> ## Draw Event Card\n'
        '> Draw the top card of the Event Deck and resolve its effects. Event Cards have numbered steps that are performed in order. If for any reason a step cannot be completed, skip it. If the card has a lasting effect, keep it visible to all players. Otherwise, place it face up in the Event Card discard pile.\n'
        '>\n'
        '> (Crystal Knights Rule Book, p. 5)\n\n'
        'The progression of enemy behaviour across rounds is governed by the Battle Rounds system:\n\n'
        '> ## Battle Rounds\n'
        '> \n'
        '> A game is divided into 3 Battle Rounds, representing escalating enemy aggression as the battle progresses.\n'
        '> \n'
        '> Round transitions occur in two ways.\n'
        '> \n'
        '> First, a new Round begins automatically after 4 Event Cards have been drawn from the current Round.\n'
        '> \n'
        "> Secondly, the Scenario Sheet describes an alternate way to trigger a Round transition. If a Round changes and there are still Event Cards remaining from the old Round, discard those cards until the new Round's cards are on top of the deck. Reveal the top card and place it face up on the Event Deck.\n"
        '> \n'
        '> (Crystal Knights Rule Book, p. 5)\n\n'
        'After resolving an Event Card, the next card is always revealed:\n\n'
        '> After resolving an Event Card, reveal the top card of the Event Deck and place it face up. If the Event Deck is empty, reshuffle the discard pile to form a new Event Deck.\n'
        '>\n'
        '> (Crystal Knights Rule Book, p. 5)\n\n'
        'In conclusion, enemy actions are resolved exclusively through the Event Card system.'
    )
    result = fix_quote_citations_in_text(response, documents)
    assert len(result.unfixable_quotes) == 0, f"Expected all quotes to be fixable, but found unfixable quotes: {result.unfixable_quotes}"


def test_quote_cross_chunk_ellipsis():
    """A quote bridging two non-adjacent chunks via '...' should be valid.

    Regression: the LLM combined text from two separate chunks on the same page
    using '...' (ellipsis).  fix_quote_citations_in_text builds a combined
    string per rulebook and rejects matches where the matched span is more than
    3× the quote length (to suppress false positives).  When many intervening
    chunks sit between the two relevant ones, the span easily exceeds that
    threshold and the quote is incorrectly flagged as unfixable.
    """
    # A large filler chunk between the two relevant chunks creates the
    # gap that causes the span to exceed len(quote) * 3.
    filler = "A" * 600
    chunks: list[Chunk] = [
        {
            "rulebook_name": "Game Rules",
            "page": "4",
            "start_index": 100,
            "end_index": 200,
            "content": "Players may use one-shot cards to help or harm others in combat. Some special cards may also be played into combat.",
        },
        {
            "rulebook_name": "Game Rules",
            "page": "4",
            "start_index": 250,
            "end_index": 850,
            "content": filler,
        },
        {
            "rulebook_name": "Game Rules",
            "page": "4",
            "start_index": 900,
            "end_index": 1000,
            "content": "You can play these either during your own combats or during someone else's combat.",
        },
    ]
    # The LLM bridges chunk 1 and chunk 3 with '...' — each half is real,
    # but the filler chunk in between inflates the combined-string span past
    # the 3× guard, so the match is rejected and the quote flagged unfixable.
    response = inspect.cleandoc("""\
        Special cards can be played during combat:

        > Players may use one-shot cards to help or harm others in combat. Some special cards may also be played into combat. ... You can play these either during your own combats or during someone else's combat.

        > (Game Rules, p. 4)

        So both halves apply.
    """)
    result = fix_quote_citations_in_text(response, chunks)
    assert len(result.unfixable_quotes) == 0, (
        f"Cross-chunk ellipsis quote should be valid, but was flagged as unfixable: {result.unfixable_quotes}"
    )


def test_unescape_table_html():
    example = inspect.cleandoc(
        """\
        > &lt;table&gt;
        > &lt;tr&gt;&lt;td&gt;PROFILE&lt;/td&gt;&lt;td&gt;M&lt;/td&gt;&lt;td&gt;WS&lt;/td&gt;&lt;td&gt;BS&lt;/td&gt;&lt;td&gt;S&lt;/td&gt;&lt;td&gt;T&lt;/td&gt;&lt;td&gt;W&lt;/td&gt;&lt;td&gt;I&lt;/td&gt;&lt;td&gt;A&lt;/td&gt;&lt;td&gt;LD&lt;/td&gt;&lt;/tr&gt;
        > &lt;tr&gt;&lt;td&gt;Grail Knight&lt;/td&gt;&lt;td&gt;4&lt;/td&gt;&lt;td&gt;5&lt;/td&gt;&lt;td&gt;3&lt;/td&gt;&lt;td&gt;4&lt;/td&gt;&lt;td&gt;3&lt;/td&gt;&lt;td&gt;1&lt;/td&gt;&lt;td&gt;4&lt;/td&gt;&lt;td&gt;1&lt;/td&gt;&lt;td&gt;9&lt;/td&gt;&lt;/tr&gt;
        > &lt;tr&gt;&lt;td&gt;Bretonnian Warhorse&lt;/td&gt;&lt;td&gt;8&lt;/td&gt;&lt;td&gt;3&lt;/td&gt;&lt;td&gt;0&lt;/td&gt;&lt;td&gt;3&lt;/td&gt;&lt;td&gt;3&lt;/td&gt;&lt;td&gt;1&lt;/td&gt;&lt;td&gt;3&lt;/td&gt;&lt;td&gt;1&lt;/td&gt;&lt;td&gt;5&lt;/td&gt;&lt;/tr&gt;
        > &lt;/table&gt;
        > 
        """
    )

    expected = inspect.cleandoc(
        """\
        > <table>
        > <tr><td>PROFILE</td><td>M</td><td>WS</td><td>BS</td><td>S</td><td>T</td><td>W</td><td>I</td><td>A</td><td>LD</td></tr>
        > <tr><td>Grail Knight</td><td>4</td><td>5</td><td>3</td><td>4</td><td>3</td><td>1</td><td>4</td><td>1</td><td>9</td></tr>
        > <tr><td>Bretonnian Warhorse</td><td>8</td><td>3</td><td>0</td><td>3</td><td>3</td><td>1</td><td>3</td><td>1</td><td>5</td></tr>
        > </table>
        > 
        """
    )

    result = unescape_table_html(example)
    assert result == expected


# ---------------------------------------------------------------------------
# fix_quote_citations_in_text – indented blockquotes
# ---------------------------------------------------------------------------


def test_fix_quote_citations_indented_blockquote_verified():
    """Indented blockquote (LLM list-item format) is detected, verified, and de-indented."""
    chunk: Chunk = {
        'content': 'The side that loses a combat must take a test to determine whether it stands and fights or turns tail and runs away. This is called a Break test.',
        'start_index': 0,
        'end_index': -1,
        'page': '41',
        'rulebook_name': 'Warhammer Rulebook',
    }

    # Mirrors the actual answer_question output where blockquotes are sub-items
    # of a numbered list — each blockquote line has 3 spaces of leading indentation.
    text = inspect.cleandoc("""\
        2. **Locate the definition of Break tests**:
           > The side that loses a combat must take a test to determine whether it stands and fights or turns tail and runs away. This is called a Break test.
           >
           > (Warhammer Rulebook, p. 41)
    """)

    result = fix_quote_citations_in_text(text, [chunk])

    assert len(result.unfixable_quotes) == 0, f"Expected no unfixable quotes, got: {result.unfixable_quotes}"
    assert len(result.referenced_chunks) == 1
    assert '   >' not in result.fixed_text, "Indented blockquote markers should be removed in fixed_text"
    assert '> The side that loses' in result.fixed_text


def test_fix_quote_citations_indented_blockquote_unverified():
    """Indented blockquote that cannot be verified is still de-indented in the output."""
    chunk: Chunk = {
        'content': 'Completely unrelated content that will not match.',
        'start_index': 0,
        'end_index': -1,
        'page': '1',
        'rulebook_name': 'Some Book',
    }

    text = "3. **Check**:\n   > Hallucinated rule that does not appear in any chunk.\n   >\n   > (Warhammer Rulebook, p. 99)"

    result = fix_quote_citations_in_text(text, [chunk])

    assert len(result.unfixable_quotes) == 1
    assert '   >' not in result.fixed_text, "Indented markers should be removed even for unverified quotes"
    assert '> Hallucinated rule' in result.fixed_text


# ---------------------------------------------------------------------------
# hint_match — chunk recovery from low-confidence matches
# ---------------------------------------------------------------------------


def test_fix_quote_citations_hint_chunks_from_paraphrase():
    """A close paraphrase that fails the high-confidence threshold (≥92) but
    passes the hint threshold (≥50) still recovers the relevant chunk via hint_match."""
    # This mirrors the real pattern seen in logs: "separate and distinct" vs "quite separate"
    chunk: Chunk = {
        'content': 'A Break test is not a psychology test. The two tests are quite separate.',
        'start_index': 0,
        'end_index': 71,
        'page': '46',
        'rulebook_name': 'Core Rulebook',
    }
    # Scores ~74% — fails ≥92 but passes ≥50
    text = (
        "> Break tests are not psychology tests. The two are separate and distinct.\n"
        ">\n"
        "> (Core Rulebook, p. 46)"
    )
    result = fix_quote_citations_in_text(text, [chunk])

    assert len(result.valid_quotes) == 0, "Paraphrase should not be verified"
    assert len(result.unfixable_quotes) == 1, "Paraphrase should be in unfixable_quotes"
    assert len(result.referenced_chunks) >= 1, "hint_match should recover the chunk"
    assert result.referenced_chunks[0]['rulebook_name'] == 'Core Rulebook'


# ---------------------------------------------------------------------------
# strip_invalid_blockquotes
# ---------------------------------------------------------------------------


def test_fix_quote_citations_strip_invalid_blockquotes():
    """With strip_invalid_blockquotes=True, unverified blockquotes have their
    '> ' markers stripped and appear as plain prose in the output."""
    chunk: Chunk = {
        'content': 'Completely unrelated content that will not match.',
        'start_index': 0,
        'end_index': -1,
        'page': '1',
        'rulebook_name': 'Some Book',
    }
    text = (
        "My analysis:\n\n"
        "> Yes, units must take the test even when immune to psychology.\n"
        ">\n"
        "> (Some Book, p. 1)"
    )
    result = fix_quote_citations_in_text(text, [chunk], strip_invalid_blockquotes=True)

    assert len(result.unfixable_quotes) == 1
    assert '>' not in result.fixed_text, "blockquote markers should be stripped"
    assert 'Yes, units must take the test' in result.fixed_text


def test_fix_quote_citations_no_strip_by_default():
    """Without strip_invalid_blockquotes (default False), unverified blockquotes
    retain their '> ' markers unchanged."""
    chunk: Chunk = {
        'content': 'Completely unrelated content that will not match.',
        'start_index': 0,
        'end_index': -1,
        'page': '1',
        'rulebook_name': 'Some Book',
    }
    text = (
        "My analysis:\n\n"
        "> Yes, units must take the test even when immune to psychology.\n"
        ">\n"
        "> (Some Book, p. 1)"
    )
    result = fix_quote_citations_in_text(text, [chunk])

    assert len(result.unfixable_quotes) == 1
    assert '>' in result.fixed_text, "blockquote markers should be preserved by default"
