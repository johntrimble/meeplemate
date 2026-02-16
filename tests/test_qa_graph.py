from langchain.messages import AnyMessage, ToolMessage
from langchain_core.messages.content import ToolCall
from langchain_core.runnables import chain
import pytest
from meeplemate.qa_graph import Chunk, ChunkSearchResult, FixQuoteInput, FixQuotesResult, QaResponse, QuoteEntry, ValidateAndFixResponseOutput, dedupe_chunks, dedupe_chunks_in_message_history, extracted_quote_to_quote_entry, fix_quote_citations_in_text, get_chunk_id_tuple, sort_chunks, tweak_and_validate_quotes_response, ValidateAndFixResponseInput, validate_and_fix_response, validate_and_fix_response
from langchain_core.messages import AIMessage, BaseMessage
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


def test_validation():
    chunks: List[Chunk] = [
        {
            'content': 'You may also give Items away without a trade, to bribe other '
                'players - "I\'ll give you my Flaming Armor if you won\'t help Bob '
                'fight that dragon!" You may show your hand to others. Like we '
                'could stop you.\n'
                '\n'
                'Selling Items for Levels: At any point during your turn except '
                'during combat or Running Away, you may discard Items worth a '
                'total of at least 1,000 Gold Pieces and immediately go up one '
                'level. ("No Value" cards are the same as zero Gold Pieces.) If '
                "you discard (for instance) 1,100 Gold Pieces worth, you don't get "
                'change. But if you can manage 2,000 worth, you can go up two '
                'levels at once, and so on. You may sell Items from your hand as '
                'well as those you are carrying. You may not sell Items to go to '
                'Level 10.\n'
                '\n'
                '## "ONE-SHOT" TREASURES\n'
                '\n'
                'A Treasure card that says "Usable once only" is often called a '
                '"one- shot" Treasure. Most of these are used during combat to '
                'strengthen the munchkins or the monsters, and may be played from '
                'your hand or from the table. Some have other effects, however, so '
                'read the card carefully! Discard these cards as soon as the '
                'combat is over or their effect is resolved.\n'
                '\n'
                'One- shot Items with a Gold Piece value may be sold for levels, '
                'just like other Items.\n'
                '\n'
                '## OTHER TREASURES\n'
                '\n'
                'Other Treasure cards (like Go Up a Level cards) are not Items. '
                'Most of these cards say when they can be played, and whether they '
                'stay in play or are discarded. A couple of specific examples: Go '
                'Up a Level cards may be played on yourself or any other player at '
                'any time, even during combat. Discard them once they are played. '
                'Exception: You cannot play a Go Up a Level card to give a player '
                'the winning level!\n'
                '\n'
                'Hireling may be played at any time, on any turn. You cannot give '
                'a Hireling an Item to carry while you are in combat, however.\n'
                '\n'
                '## COMBAT',
            'start_index': 3678,
            'end_index': -1,
            'page': "2",
            'rulebook_name': 'Munchkin Rules'
        },
        {
            'content': 'Likewise, some Items have restrictions: for instance, the Mace of '
                'Sharpness can only be wielded by a Cleric. Its bonus only counts '
                'for someone who is, at the moment, a Cleric.\n'
                '\n'
                'You cannot discard Item cards "just because." You may sell Items '
                'for a level, trade Items with other players, or give an Item to '
                'another player who wants it (see below). You may discard Items to '
                'power certain Class and Race abilities. And a Curse or a '
                "monster's Bad Stuff (see p. 5) may force you to get rid of "
                'something!\n'
                '\n'
                'Big Items: You may carry any number of Small items, but only one '
                'Big one. (Any item not marked Big is considered Small.) You may '
                'not discard one Big item to play another; you must sell the first '
                'Item, trade it, lose it to a Curse or Bad Stuff, or discard it to '
                'power a Class or Race ability.\n'
                '\n'
                'If something lets you have more than one Big item (for instance, '
                'the Dwarf race) and you lose that ability, you must either '
                'correct the problem immediately or get rid of all but one Big '
                "item. If it's your turn and you're not in combat, you can sell "
                'the excess Big items (as long as you have at least\n'
                '\n'
                "## Level Counters: It's Not Cheating, It's Using the Rules!\n"
                '\n'
                "If you have an iOS or Android device, you'll like our Level "
                'Counter app. Just search for "Munchkin level counter" or click '
                'the link at levelcounter.sjgames.com. Even better, it gives you '
                'personal in- game advantages to make your friends jealous ... '
                'which is what being a munchkin is all about!\n'
                '\n'
                '1,000 Gold Pieces of Items to sell). Otherwise, you must give '
                'them to the lowest- Level player(s) who can carry them! If any '
                'Big items are still left over, discard them.\n'
                '\n'
                'Trading: You may trade Items (but no other cards) with other '
                'players. You may only trade Items from the table - not from your '
                'hand. You may trade at any time except when you or your trading '
                'partner are in combat - in fact, the best time to trade is when '
                "it's not your turn. Any Item you receive in a trade must remain "
                'in play.',
            'start_index': 1729,
            'end_index': -1,
            'page': "2",
            'rulebook_name': 'Munchkin Rules'
        },
    ]

    response:QaResponse = {
        'identified_mechanics': {
            'primary_mechanics': ['Selling Items for Levels'],
            'secondary_mechanics': ['One-Shot Treasures'],
            'reasoning': 'User is asking about selling items from hand for levels'
        },
        'relationship_statements': [],
        'definitions': [
            {
                'term': 'Selling Items for Levels',
                'quotes': [
                    {
                        'text': 'Selling Items for Levels: At any point during your turn except during combat or Running Away, you may discard Items worth a total of at least 1,000 Gold Pieces and immediately go up one level. ("No Value" cards are the same as zero Gold Pieces.) If you discard (for instance) 1,100 Gold Pieces worth, you don\'t get change. But if you can manage 2,000 worth, you can go up two levels at once, and so on. You may sell Items from your hand as well as those you are carrying. You may not sell Items to go to Level 10.',
                        'rulebook_name': 'Munchkin Rules',
                        'page': "2"
                    }
                ],
                'defines_term': True,
                'clarifying_question': ''
            },
            {
                'term': 'Items',
                'quotes': [
                    {
                        'text': 'You cannot discard Item cards "just because." You may sell Items for a level, trade Items with other players, or give an Item to another player who wants it (see below). You may discard Items to power certain Class and Race abilities. And a Curse or a monster\'s Bad Stuff (see p. 5) may force you to get rid of something!',
                        'rulebook_name': 'Munchkin Rules',
                        'page': "2"
                    }
                ],
                'defines_term': True,
                'clarifying_question': ''
            },
            {
                'term': 'One-Shot Treasures',
                'quotes': [
                    {
                        'text': 'One-shot Items with a Gold Piece value may be sold for levels, just like other Items.',
                        'rulebook_name': 'Munchkin Rules',
                        'page': "2"
                    }
                ],
                'defines_term': True,
                'clarifying_question': ''
            }
        ],
        'general_rules': [
            {
                'mechanic': 'Selling Items for Levels',
                'quotes': [
                    {
                        'text': 'Selling Items for Levels: At any point during your turn except during combat or Running Away, you may discard Items worth a total of at least 1,000 Gold Pieces and immediately go up one level.',
                        'rulebook_name': 'Munchkin Rules',
                        'page': "2"
                    }
                ],
                'summary': 'Items worth 1000+ GP can be sold for levels'
            }
        ],
        'exceptions': [
            {
                'exception_source': 'Discarding Items rule',
                'exception_scope_language': 'You cannot discard Item cards "just because."',
                'target_mechanic': 'Selling Items for Levels',
                'step1_scope_analysis': 'This restricts when items can be discarded',
                'step2_explicit_naming': {
                    'does_exception_name_target': True,
                    'explanation': 'The rule explicitly mentions selling items for a level as an allowed exception'
                },
                'step3_relationship_check': {
                    'relationship_exists': True,
                    'quotes': [
                        {
                            'text': 'You cannot discard Item cards "just because." You may sell Items for a level, trade Items with other players, or give an Item to another player who wants it (see below). You may discard Items to power certain Class and Race abilities. And a Curse or a monster\'s Bad Stuff (see p. 5) may force you to get rid of something!',
                            'rulebook_name': 'Munchkin Rules',
                            'page': "2"
                        }
                    ],
                    'explanation': 'Selling items for levels is explicitly listed as an allowed way to discard items'
                },
                'step4_separation_check': {
                    'separation_exists': False,
                    'quotes': [],
                    'explanation': 'No separation between selling for levels and allowed discarding'
                },
                'does_exception_apply': True,
                'precedence_level': 'Level 4',
                'clarifying_question': ''
            }
        ],
        'precedence_analysis': '',
        'reasoning': '- The user is asking whether they can sell items from their hand to gain a level, assuming they can sell 1,000 gold pieces worth of items.\n- According to the rule titled "Selling Items for Levels", you may sell Items from your hand as long as they are worth at least 1,000 Gold Pieces.\n- Additionally, "One-shot Items with a Gold Piece value may be sold for levels, just like other Items," which confirms that even special types of Items can be used in this process.\n- The rule explicitly states that selling Items is allowed during your turn (except during combat or Running Away), and that you can go up multiple levels if the total value of discarded Items is sufficient.\n- Therefore, based on the provided rules, the user can indeed sell items from their hand to go up a level if the total value is at least 1,000 Gold Pieces.',
        'final_answer': 'Yes, you can sell items from your hand to go up a level if the total value of the items is at least 1,000 Gold Pieces. This is explicitly stated in the Munchkin Rules:\n\n> "Selling Items for Levels: At any point during your turn except during combat or Running Away, you may discard Items worth a total of at least 1,000 Gold Pieces and immediately go up one level. (\'No Value\' cards are the same as zero Gold Pieces.) ... You may sell Items from your hand as well as those you are carrying."\n\nFurthermore, the rule confirms that even one-shot items with a Gold Piece value can be used for leveling:\n\n> "One-shot Items with a Gold Piece value may be sold for levels, just like other Items."\n\n(Munchkin Rules, p. 2)\n\nTherefore, if you have items in your hand totaling at least 1,000 Gold Pieces, you can use them to increase your level.',
        'sufficient_information_to_answer': True
    }

    result = tweak_and_validate_quotes_response(response, chunks)
    for quote in result.invalid_quotes:
        from pprint import pprint
        pprint(quote)
    assert len(result.invalid_quotes) == 0
    print()
    print(result.revised_response["final_answer"])
    print()
    for quote in result.valid_quotes:
        print(quote["rulebook_name"], quote["page"])
        print(quote["text"])
        print("-----")


def test_validation_citation_on_separate_line():
    """Test that citation on separate line is moved to blockquote"""
    chunks: List[Chunk] = [
        {
            'content': 'One- shot Items with a Gold Piece value may be sold for levels, just like other Items.',
            'start_index': 0,
            'end_index': -1,
            'page': "5",
            'rulebook_name': 'Game Rules'
        }
    ]

    response: QaResponse = {
        'identified_mechanics': {
            'primary_mechanics': [],
            'secondary_mechanics': [],
            'reasoning': ''
        },
        'relationship_statements': [],
        'definitions': [],
        'general_rules': [],
        'exceptions': [],
        'precedence_analysis': '',
        'reasoning': '',
        'final_answer': '> "One-shot Items with a Gold Piece value may be sold for levels, just like other Items."\n\n(Game Rules, p. 5)',
        'sufficient_information_to_answer': True
    }

    result = tweak_and_validate_quotes_response(response, chunks)

    # Should have no invalid quotes
    assert len(result.invalid_quotes) == 0, f"Expected 0 invalid quotes, got {len(result.invalid_quotes)}"

    # Citation should be moved to end of blockquote line
    expected = '> "One-shot Items with a Gold Piece value may be sold for levels, just like other Items." (Game Rules, p. 5)'
    assert result.revised_response['final_answer'] == expected, f"Expected:\n{expected}\n\nGot:\n{result.revised_response['final_answer']}"


def test_validation_multiline_blockquote_separate_citation():
    """Test multi-line blockquote with citation on separate line"""
    chunks: List[Chunk] = [
        {
            'content': 'First line of the rule continues here with more text and even more content on multiple lines.',
            'start_index': 0,
            'end_index': -1,
            'page': "3",
            'rulebook_name': 'Test Book'
        }
    ]

    response: QaResponse = {
        'identified_mechanics': {
            'primary_mechanics': [],
            'secondary_mechanics': [],
            'reasoning': ''
        },
        'relationship_statements': [],
        'definitions': [],
        'general_rules': [],
        'exceptions': [],
        'precedence_analysis': '',
        'reasoning': '',
        'final_answer': '> "First line of the rule\n> continues here with more text."\n\n(Test Book, p. 3)',
        'sufficient_information_to_answer': True
    }

    result = tweak_and_validate_quotes_response(response, chunks)

    # Should have no invalid quotes
    assert len(result.invalid_quotes) == 0

    # Citation should be on last blockquote line
    assert '> continues here with more text." (Test Book, p. 3)' in result.revised_response['final_answer']
    # Should not have citation on separate line
    assert not result.revised_response['final_answer'].endswith('\n\n(Test Book, p. 3)')


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

    response: QaResponse = {
        'identified_mechanics': {
            'primary_mechanics': [],
            'secondary_mechanics': [],
            'reasoning': ''
        },
        'relationship_statements': [],
        'definitions': [],
        'general_rules': [],
        'exceptions': [],
        'precedence_analysis': '',
        'reasoning': '',
        'final_answer': '> "Items can be sold for levels." (Rules, p. 1)',
        'sufficient_information_to_answer': True
    }

    result = tweak_and_validate_quotes_response(response, chunks)

    # Should have no invalid quotes
    assert len(result.invalid_quotes) == 0

    # Should remain unchanged
    assert result.revised_response['final_answer'] == '> "Items can be sold for levels." (Rules, p. 1)'


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

    response: QaResponse = {
        'identified_mechanics': {
            'primary_mechanics': [],
            'secondary_mechanics': [],
            'reasoning': ''
        },
        'relationship_statements': [],
        'definitions': [],
        'general_rules': [],
        'exceptions': [],
        'precedence_analysis': '',
        'reasoning': '',
        'final_answer': '> "Blockquote text here."\n\n(Book A, p. 1)\n\nAdditionally, "Inline quote text here." (Book B, p. 2)',
        'sufficient_information_to_answer': True
    }

    result = tweak_and_validate_quotes_response(response, chunks)

    # Should have no invalid quotes
    assert len(result.invalid_quotes) == 0

    # Blockquote should have inline citation
    assert '> "Blockquote text here." (Book A, p. 1)' in result.revised_response['final_answer']
    # Inline quote should remain unchanged
    assert '"Inline quote text here." (Book B, p. 2)' in result.revised_response['final_answer']


def test_validation_removes_standalone_citation():
    """Test that citations not associated with any quote are removed"""
    chunks: List[Chunk] = [
        {
            'content': 'Some rule text.',
            'start_index': 0,
            'end_index': -1,
            'page': "1",
            'rulebook_name': 'Book'
        }
    ]

    response: QaResponse = {
        'identified_mechanics': {
            'primary_mechanics': [],
            'secondary_mechanics': [],
            'reasoning': ''
        },
        'relationship_statements': [],
        'definitions': [],
        'general_rules': [],
        'exceptions': [],
        'precedence_analysis': '',
        'reasoning': '',
        'final_answer': 'Some text here.\n\n(Book, p. 99)\n\nMore text.',
        'sufficient_information_to_answer': True
    }

    result = tweak_and_validate_quotes_response(response, chunks)

    # The standalone citation should be removed since it's not associated with any quote
    assert '(Book, p. 99)' not in result.revised_response['final_answer']
    assert 'Some text here.' in result.revised_response['final_answer']
    assert 'More text.' in result.revised_response['final_answer']


def test_validation_multiple_blockquotes_separate_citations():
    """Test multiple blockquotes with their own citations"""
    chunks: List[Chunk] = [
        {
            'content': 'First rule text.',
            'start_index': 0,
            'end_index': -1,
            'page': "1",
            'rulebook_name': 'Book'
        },
        {
            'content': 'Second rule text.',
            'start_index': 100,
            'end_index': -1,
            'page': "2",
            'rulebook_name': 'Book'
        }
    ]

    response: QaResponse = {
        'identified_mechanics': {
            'primary_mechanics': [],
            'secondary_mechanics': [],
            'reasoning': ''
        },
        'relationship_statements': [],
        'definitions': [],
        'general_rules': [],
        'exceptions': [],
        'precedence_analysis': '',
        'reasoning': '',
        'final_answer': '> "First rule text."\n\n(Book, p. 1)\n\nAlso:\n\n> "Second rule text."\n\n(Book, p. 2)',
        'sufficient_information_to_answer': True
    }

    result = tweak_and_validate_quotes_response(response, chunks)

    # Should have no invalid quotes
    assert len(result.invalid_quotes) == 0

    # Both blockquotes should have inline citations
    assert '> "First rule text." (Book, p. 1)' in result.revised_response['final_answer']
    assert '> "Second rule text." (Book, p. 2)' in result.revised_response['final_answer']

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
        
        > "## WARPSTORM SCROLL
        > Bearer can cast spell in his magic phase. All creatures flying high's suffer D6 56 hits, and are forced down to earth, re- entering the table on their own side's table edge in their following turn." (Warhammer Magic, p. 44)
        
        The rule explicitly states that the scroll affects "creatures flying high," which aligns with the definition of flying high:
        
        > "## FLYING HIGH
        > A flyer may choose to fly high during his turn instead of making a normal flying move. This represents a flyer ascending far into the air above the battlefield..." (Warhammer Rulebook, p. 74)
        
        Thus, the effect of the Warpstorm Scroll applies to models in the flying high status.
        '''
    )

    response: QaResponse = {
        'identified_mechanics': {
            'primary_mechanics': [],
            'secondary_mechanics': [],
            'reasoning': ''
        },
        'relationship_statements': [],
        'definitions': [],
        'general_rules': [],
        'exceptions': [],
        'precedence_analysis': '',
        'reasoning': '',
        'final_answer': before,
        'sufficient_information_to_answer': True
    }

    result = tweak_and_validate_quotes_response(response, chunks)

    assert result.revised_response['final_answer'] == expected

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

        > "Some rulebook text. It goes on and on. For some sentences." (Some Rulebook, p. 44)

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

    result = fix_quote_citations_in_text(before, chunks)
    assert result.fixed_text == expected


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

        > The side that loses a combat must take a test to determine whether it stands and fights or turns tail and runs away This is called a Break test. (Warhammer Rulebook, p. 42)

        This means that regardless of the enemy type or unit abilities, any unit that loses a combat must perform a Break test.

        However, the Grail Knights have a special immunity that protects them from psychological effects:

        > The unit never needs test for any of the psychology rules, whether panic, fear, terror or whatever. The Knights are unaffected by any psychology. (Bretonnia Army Book, p. 49)

        This protection applies specifically to psychology-related tests, including those caused by Fear, Panic, or similar effects.

        But the Break test is explicitly not a psychology test:

        > However, a Break test is not a psychology test. The two tests are quite separate. This is important because some bonuses apply specifically to Break tests and others apply specifically to psychology tests. (Warhammer Rulebook, p. 47)

        Since the Break test is functionally distinct from psychology tests, and the Grail Knights’ immunity only applies to psychology rules, their protection does not extend to Break tests.

        Therefore, even though the Grail Knights are immune to the Green Dragon’s corrosive fumes (which trigger a Leadership test “in the same way as for a fear or other psychology test”), this does not affect the outcome of the Break test after combat loss.

        In conclusion: Losing combat triggers a Break test, and since the Break test is not a psychology test, the Grail Knights’ immunity does not prevent them from taking it.
        '''
    )

    result = fix_quote_citations_in_text(before, chunks)
    assert result.fixed_text == expected

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

    fixed = '**Yes, Grail Knights must take a Break test when they lose combat, despite their immunity to psychological effects.**\n\nThe general rule for losing combat requires a Break test:\n\n> The side that loses a combat must take a test to determine whether it stands and fights or turns tail and runs away. This is called a Break test. You need to take a separate Break test for every unit involved in the combat. (Warhammer Rulebook, p. 42)\n\nThis means that any unit which loses a combat must attempt a Break test, regardless of other traits.\n\nHowever, Grail Knights possess the Grail Virtue, which grants immunity to psychological effects:\n\n> Grail Knights have the most noble chivalric virtue of all – the Grail Virtue. This means that they are unaffected by any of the psychology rules; any such tests they are called upon to take are disregarded with a cool and steely countenance. The Knight knows neither fear nor terror, nor will he panic, for the grail sustains his noble will better than any magic trickery. (Bretonnia Army Book, p. 44)\n\nThe key distinction lies in the categorization of Break tests:\n\n> However, a Break test is not a psychology test. The two tests are quite separate. This is important because some bonuses apply specifically to Break tests and others apply specifically to psychology tests. (Warhammer Rulebook, p. 47)\n\nSince Break tests are explicitly stated to be *not* psychology tests, and the Grail Virtue only applies to "psychology rules" and "such tests" — which refer exclusively to Panic, Fear, Terror, and Stupidity — the immunity does not extend to Break tests.\n\nTherefore, even though Grail Knights are immune to psychological effects, they are still required to take a Break test when they lose combat, as the rule for Break tests is not overridden by the Grail Virtue.'

    result = fix_quote_citations_in_text(before, chunks)
    print(result.fixed_text)
    assert result.fixed_text == fixed


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

        > Some quote text here matching chunk content. (Some Rulebook, p. 42)

        This is the text after the quote.
        '''
    )

    result = fix_quote_citations_in_text(before, chunks)
    assert result.fixed_text == expected


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
