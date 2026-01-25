from langchain.messages import AnyMessage, ToolMessage
from langchain_core.messages.content import ToolCall
from meeplemate.qa_graph import Chunk, ChunkSearchResult, QaResponse, dedupe_chunks_in_message_history, get_chunk_id_tuple, tweak_and_validate_quotes_response
from langchain_core.messages import AIMessage, BaseMessage
from typing import List
import json

def test_dedupe_chunks_in_message_history():
    result1 = ChunkSearchResult(
        chunk=Chunk(
            rulebook_name="Rulebook 1",
            page=1,
            offset=3,
            content="This is chunk 1",   
        ),
        relevance_reason="Relevant to the query",
    )
    result2 = ChunkSearchResult(
        chunk=Chunk(
            rulebook_name="Rulebook 1",
            page=2,
            offset=5,
            content="This is chunk 2",   
        ),
        relevance_reason="Also relevant to the query",
    )
    result3 = ChunkSearchResult(
        chunk=Chunk(
            rulebook_name="Rulebook 2",
            page=1,
            offset=0,
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
            'offset': 3678,
            'page': 2,
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
            'offset': 1729,
            'page': 2,
            'rulebook_name': 'Munchkin Rules'
        },
    ]

    response:QaResponse = {
        'definitions': [
            {
                'term': 'Selling Items for Levels',
                'quotes': [
                    {
                        'text': 'Selling Items for Levels: At any point during your turn except during combat or Running Away, you may discard Items worth a total of at least 1,000 Gold Pieces and immediately go up one level. ("No Value" cards are the same as zero Gold Pieces.) If you discard (for instance) 1,100 Gold Pieces worth, you don\'t get change. But if you can manage 2,000 worth, you can go up two levels at once, and so on. You may sell Items from your hand as well as those you are carrying. You may not sell Items to go to Level 10.', 
                        'rulebook_name': 'Munchkin Rules', 
                        'page': 2
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
                        'page': 2
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
                        'page': 2
                    }
                ],
                'defines_term': True,
                'clarifying_question': ''
            }
        ], 
        'exceptions': [
            {
                'general_rule': 'Discarding Items',
                'quotes': [
                    {
                        'text': 'You cannot discard Item cards "just because." You may sell Items for a level, trade Items with other players, or give an Item to another player who wants it (see below). You may discard Items to power certain Class and Race abilities. And a Curse or a monster\'s Bad Stuff (see p. 5) may force you to get rid of something!', 
                        'rulebook_name': 'Munchkin Rules', 
                        'page': 2
                    }
                ], 
                'exception_names_general_rule': True, 
                'quotes_discounting_link': [], 
                'does_exception_apply': True, 
                'clarifying_question': ''
            }
        ], 
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