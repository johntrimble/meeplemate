import inspect
from collections import Counter

from pytest_unordered import unordered

from meeplemate.quote_util import find_quote_with_gaps, find_quotes_in_text

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
