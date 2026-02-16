## Invalid Quote

The following is an invalid quote that was extracted from the above documents. Your job is to find a passage in the documents that matches the meaning of this quote best and to then output that passage as the fixed quote.


<invalid_quote id="{{invalid_quote.id}}" rulebook_name="{{invalid_quote.rulebook_name}}" page="{{invalid_quote.page}}">
{{invalid_quote.text}}
</invalid_quote>

## Instructions

For each quote:

1. Scan the documents for relevant passages.
2. Determine if any passage has a roughly similar meaning to the invalid quote, even if the wording is different.
3. If a passage is found, provide the fixed verbatim quote from the document.
4. If no passage is found that matches the meaning, then mark the quote as "unfixable" and provide an explanation of why it is unfixable.

Rules:

- No paraphrasing is allowed. You must use the exact wording from the document if you find a match.
- If multiple passages match the meaning, choose the one that is most complete and provides the most context, rather than a fragment. Use ellipses (...) only if absolutely necessary.
- A quote is fixable if a suitable VERBATIM passage can be found with roughly the same meaning. A quote is unfixable if no passage can be found that matches the meaning, even if the wording is different. In other words, a quote is unfixable if the information it conveys cannot be found in any form in the documents.
- Fixed quotes must be take VERBATIM from the documents.


## Output Format

Your output should be a JSON object with the following structure:

```json
{
    "reasoning": "A brief explanation of the reasoning process for fixing the quote or why it was deemed unfixable.",
    "fixable": true/false, // true if the quote was successfully fixed, false if it was deemed unfixable
    // If fixable is true, include the fixed quote information below. If fixable is false, this field can be omitted or set to null.
    "fixed_quote": {
        "text": "Corrected quote taken _verbatim_ from the documents",
        "rulebook_name": "Name of the rulebook the quote was taken from",
        "page": "Page number the quote was taken from",
    }
}
```
