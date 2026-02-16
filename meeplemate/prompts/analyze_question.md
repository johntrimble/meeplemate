## Explain User's Query

Consider the user's query. Do NOT answer the user's query. Explain the key rules and rule interactions involved in the query. After providing this explanation, list 1-3 fully self-contained subquestions, the answers to which would collectively answer the user's query. Each subquestion should be focused on a specific aspect of the rules or rule interactions. Each subquestion should have an explanation as to how it relates to the user's original query. Each subquestion should involve fewer rules than the original query. Do NOT answer the subquestions. Only list the subquestions. Do NOT make assumptions about the rules. If you fail to break down the query into subquestions effectively, you will be punished.

## Classify User Query

Some user queries involve a simple rule lookup. For example:

- What is the hand limit?
- Can I fire my bow after moving?
- What happens when my character dies?
- etc.

In summary, these questions are about simple facts in the rulebook requiring little to no interpretation. These questions are classified as SIMPLE.

Other user queries involve multiple rules and rule interactions. For example:

- Can I use my action to attack with my sword and then use my bonus action to cast a spell?
- Does teleportation count as movement? Can I use movement bonuses for it?
- If my opponent successfully dispells my spell, does it still count as having been successfully cast?
- etc.

In summary, these rules require a degree of interpretation, logic, and reasoning in order to answer. These questions are classified as COMPLEX.

Output the classification of the user's query as either SIMPLE or COMPLEX, along with the explanation and subquestions as described above.

{{#retrieval}}
## Retrieval

You can lookup information for rules questions by using the `search_chunks` tool.
{{/retrieval}}

## Output Format (JSON)

{
    // Explanation is free form markdown text
    "explanation": "Explanation of what the user asking and the key rules and rule interactions involved in the user's query.",
    "subquestions": [
        {
            "subquestion": "subquestion 1",
            "explanation": "Explanation of how this subquestion relates to the user's original query"
        },
        {
            "subquestion": "subquestion 2",
            "explanation": "Explanation of how this subquestion relates to the user's original query"
        }
        // ... up to 3 subquestions
    ],
    "classification": "SIMPLE or COMPLEX"
}
