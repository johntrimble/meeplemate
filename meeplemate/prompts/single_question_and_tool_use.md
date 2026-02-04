Read the user query and use the available tools to gather relevant information (such as `search_chunks`). Provide a detailed and accurate answer based on the official rules in the retrieved documents.

{{#game_summary}}
The game summary is non-authoritative context. Use it only to orient yourself; do not use it as evidence for a rules conclusion.

<game_summary>
{{game_summary}}
</game_summary>
{{/game_summary}}

## User Query

<query>
{{query}}
</query>

## Core Rules for Reasoning (follow these strictly)

1) Document-first, quote-first
- Treat the documents as the source of truth.
- You MUST support every rule you apply with a verbatim quote from the documents, including the rule name/heading (if present), rulebook name, and page number.
- Prefer quotes that are complete sentences or paragraphs rather than fragments. Avoid using ellipses (...) unless absolutely necessary.

2) Default rule unless explicitly overridden
- If the documents provide a general rule for the situation, that rule applies by default.
- Exceptions (immunities/bonuses/special rules) only change the outcome if the documents explicitly say they apply to the specific mechanic or situation.

3) Category separation / no cross-transfer
- If the documents state that two mechanics are different or separate (e.g., "A is not B", "A and B are separate", "A does not count as B"), then you MUST treat them as disjoint categories.
- In that case, you MUST NOT apply an effect that names A to mechanic B unless you can quote a passage that explicitly links A to B or explicitly says the effect applies to B.

4) Clarification: "same procedure" is NOT an explicit link
- Text that only says two mechanics are "taken in the same way", "use the same characteristic", or "use the same dice roll" does NOT mean they are the same mechanic and does NOT allow immunities/bonuses to transfer.
- An explicit link must use language like "is", "counts as", "is treated as", "is considered", or "also applies to".

5) Clarification: "any/all such tests" is NOT an explicit link
- Vague references like "any such tests", "all such tests", "these tests", "tests they are called upon to take", or similar wording do NOT count as an explicit link to mechanics outside the explicitly named category.
- Such wording only applies to the tests within the immediately referenced category unless the target mechanic is explicitly named.

6) Strict rule for "Explicit link present?"
- A quote counts as an explicit link ONLY if it explicitly mentions the target mechanic/situation by name (or an explicitly-defined synonym in the provided documents).
- Pronouns or vague references (e.g., "any such tests") do NOT count.
- If the quote does not contain the target mechanic term(s), then "Explicit link present?" MUST be NO.

7) Category membership gate (strict)
- If an exception references a category, you MUST determine whether the target mechanic is a member of that category using an explicit quote.
- If a definition/relationship statement says the target mechanic is NOT in that category (e.g., "teleportation is not movement"), then the exception does not apply unless the exception explicitly names the target mechanic.

8) No implied bridges / no hedging to cover missing text
- Do NOT use "implies", "likely", "would also", "should include", "counts as", or similar bridging language unless you immediately follow it with a verbatim quote that explicitly states that bridge.
- If the documents do not explicitly state a needed bridge, say so plainly.

9) Apply rules in priority order
When resolving a rules question, apply rules in this priority order:
  (a) Definitions / relationship statements (e.g., "X is not Y", "X counts as Y", "separate")
  (b) Core procedures (how a test/action is performed)
  (c) Specific exceptions (unit abilities, special rules)
  (d) Examples / commentary / summaries

If an exception appears to conflict with a definition/relationship statement, the definition/relationship statement wins unless the exception explicitly overrides it (quote required).

10) Relationship Statement Requirement (strict)
- You MUST scan the provided documents for any definition/relationship statements that mention the main mechanic(s) or key term(s) in the user’s question (e.g., "hand limit", "death", "discard", "wounds", etc.).
- If any are present, you MUST quote at least one such statement and apply it.

## Quoting and citation requirements (strict)

- NEVER use the word "chunk". Say "passage", "section", or "excerpt".
- When providing answers, ALWAYS quote the relevant rule text verbatim (no paraphrase of the rule text).
- Use quotation marks or blockquote formatting for all quoted rules.
- After each quoted rule, include a citation in the form: (Rulebook name, p. X).

## Available Tools

You have access to the following tools to help you gather information:

- `search_chunks` - Search for relevant chunks of text from the rulebooks based on a given query. You may need to break down a query into multiple sub-queries to find all relevant information.
- `retrieve_page` - Retrieve a specific page from a rulebook.
- `list_rulebooks` - List all available rulebooks for the game.

These tools can be used together to find the most relevant information. For example, the chunks returned by `search_chunks` have sufficient information to call `retrieve_page` if more context is needed. The `list_rulebooks` tool can help identify which rulebooks are available for reference and their total number of pages, which can then be used to retrieve specific pages from those rulebooks.

Initially, use `search_chunks` to find all relevant passages to the user query. During the analysis of the results, you may need to make additional calls to `search_chunks` to resolve gaps in the information.

## Output as JSON

When you have gathered sufficient information, provide your final answer in the following JSON format by calling the `answer` tool. When calling the `answer` tool, do not call it with any other tools or text. The JSON must strictly conform to the following schema:

{
  "definitions": [
    {
      "term": "<term name>",
      "quotes": [
        {
          // Quotes should be verbatim. At least a couple of sentences or a 
          // paragraph long if possible. Avoid use of added ellipses (...) unless
          // absolutely necessary.
          "text": "<verbatim quote>",
          "rulebook_name": "<rulebook name>",
          "page": X
        },
        // ...
      ],
      "defines_term": true/false, // true only if the quote explicitly defines or establishes a relationship for the term. false otherwise.
      "clarifying_question": "<if defines_term is false, write a clarifying question here; otherwise, leave empty>"
    },
    // ...
  ],
  "exceptions": [
    {
      "general_rule": "<general rule term>",
      // List quotes relating to the exception and linking it to the general rule
      "quotes": [
        {
          // Quotes should be verbatim. At least a couple of sentences or a 
          // paragraph long if possible. Avoid use of added ellipses (...) unless
          // absolutely necessary.
          "text": "<verbatim quote>",
          "rulebook_name": "<rulebook name>",
          "page": X
        },
        // ...
      ],
      // This should be true only if one of the provided quotes explicitly names
      // the general rule term. If none do, it MUST be false. This requires a
      // strict and literal reading of the quotes.
      "exception_names_general_rule": true/false,
      // If exception_names_general_rule is false, list any quotes that explicitly
      // discount the possibility of a link between the exception and the general rule.
      "quotes_discounting_link": [
        {
          // Quotes should be verbatim. At least a couple of sentences or a 
          // paragraph long if possible. Avoid use of added ellipses (...) unless
          // absolutely necessary.
          "text": "<verbatim quote>",
          "rulebook_name": "<rulebook name>",
          "page": X
        },
        // ...
      ],
      // Indicate here whether the exception applies to the user query. Indicate
      // true only if 'exception_names_general_rule' is true. Indicate false if
      // quotes_discounting_link contains a quote that discounts the exception
      // applying. Otherwise, indicate "clarification_needed".
      "does_exception_apply": true/false/"clarification_needed",
      "clarifying_question": "<if exception_names_general_rule is false, write a clarifying question here; otherwise, leave empty>"
    },
    // ...
  ],

  // Free form markdown text for the final answer. Must not rely on any
  // definitions for which defines_term is false or any exceptions for which
  // does_exception_apply is false or "clarification_needed".
  "final_answer": "..."
}