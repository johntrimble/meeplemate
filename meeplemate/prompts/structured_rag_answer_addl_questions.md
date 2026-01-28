Read the user query and the documents below carefully. Provide a detailed and accurate answer based on the official rules in the documents.

## User Query

<query>
{{query}}
</query>

## Documents

<documents>
{{#documents}}
<document rulebook_name="{{rulebook_name}}" page="{{page}}" offset="{{offset}}">
{{content}}
</document>
{{/documents}}
</documents>

## Clarifying Questions and Aswers

These are additional clarifying questions and answers to help resolve ambiguities in the user query and the rules. Use them to better understand how the rules interact with eachother and to resolve uncertainties about definitions and exceptions.

IMPORTANT: These questions and answers are only to be used to clarify ambiguities in the user query and the rules. They may NOT be cited as rule text or used as a substitute for actual rule quotes from the documents. They may NOT be used in the final answer text.

<clarifying_questions_and_answers>
{{#clarifying_questions_and_answers}}
<clarifying_question_and_answer>
<question>
{{question}}
</question>
<answer>
{{answer}}
</answer>
</clarifying_question_and_answer>
{{/clarifying_questions_and_answers}}
</clarifying_questions_and_answers>

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

9) Rule precedence
Apply rules in the following precedence order when conflicts arise or exceptions are present:
  (a) Definitions / relationship statements (e.g., "X is not Y", "X counts as Y", "separate")
  (b) Rules on cards/units/abilities/special scenarios
  (c) Specific exceptions (e.g., immunities, bonuses)
  (d) Core procedures (how a test/action is performed)
  (e) Examples / commentary / summaries

For example, suppose the core procedure for movement says "All units can move up to 2 hexes." But a character has an item equipped that says "This character can move up to 4 hexes." In this case, the character's item would take precedence over the core procedure, and the character would be able to move up to 4 hexes.

Note that the rulebooks ultimately govern the precedence of rules. They may set limits on how exceptions can override core procedures or definitions. For example, the rulebook might specify that a character can never have a movement greater than 6 hexes, regardless of how many movement-boosting items they have equipped. In such cases, the rulebook's limits would take precedence over the exceptions.

10) Relationship Statement Requirement (strict)
- You MUST scan the provided documents for any definition/relationship statements that mention the main mechanic(s) or key term(s) in the user’s question (e.g., "Break test", "hand limit", "death", "discard", "wounds", etc.).
- If any are present, you MUST quote at least one such statement and apply it.

## Quoting and citation requirements (strict)

- NEVER use the word "chunk". Say "passage", "section", or "excerpt".
- When providing answers, ALWAYS quote the relevant rule text verbatim (no paraphrase of the rule text).
- Use blockquote formatting for all quoted rules.
- After each quoted rule, include a citation in the form: (Rulebook name, p. X).
- Quotes should be at least a couple of sentences or a paragraph long if possible. Avoid use of added ellipses (...) unless absolutely necessary.

## Output as JSON

{
  "definitions": [
    {
      "term": "<term name>",
      // List quotes relating to the definition of the term. Use an empty list if
      // no relevant quotes are found and populate clarifying_question below.
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
      // Clarifying questions here are used when defines_term is false. They should be fully self-contained questions
      // that seek to define the term in question. They should not repeat the original user query or depend on context outside the question itself.
      "clarifying_question": "<if defines_term is false, write a clarifying question here; otherwise, leave empty>"
    },
    // ...
  ],
  "exceptions": [
    {
      "reasoning_about_exception": "<step-by-step reasoning about how the exception might apply>",
      "general_rule": "<general rule term>",
      "exception_rule": "<exception rule term>",
      // List quotes relating to the exception rule and linking it to the
      // general rule. Use an empty list if no relevant quotes are found and 
      // populate clarifying_question below.
      "quotes": [
        {
          // Quotes should be verbatim. At least a couple of sentences or a 
          // paragraph long if possible. Avoid use of added ellipses (...) unless
          // absolutely necessary.
          "text": "<verbatim quote>",
          "rulebook_name": "<rulebook name>",
          "page": X
        },
        // ... there may be multiple quotes supporting the link ...
      ],
      // This should be true only if one of the provided quotes explicitly names
      // the general rule term. If none do, it MUST be false. This requires a
      // strict and literal reading of the quotes.
      "exception_names_general_rule": true/false,
      // If exception_names_general_rule is false, list any quotes that explicitly
      // discount the possibility of a link between the exception and the general rule.
      // Use an empty list if no such quotes are found.
      "quotes_discounting_link": [
        {
          // Quotes should be verbatim. At least a couple of sentences or a 
          // paragraph long if possible. Avoid use of added ellipses (...) unless
          // absolutely necessary.
          "text": "<verbatim quote>",
          "rulebook_name": "<rulebook name>",
          "page": X
        },
        // ... there may be multiple quotes discounting the link ...
      ],
      // Indicate here whether the exception applies to the user query. Indicate
      // true only if 'exception_names_general_rule' is true. Indicate false if
      // quotes_discounting_link contains a quote that discounts the exception
      // applying. Otherwise, indicate "clarification_needed".
      "does_exception_apply": true/false/"clarification_needed",
      // Clarifying questions here are used when exception_names_general_rule is
      // false or does_exception_apply is "clarification_needed". They should be
      // fully self-contained questions that seek to clarify the link between the
      // exception and the general rule. The question should not depend on context
      // outside the question itself. They should not repeat the original user 
      // query or depend on context outside the question itself.
      "clarifying_question": "<if exception_names_general_rule is false, write a clarifying question here; otherwise, leave empty>"
    },
    // ...
  ],

  // Provide your step-by-step reasoning process here using bullet points. This
  // will not be shown to the user, but will help ensure your final answer is
  // well reasoned and accurate.
  "reasoning": "...",

  // Free form markdown text for the final answer. Must not rely on any
  // definitions for which defines_term is false or any exceptions for which
  // does_exception_apply is false or "clarification_needed". May use multiple
  // lines of text, tables, bullet points, and any other markdown formatting.
  // Adhere to all quoting and citation requirements above. In particular, only
  // use blockquotes for quotes (never inline quotes), and always include
  // citations after each quote. Do NOT paraphrase.
  // The final answer should be consistent with the reasoning provided. The
  // answer should also be consistent with itself. The answer should ignore
  // exceptions for which does_exception_apply is false or "clarification_needed".
  "final_answer": "...",
  // Indicate whether "final_answer" contains sufficient information to answer the
  // user query based on the provided documents. Indicate true only if the final
  // answer does not rely on any definitions for which defines_term is false or
  // any exceptions for which does_exception_apply is false or "clarification_needed".
  "sufficient_information_to_answer": true/false
}