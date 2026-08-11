## Your Task

Based on the answer you just provided to the user's query, create a well-formatted markdown response. Before writing the final answer, reflect on the answer and decide on the best structure.

**CRITICAL: Do not change the conclusion.** Your reasoning was already done in the previous step. This step is only about presentation. You MAY rephrase the answer and explanation for clarity, conciseness, and to better fit the question actually asked. Use the user's query (provided in the system prompt) to frame your response — ensure the opening sentence directly addresses what was asked.

**CRITICAL: A blockquote in your response may only reproduce a blockquote from the previous answer.** The previous answer marked its rulebook quotes by starting every line with `>`. That `>`-marked text is the *only* material eligible to become a blockquote here, so reproduce one verbatim or leave it out. Everything else in the previous answer is prose, however citation-like it looks. The only thing you compose fresh is the opening sentence and brief connective prose. The rules below are binding; the *Blockquote requirements* section spells them out.

## Step 1: Reflect

Before writing the answer, reason inside `<reasoning></reasoning>` tags. Cover:

1. **Question type**: Is this a simple lookup (one rule, no interactions), or a complex interaction (exceptions, precedence, multiple mechanics)?
2. **Structure**: Based on the question type, which format will you use — Simple or Complex (defined below)?
3. **Load-bearing quotes**: List the `>`-marked blockquotes in the previous answer, which are your only candidates. Which of them are actually necessary to support the conclusion? Drop the rest. Text that merely *looks* cited is not a candidate: a paragraph trailed by a parenthetical on its own line, a quoted string sitting inline in a sentence, anything labelled `(Clarifying Question/Answer)` or similar. If you need a point such text makes, state it in your own words, with no quotation marks and no citation.
4. **Opening sentence**: Construct the opening sentence fresh from two inputs: the user's query and the actual conclusion embedded in the answer. Do not copy the opening of the previous answer.
   - First, identify the conclusion: strip away the reasoning steps and ask "what does the answer ultimately say is true?"
   - Then, ask: "given the user's query, how would a person naturally express that conclusion as a direct answer?"
   - If the query is a yes/no question (starts with "Can", "Is", "Are", "Do", "Does", "Must", "Should", "Will", "May"): open with "Yes" or "No" matching the conclusion, followed by a restatement in the user's terms. If the conclusion is a prohibition or restriction, open with "No". If it is a permission or affirmation, open with "Yes".
   - If the query is an open question (starts with "When", "How", "What", "Which", "Where", "Who", "Why"): open directly with the conclusion in the user's terms — do not use "Yes" or "No".
   - Watch for malformed openers in the previous answer such as "Yes, you cannot..." or "Yes, you must not..." — these are contradictions. If you see one, identify the true conclusion from the substance of the answer and construct a correct opener from scratch.

## Step 2: Write the answer

Use the format you chose in Step 1.

---

### Simple format

Use this when the answer follows directly from one rule with no exceptions or rule interactions to explain.

**Structure:**
1. Direct answer (one sentence)
2. The relevant rule, quoted and cited
3. One sentence of explanation only if genuinely needed — otherwise stop after the quote

**Example (affirmative):**

Yes, you must discard down to your hand limit at the end of your turn.

> At the end of your turn, discard cards until you have five cards in your hand.
>
> (Core Rulebook, p. 4)

**Example (prohibition):**

No, you cannot have multiple Steeds equipped at the same time.

> No player can have more than one Steed except by using a Cheat! card.
>
> (Munchkin 4 - The Need for Steed, p. 1)

---

### Complex format

Use this when the answer involves interacting rules, exceptions, or precedence decisions that require explanation.

**Structure:**
1. Direct answer (one sentence)
2. General rule — quoted and cited
3. Exception or special case (if one exists and is relevant) — quoted and cited
4. Conclusion — explain why the exception applies or doesn't apply, or why the general rule governs. **Only include this if there is genuinely something to resolve.** Do not write a conclusion that merely restates that no exception exists.

**Example:**

Yes, the Seeking Shot ability can target Stealth units.

The general rule prohibits targeting Stealth units:

> Stealth units cannot be targeted by ranged attacks unless the attacking unit has a special ability that allows it to detect hidden targets.
>
> (Core Rulebook, p. 28)

However, Seeking Shot explicitly creates an exception:

> Seeking Shot: This ability allows the unit to target any enemy unit, including those with Stealth. The shot automatically detects and tracks hidden targets.
>
> (Advanced Abilities, p. 15)

Since Seeking Shot explicitly names Stealth units, it overrides the general rule.

---

## Blockquote requirements

- Every blockquote must be a verbatim copy of one `>`-marked blockquote from the previous answer. Choose freely which to use and in what order; never alter the text, merge two into one, extend one with neighbouring sentences, or reconstruct a rule the previous answer only paraphrased, not even if you believe you know it from memory
- NEVER promote unmarked text into a blockquote: a quoted string sitting inline in prose, or a paragraph trailed by a parenthetical label, was not a quote in the previous answer and is not one here
- Use `>` at the start of EVERY line of a quote, including blank lines within the quote. A quote you carry over is always a blockquote, never an inline `"text here" (Rulebook, p. X)` construction
- Put the citation inside the blockquote on its own line, separated from the quote text by a blank `>` line, never outside it:

  ```
  > Quote text here.
  >
  > (Rulebook Name, p. X)
  ```

- Keep whatever citation the previous answer gave a quote; never swap in a different rulebook or page. A citation names a rulebook and a page, so `(Clarifying Question/Answer)`, `(Clarifying Q&A)` and similar labels are not citations, and the text they trail is not quotable
- When quoting a table, quote the entire table as it appears in the rulebook
- NEVER escape HTML tags in quotes — if the rulebook uses `<table>`, quote it as-is
