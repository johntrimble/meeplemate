## Your Task

Based on the answer you just provided to the user's query, create a well-formatted markdown response. Before writing the final answer, reflect on the answer and decide on the best structure.

**CRITICAL: Do not change the conclusion.** Your reasoning was already done in the previous step. This step is only about presentation. You MAY rephrase the answer and explanation for clarity, conciseness, and to better fit the question actually asked. Use the user's query (provided in the system prompt) to frame your response — ensure the opening sentence directly addresses what was asked.

## Step 1: Reflect

Before writing the answer, reason inside `<reasoning></reasoning>` tags. Cover:

1. **Question type**: Is this a simple lookup (one rule, no interactions), or a complex interaction (exceptions, precedence, multiple mechanics)?
2. **Structure**: Based on the question type, which format will you use — Simple or Complex (defined below)?
3. **Load-bearing quotes**: Which quotes from the previous answer are actually necessary to support the conclusion? Any quote that does not directly support the conclusion should be dropped.
4. **Polarity**: Determine the query type from its **first word**, not from words within it.
   - **Yes/no questions** start with: "Can", "Is", "Are", "Do", "Does", "Must", "Should", "Will", "May" — answer with **"Yes"** or **"No"** to match the conclusion. Never open with "Yes" when the conclusion is a restriction or prohibition.
   - **Open questions** start with: "When", "How", "What", "Which", "Where", "Who", "Why" — do **not** open with "Yes" or "No". Answer directly. For example, "When can I discard a Race card?" should open with "You can discard a Race card..." not "Yes, you can...".

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

- Quote text from the provided documents as needed.
- Use `>` at the start of EVERY line of a quote, including blank lines within the quote
- Put the citation inside the blockquote on its own line, separated from the quote text by a blank `>` line:

  ```
  > Quote text here.
  >
  > (Rulebook Name, p. X)
  ```

- NEVER place the citation outside the blockquote
- NEVER use inline quotes like `"text here" (Rulebook, p. X)` — always use blockquotes
- NEVER paraphrase rules — use exact text
- When quoting a table, quote the entire table as it appears in the rulebook
- NEVER escape HTML tags in quotes — if the rulebook uses `<table>`, quote it as-is
