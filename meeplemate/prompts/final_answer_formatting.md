# Final Answer Formatting Prompt (Second Pass)

You previously analyzed a rules question and produced a structured JSON response with quotes, reasoning, and a preliminary answer. Now you need to create a properly formatted final answer for the user.

## Your Task

Create a well-formatted markdown answer that:
1. Directly answers the user's question
2. Cites the relevant rules using proper blockquote formatting
3. Explains the reasoning clearly

## Required Markdown Format

**For every rule you cite, you MUST use this exact structure:**

```markdown
[Introductory text explaining the rule]

> Verbatim quote text from the rulebook.
> Multiple lines if the quote spans multiple lines.

(Rulebook Name, p. XX)

[Explanation of what this means]
```

**Critical Requirements:**
- Use `>` at the start of EVERY line of a quote
- Put a blank line before the quote
- Put a blank line after the quote
- Put the citation `(Rulebook Name, p. XX)` on its own line after the blank line
- Copy quotes VERBATIM from the quote collections in your previous response
- NEVER use inline quotes like `"text here" (Rulebook, p. XX)` - always use blockquotes
- NEVER paraphrase rules - use exact text

## Source Material

Use the quotes from your previous structured response:
- `supporting_quotes[]` - for all relevant rules, exceptions, and relationships
- `exceptions[].exception_quotes[]` - for exception sources (cards, abilities, etc.)

All quotes are stored as plain text in the JSON. When formatting them in markdown, add the `>` character at the start of each line.

## Structure Your Answer

A good answer typically follows this structure:

1. **Direct answer** to the question (Yes/No, or a clear statement)
2. **General rule** (if relevant) - quoted with citation
3. **Exception or special case** (if applicable) - quoted with citation
4. **Conclusion** explaining why the exception applies or doesn't apply, referencing precedence if needed

## Examples

### Example 1: Simple case with one rule

**Good:**
```markdown
No, Stealth units cannot be targeted by standard ranged attacks.

> Stealth units cannot be targeted by ranged attacks unless the attacking unit has a special ability that allows it to detect hidden targets.

(Core Rulebook, p. 28)

Since standard ranged attacks do not have such an ability, they cannot target Stealth units.
```

**Bad:**
```markdown
No, because "Stealth units cannot be targeted by ranged attacks" (Core Rulebook, p. 28).
```
Problem: Inline quote with quotation marks instead of blockquote

### Example 2: Exception overriding general rule

**Good:**
```markdown
Yes, the Seeking Shot ability can target Stealth units.

The general rule prohibits targeting Stealth units:

> Stealth units cannot be targeted by ranged attacks unless the attacking unit has a special ability that allows it to detect hidden targets.

(Core Rulebook, p. 28)

However, the Seeking Shot ability explicitly creates an exception:

> Seeking Shot: This ability allows the unit to target any enemy unit, including those with Stealth. The shot automatically detects and tracks hidden targets.

(Advanced Abilities, p. 15)

Since Seeking Shot explicitly states it can target "units with Stealth," this specific ability overrides the general rule. According to the precedence hierarchy, specific abilities on cards override general rules when they explicitly name the situation.
```

**Bad:**
```markdown
Yes, because the general rule says "Stealth units cannot be targeted" (Core Rulebook, p. 28) but Seeking Shot says it "can target any enemy unit, including those with Stealth" (Advanced Abilities, p. 15).
```
Problem: Inline quotes instead of blockquotes

## Now Generate the Final Answer

Based on your previous structured analysis, create a properly formatted final answer following all the rules above. Use blockquotes for all rule citations and copy the exact text from your quote collections.