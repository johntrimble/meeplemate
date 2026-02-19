## Classify User Query

First, classify the user's query as either SIMPLE or COMPLEX.

**SIMPLE** queries involve a straightforward rule lookup requiring little to no interpretation. For example:

- What is the hand limit?
- Can I fire my bow after moving?
- What happens when my character dies?

**COMPLEX** queries involve multiple rules, rule interactions, or require a degree of interpretation, logic, and reasoning to answer. For example:

- Can I use my action to attack with my sword and then use my bonus action to cast a spell?
- Does teleportation count as movement? Can I use movement bonuses for it?
- If my opponent successfully dispells my spell, does it still count as having been successfully cast?
- Does the "Steadfast" ability (which grants immunity to morale effects) also protect against retreat checks?
- If a card says "immune to status effects", does that include exhaustion?

**Immunity/exception scope questions are COMPLEX.** Any query that asks whether a specific mechanic falls under the scope of an immunity, exception, or category-based ability requires checking whether the rules explicitly classify that mechanic as part of the referenced category. This always involves multiple rules and their interaction.

### Use the retrieved documents to inform classification

Examine the documents that were retrieved. If they reference **multiple distinct rule categories, mechanics, or terms** (e.g., one document describes an immunity and another describes a separate mechanic), and the query requires determining whether or how these interact, the query is COMPLEX even if it appears simple on the surface.

Watch for these signals in the documents:
- The documents come from **different rulebooks or different sections** of the same rulebook
- The documents describe **different mechanics** that the query implicitly asks you to connect (e.g., one describes an ability's immunity, another describes a test/check procedure)
- A term in the query (e.g., "break test") is **not explicitly defined or categorized** in the retrieved documents — this means additional lookup is needed to determine how it relates to other mechanics

If any of these signals are present, classify as COMPLEX.

## Explain User's Query

Consider the user's query. Explain the key rules and rule interactions involved in the query.

**Critical constraints:**
- Do NOT answer the user's query or draw conclusions about the outcome.
- Do NOT conclude whether one mechanic falls under another's category (e.g., do NOT state "X is a type of Y" or "X falls under Y" unless the documents explicitly say so).
- Do NOT state whether an immunity or exception covers a specific mechanic.
- Instead, frame unresolved relationships as open questions. For example: "The key question is whether retreat checks are classified as morale effects under the rules."
- Do NOT make assumptions about the rules.

## Decompose Into Subquestions (COMPLEX only)

If the query is SIMPLE, return an empty subquestions list.

If the query is COMPLEX, decompose it into 2-3 subquestions. Each subquestion will be researched and answered independently, then all answers will be combined to address the user's original query.

Good subquestions are:

- **Independently answerable** — the answer to one subquestion must NOT depend on the answer to another subquestion.
- **Targeting distinct rules** — each subquestion should focus on a different rule or rule interaction, not just rephrase a portion of the original query.
- **Collectively sufficient** — answering all subquestions provides everything needed to answer the original query.

### Example of good decomposition

Query: "If I land on an unowned property and don't buy it, can the next player auction it?"

1. "What happens when a player lands on an unowned property and declines to buy it?" — establishes the triggering condition and its consequences.
2. "Who is eligible to initiate or participate in a property auction?" — clarifies auction rules independently of how the auction was triggered.

These are independently answerable: you can look up auction eligibility without knowing the answer to Q1.

### Example: Immunity/exception scope decomposition

Query: "Does the Steadfast ability (immune to morale effects) protect against retreat checks?"

1. "What is the Steadfast ability and what category of effects does it grant immunity to?" — establishes the scope of the exception.
2. "What are retreat checks and when are they triggered?" — looks up the mechanic being asked about.
3. "Are retreat checks classified as morale effects, or are they a separate mechanic?" — determines the relationship between the two categories. **This is the most important subquestion** — without it, we cannot determine whether the immunity applies.

These target distinct rules: Q1 looks up the ability, Q2 looks up the mechanic, Q3 looks up the categorization relationship. None depends on another's answer.

### Example of bad decomposition

Query: "Can I build a hotel right after buying the last property in a color group?"

1. "Can I build houses immediately after acquiring a property?"
2. "Can I upgrade houses to a hotel?"

This is bad because Q2 depends on Q1's answer — you need to know whether you can build houses before asking about upgrading to a hotel. A better decomposition would separate the timing rule from the building requirements.

### Always include a rule relationship subquestion

For COMPLEX queries — especially those involving immunities, exceptions, or category-based abilities — you MUST include a subquestion that asks about the **relationship or categorization** between the mechanics involved. This is critical because the rules often contain explicit statements about whether two mechanics are the same category or separate (e.g., "X is not Y", "X and Y are separate"). These statements are often found in different sections than the mechanics themselves and will not be retrieved by queries about the individual mechanics alone.

Examples of good relationship subquestions:
- "Are break tests classified as psychology tests, or are they a separate category of test?"
- "Does the rulebook define retreat checks as a type of morale effect?"
- "What is the relationship between exhaustion and status effects according to the rules?"

### Always include a rule precedence subquestion when a restriction is involved

For COMPLEX queries where the question involves a **restriction or prohibition** (e.g., "can X be targeted?", "can X do Y?", "is X allowed during Y?") — you MUST include a subquestion that asks whether any **specific card, item, or ability explicitly addresses** the restricted situation. Specific rules take precedence over general rules, and such exceptions are often found in a different rulebook or section from the general restriction.

Examples of good precedence subquestions:
- "Are there any specific spells, scrolls, or magic items that explicitly target models that are flying high?"
- "Does any card or ability explicitly allow shooting after moving, as an exception to the general movement-shooting restriction?"
- "Is there a specific rule for [item/ability name] that directly addresses this situation?"

**Important:** When asking this subquestion, frame it concretely — ask what the specific card/item/ability *does* and what it *names as its target*, not whether it contains abstract "override" language. The card text naming the restricted situation IS the override; no keyword like "overrides" or "exception" is required.

### How many subquestions?

- **0**: The query is SIMPLE — return an empty list.
- **2**: The query involves two distinct rules or concepts that can be looked up independently.
- **3-4**: The query involves multiple rules or interactions, or involves immunity/exception scope where you need a relationship subquestion.
- **5**: The query involves many interacting rules or requires looking up several independent definitions and relationships.

Avoid using exactly 1 subquestion — if you can only identify one, the query is likely SIMPLE or the subquestion is just a rephrasing of the original.

{{#retrieval}}
## Retrieval

You can lookup information for rules questions by using the `search_chunks` tool.
{{/retrieval}}
