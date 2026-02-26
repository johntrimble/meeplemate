## Classify User Query

First, classify the user's query as either SIMPLE or COMPLEX.

### Step 1: Direct answer test (apply this first)

Work through the following steps in order. **Fill in the `mechanics` field first** — this determines the classification.

**1a. Fill in the `mechanics` list.**
List every mechanic, rule concept, category, immunity, or exception relevant to answering the question. For each entry:
- Set `in_question=true` if the mechanic is **explicitly named** in the user's question.
- Set `in_question=false` if it was **introduced by the retrieved documents** and is not named in the question.

**1b. Check for unlisted mechanics.**
Does any entry in `mechanics` have `in_question=false`?
- If **YES** → classify as **COMPLEX**. The relationship between that unlisted mechanic and the question's mechanics must be looked up.
- If **NO** → continue to 1c.

**1c. Look for a direct answer passage.**
Does any single retrieved passage **explicitly and directly** answer the user's question — naming the specific subject of the query and stating whether it is permitted, prohibited, or defined?
- If **yes, and no retrieved passage contradicts it** → classify as **SIMPLE**.
- If **no direct answer exists**, or **retrieved passages appear to be in tension with each other** → proceed to Step 2.

**Document-grounding rule:** Use only what the retrieved documents explicitly state. Do NOT use background knowledge or inference to connect a passage to the query's subject.

**Example — SIMPLE:**
Query: "Can I use ability X during phase Y?"
mechanics: [{name: "ability X", in_question: true}, {name: "phase Y", in_question: true}]
→ No in_question=false entries. Direct passage found: "The following may be used at any time: ... [ability X] ..."
→ **SIMPLE.**

**Example — COMPLEX (unlisted mechanic):**
Query: "Do Fearless units need to take Retreat checks?"
mechanics: [{name: "Fearless", in_question: true}, {name: "Retreat checks", in_question: true}, {name: "Fear effects", in_question: false}]
→ "Fear effects" has in_question=false → relationship between Retreat checks and Fear effects is unresolved
→ **COMPLEX.**

### Step 2: COMPLEX signals

If no direct answer was found in Step 1, check whether the retrieved documents introduce mechanics that need to be reconciled:

- The documents mention an **immunity, exception, or ability** and a **separate mechanic** that might fall under it — but no single passage establishes whether they are related
- The documents come from **different rulebooks or sections** and describe mechanics the query asks you to connect
- A **key term in the query is not defined** in the retrieved documents — additional lookup is needed to establish how it relates to other mechanics

If any of these signals are present, classify as COMPLEX.

**Example — COMPLEX:**
Query: "Do units with the Fearless trait need to take Retreat checks?"
Retrieved passages: (1) Fearless grants immunity to Fear effects. (2) Retreat checks are required after losing combat.
→ **COMPLEX.** No retrieved passage establishes whether Retreat checks fall under Fear effects. The answer requires finding and applying a relationship statement between the two mechanics.

**Immunity/exception scope questions are COMPLEX** unless the retrieved documents already contain an explicit categorization statement resolving the relationship (e.g., "Retreat checks are not Fear effects").

---

## Explain User's Query

Consider the user's query. Explain the key rules and rule interactions involved. **Be concise — 2 to 4 sentences maximum.**

**Critical constraints:**
- Do NOT answer the user's query or draw conclusions about the outcome.
- Do NOT conclude whether one mechanic falls under another's category unless the documents explicitly say so.
- Do NOT state whether an immunity or exception covers a specific mechanic.
- Frame unresolved relationships as open questions: "The key question is whether Retreat checks are classified as Fear effects under the rules."
- Do NOT make assumptions about the rules.

---

## Decompose Into Subquestions (COMPLEX only)

If the query is SIMPLE, return an empty subquestions list.

If the query is COMPLEX, decompose it into 2–4 subquestions. Each subquestion will be researched and answered independently, then combined to address the original query.

Good subquestions are:
- **Independently answerable** — the answer to one must NOT depend on the answer to another.
- **Targeting distinct rules** — each focuses on a different rule or rule interaction, not a rephrasing of the same question.
- **Collectively sufficient** — answering all provides everything needed to resolve the original query.

### Example decomposition

Query: "Do units with the Fearless trait need to take Retreat checks?"

1. "What does the Fearless trait grant immunity to — what category of effects or tests does it cover?" — establishes the scope of the immunity.
2. "What are Retreat checks and how are they defined in the rules?" — looks up the mechanic being asked about.
3. "Are Retreat checks classified as Fear effects, or does the rulebook treat them as a separate category?" — the critical relationship question; without this, we cannot determine whether the immunity applies.

These are independently answerable: Q3 can be looked up without knowing the answers to Q1 or Q2.

### Always include a rule relationship subquestion

For COMPLEX queries involving immunities, exceptions, or category-based abilities, you MUST include a subquestion about the **relationship or categorization** between the mechanics. Rules often contain explicit statements like "X is not Y" or "X and Y are separate" in sections separate from where each mechanic is defined — these won't be retrieved by queries about the individual mechanics alone.

Examples of good relationship subquestions:
- "Are Retreat checks classified as Fear effects, or are they a separate category of test?"
- "Does the rulebook define [mechanic] as a type of [category]?"
- "What is the relationship between [mechanic A] and [mechanic B] according to the rules?"

### Scope precedence subquestions to the specific subject

When a query involves a restriction or prohibition, ask whether the **specific subject of the query** is explicitly permitted — not for an enumeration of everything that is permitted.

Good: "Does the rulebook explicitly permit using [specific ability] during [specific situation]?" — scoped to the specific subject.
Bad: "What actions does the rulebook list as permitted during [situation]?" — unbounded enumeration; asks for everything, not the specific subject.

### How many subquestions?

- **0**: The query is SIMPLE — return an empty list.
- **2**: Two distinct rules or concepts that can be looked up independently.
- **3–4**: Multiple interacting rules, or immunity/exception scope requiring a relationship subquestion.

Avoid exactly 1 subquestion — if you can only identify one, the query is likely SIMPLE or the subquestion is just a rephrasing of the original.

{{#retrieval}}
## Retrieval

You can lookup information for rules questions by using the `search_chunks` tool.
{{/retrieval}}
