# How a rules question gets answered

The path a player's question takes, from the moment they ask it to the answer
coming back with rulebook citations attached.

This describes the flow rather than the implementation. For the code, see
`build_chatloop_graph` in `meeplemate/chatloop.py` and the three graph builders in
`meeplemate/qa_graph.py` (`build_coordinating_agent_graph`,
`build_analyze_question_graph`, `build_question_answer_graph`).

## Main flow

Simple questions get answered once; anything else is split up, answered piece by
piece, and reassembled.

```mermaid
flowchart TD
    A([Player asks a rules question]) --> B["Compress the conversation<br/>history"]
    B --> C["Rewrite the question so it<br/>stands on its own"]
    C --> D["Search the rulebooks for<br/>relevant passages"]
    D --> E["Analyse what the question<br/>is really asking"]
    E --> F{"Is it a simple<br/>question?"}

    F -->|Yes| G[["Answer the question"]]
    F -->|No| H["Break it into<br/>subquestions"]

    H --> I[["Answer each subquestion<br/>(all at once)"]]
    I --> K["Filter out subanswers<br/>with invalid quotes"]
    K --> M["Pool the surviving subanswers<br/>and their cited passages"]
    M --> N[["Answer the original question<br/>using the pooled passages"]]

    G --> O[/"Stream the answer and its<br/>citations back to the player"/]
    N --> O
    O --> P([Done])
```

## Answering a question

The boxed-in step above appears three times — for a simple question, for each
subquestion, and for the final pass over the pooled passages. It expands to this:

```mermaid
flowchart TD
    A([Answer the question]) --> B{"Do we already have<br/>the passages?"}

    B -->|No| C["Search the rulebooks"]
    C --> D["Drop duplicate passages"]
    D --> E["Draft an answer from<br/>the passages"]
    B -->|Yes| E

    E --> F["Write it up, quoting the<br/>rulebook as support"]
    F --> G{"Does every quote really<br/>appear in the passages?"}

    G -->|Yes| H([Return the answer,<br/>marked verified])
    G -->|No| I{"Rewritten it<br/>five times?"}
    I -->|No| F
    I -->|Yes| J([Return the answer,<br/>marked unverified])
```

The only loop in the whole system is the rewrite that fires when the model quotes
something the rulebook doesn't actually say.

## Worth knowing

**Hard questions get answered several times over.** A question that isn't simple is
split into subquestions, each answered independently and in parallel. Their answers
and passages are then pooled and the original question is answered afresh on top of
them — so a complex question costs several model passes, not one.

**A subanswer that misquotes the rulebook is thrown away, not fixed.** Quote checking
happens twice: once per subanswer, where failure means the subanswer is silently
dropped from the pool, and again on the final answer, where failure triggers a
rewrite instead.

**Passages are searched for once and then reused.** The final pass over the pooled
passages skips searching entirely, because the subquestions already did that work.

**An unverified answer still reaches the player.** After five failed rewrites the
system stops trying and returns the answer flagged as unverified, rather than
failing outright.
