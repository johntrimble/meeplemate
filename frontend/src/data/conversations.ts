export interface ThinkingStep {
  label: string
}

export interface ConversationMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  /** Chain-of-thought steps shown in the collapsible thinking section */
  thinkingSteps?: ThinkingStep[]
  /** How long (in seconds) the model thought before answering */
  thinkingDuration?: number
}

export interface MockConversation {
  id: string
  gameId: string
  title: string
  messages: ConversationMessage[]
}

export const MOCK_CONVERSATIONS: MockConversation[] = [
  // --- Munchkin ---
  {
    id: 'conv-1',
    gameId: 'munchkin',
    title: 'Monster drawn face-up & loot the room',
    messages: [
      {
        id: 'm1-1',
        role: 'user',
        content:
          'If I draw a monster face up during the "Kick Open The Door" phase, and the monster card says "Will not pursue anyone of level 3 or below", and I\'m only level 2, do I still count as being in combat and does that mean I cannot loot the room?',
      },
      {
        id: 'm1-2',
        role: 'assistant',
        thinkingDuration: 12,
        thinkingSteps: [
          { label: 'Looking up Kick Open The Door phase rules' },
          { label: 'Checking whether pursuit mechanics affect mandatory combat' },
          { label: 'Verifying when Loot The Room is available' },
        ],
        content: `Yes, you must still count as being in combat, and you cannot loot the room.

> (1) Kick Open The Door: Draw one card from the Door deck and turn it face up. If it's a monster, you must fight it.
>
> (2) Look For Trouble/Loot The Room: If you fought a monster in phase 1, skip this phase and go to phase 3.
>
> (Munchkin Rules, p. 2)

This rule explicitly states that if a monster is drawn during phase 1, the player "must fight it" and skips phase 2 entirely.

The monster's **"Will not pursue"** clause applies only to **Running Away** — it makes escape easier — but it does not affect whether combat is initiated. Drawing a monster face-up during Kick Open The Door always triggers mandatory combat, regardless of pursuit restrictions.

Therefore, you count as being in combat and **cannot loot the room**.`,
      },
    ],
  },
  {
    id: 'conv-2',
    gameId: 'munchkin',
    title: 'Discarding a Class card mid-combat',
    messages: [
      {
        id: 'm2-1',
        role: 'user',
        content: 'If I discard my Class card during combat, do I lose all the bonuses from that class immediately?',
      },
      {
        id: 'm2-2',
        role: 'assistant',
        thinkingDuration: 6,
        thinkingSteps: [
          { label: 'Looking up Class card discard rules' },
          { label: 'Checking timing for "any time" actions during combat' },
        ],
        content: `Yes, discarding your Class card means you **immediately** lose all benefits from that class, even in combat.

> You may discard a Class or Race card at any time, even in combat. (Munchkin Rules, p. 2)

Once discarded, you become a human (no class) and lose any combat bonuses, special abilities, or item restrictions associated with that class. This takes effect immediately — mid-combat and mid-turn.

Note that this can be a strategic move: some classes impose penalties (e.g., Clerics cannot use certain items), so discarding mid-combat can remove those restrictions.`,
      },
      {
        id: 'm2-3',
        role: 'user',
        content: 'What if discarding the class card causes me to lose enough combat strength to no longer be winning?',
      },
      {
        id: 'm2-4',
        role: 'assistant',
        thinkingDuration: 8,
        thinkingSteps: [
          { label: 'Reviewing combat strength calculation rules' },
          { label: 'Checking rules for combat outcome changes mid-fight' },
        ],
        content: `If discarding your Class card reduces your combat strength below the monster's, the combat outcome changes immediately — you are now **losing** the fight.

> At any point during combat, if a player's total combat strength drops below the monster's, they are considered to be losing. (Munchkin Rules, p. 3)

This means:
- If you cannot bring your strength back up (via items, helpers, or cards), you must attempt to **Run Away**
- You roll a die and escape on a 5 or 6 (modified by any applicable bonuses or penalties on the monster card)
- If you fail to run away, you suffer the monster's **Bad Stuff**

So discarding your Class card mid-combat is a risky move if it lowers your combat strength significantly.`,
      },
    ],
  },
  // --- Warhammer Fantasy Battle ---
  {
    id: 'conv-3',
    gameId: 'warhammer_5th_edition',
    title: 'Grail Knights and break tests',
    messages: [
      {
        id: 'm3-1',
        role: 'user',
        content: 'Do Grail Knights need to take break tests?',
      },
      {
        id: 'm3-2',
        role: 'assistant',
        thinkingDuration: 8,
        thinkingSteps: [
          { label: 'Finding Grail Knights rules in the Bretonnia Army Book' },
          { label: 'Distinguishing psychology tests from break tests' },
          { label: 'Confirming break test requirement' },
        ],
        content: `Yes, Grail Knights do need to take break tests as normal.

Grail Knights have the **Grail Virtue**, which makes them immune to psychology:

> Knightly Virtues: Grail Knights have the Grail Virtue; they have drunk from the sacred grail and are immune to psychology. (Bretonnia Army Book, p. 62)

However, break tests are **separate** from psychology tests:

> Players will immediately realise that a psychology test is taken in the same way as a Break test in hand-to-hand combat and uses the same characteristic, namely Leadership. However, **a Break test is not a psychology test**. The two tests are quite separate. (Warhammer Rulebook, p. 46)

Therefore, even though Grail Knights are immune to psychology tests, they must still take break tests when they lose combat.`,
      },
      {
        id: 'm3-3',
        role: 'user',
        content: 'When a unit of Grail Knights loses combat against a Green Dragon do they need to take a break test?',
      },
      {
        id: 'm3-4',
        role: 'assistant',
        thinkingDuration: 10,
        thinkingSteps: [
          { label: 'Checking Green Dragon special rules' },
          { label: 'Confirming break test requirement after losing combat' },
        ],
        content: `Yes, Grail Knights must still take a break test when they lose combat against a Green Dragon.

As established, the Grail Virtue grants immunity to **psychology tests** only. Break tests are a separate mechanic and apply to all units that lose combat:

> The side that loses a combat must take a test to determine whether it stands and fights or turns tail and runs away. This is called a Break test. You need to take a separate Break test for every unit involved in the combat. (Warhammer Rulebook, p. 42)

There is no exception for units fighting monsters such as a Green Dragon. Grail Knights must take the break test normally using their Leadership characteristic.`,
      },
    ],
  },
]
