export interface Game {
  id: string
  name: string
  shortName: string
  emoji: string
  bgColor: string
  exampleQuestions?: string[]
}

export interface GameInfo {
  id: string
  name: string
  emoji?: string
  background_color?: string
  example_questions?: string[]
}

export interface GamesPage {
  pageInfo: {
    hasNextPage: boolean
    startCursor?: string
    endCursor?: string
  }
  data: GameInfo[]
}

export function apiGameToGame(g: GameInfo): Game {
  return {
    id: g.id,
    name: g.name,
    shortName: g.name,
    emoji: g.emoji ?? '🎲',
    bgColor: g.background_color ?? '#374151',
    exampleQuestions: g.example_questions,
  }
}
