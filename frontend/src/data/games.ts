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
  summary?: string
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

export const GAMES: Game[] = [
  // Games from eval test cases
  { id: 'munchkin', name: 'Munchkin', shortName: 'Munchkin', emoji: '⚔️', bgColor: '#7f1d1d' },
  { id: 'warhammer_5th_edition', name: 'Warhammer Fantasy Battle', shortName: 'Warhammer', emoji: '🏰', bgColor: '#1e293b' },
  // Popular board games
  { id: 'catan', name: 'Catan', shortName: 'Catan', emoji: '🏝️', bgColor: '#92400e' },
  { id: 'pandemic', name: 'Pandemic', shortName: 'Pandemic', emoji: '🦠', bgColor: '#134e4a' },
  { id: 'ticket_to_ride', name: 'Ticket to Ride', shortName: 'Ticket to Ride', emoji: '🚂', bgColor: '#7c2d12' },
  { id: 'codenames', name: 'Codenames', shortName: 'Codenames', emoji: '🕵️', bgColor: '#1e3a5f' },
  { id: 'carcassonne', name: 'Carcassonne', shortName: 'Carcassonne', emoji: '🏯', bgColor: '#14532d' },
  { id: 'dominion', name: 'Dominion', shortName: 'Dominion', emoji: '👑', bgColor: '#713f12' },
  { id: 'arkham_horror', name: 'Arkham Horror', shortName: 'Arkham Horror', emoji: '👁️', bgColor: '#3b0764' },
  { id: 'gloomhaven', name: 'Gloomhaven', shortName: 'Gloomhaven', emoji: '🗡️', bgColor: '#44403c' },
  { id: '7_wonders', name: '7 Wonders', shortName: '7 Wonders', emoji: '🏛️', bgColor: '#312e81' },
  { id: 'wingspan', name: 'Wingspan', shortName: 'Wingspan', emoji: '🦅', bgColor: '#0c4a6e' },
  { id: 'azul', name: 'Azul', shortName: 'Azul', emoji: '🎨', bgColor: '#164e63' },
]
