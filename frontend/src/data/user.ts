export interface MockUser {
  name: string
  initials: string
  recentGameIds: string[]
}

export const MOCK_USER: MockUser = {
  name: 'Alex',
  initials: 'AJ',
  // Recently used games — Munchkin and Warhammer from eval test cases, plus two others
  recentGameIds: ['munchkin', 'warhammer_5th_edition', 'catan', 'codenames'],
}
