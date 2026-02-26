import { useNavigate } from 'react-router-dom'
import { GAMES, type Game } from '@/data/games'
import { MOCK_USER } from '@/data/user'

function GameCard({ game, size = 'md' }: { game: Game; size?: 'sm' | 'md' }) {
  const navigate = useNavigate()
  const textSize = size === 'sm' ? 'text-xl' : 'text-3xl'

  return (
    <button
      onClick={() => navigate('/chat')}
      className="flex flex-col items-center gap-1.5 group"
    >
      <div
        className="w-full aspect-square rounded-xl flex items-center justify-center shadow-sm transition-transform group-hover:scale-105 active:scale-95"
        style={{ backgroundColor: game.bgColor }}
      >
        <span className={textSize}>{game.emoji}</span>
      </div>
      <span className="text-[11px] text-muted-foreground text-center leading-tight w-full px-0.5 line-clamp-2">
        {game.shortName}
      </span>
    </button>
  )
}

export default function SelectGamePage() {
  const recentGames = MOCK_USER.recentGameIds
    .map((id) => GAMES.find((g) => g.id === id))
    .filter(Boolean) as Game[]

  return (
    <div className="min-h-screen bg-background flex flex-col max-w-lg mx-auto">
      {/* Header */}
      <header className="flex items-center justify-between px-4 pt-4 pb-2">
        <h1 className="text-lg font-semibold text-foreground">Select a Game</h1>
        <div className="w-9 h-9 rounded-full bg-muted flex items-center justify-center shrink-0">
          <span className="text-xs font-semibold text-muted-foreground">{MOCK_USER.initials}</span>
        </div>
      </header>

      <div className="flex-1 overflow-y-auto px-4 pb-8">
        {/* Recently Used */}
        <section className="mt-4">
          <h2 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-3">
            Recently Used
          </h2>
          <div className="grid grid-cols-4 gap-3">
            {recentGames.map((game) => (
              <GameCard key={game.id} game={game} size="md" />
            ))}
          </div>
        </section>

        {/* All Games */}
        <section className="mt-6">
          <div className="grid grid-cols-3 gap-4">
            {GAMES.map((game) => (
              <GameCard key={game.id} game={game} size="md" />
            ))}
          </div>
        </section>
      </div>
    </div>
  )
}
