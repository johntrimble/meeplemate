import { useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { type Game } from '@/data/games'
import { MOCK_USER } from '@/data/user'
import { useGameList } from '@/hooks/useGameList'
import { useRecentGames } from '@/hooks/useRecentGames'

function GameCard({ game, size = 'md' }: { game: Game; size?: 'sm' | 'md' }) {
  const navigate = useNavigate()
  const textSize = size === 'sm' ? 'text-xl' : 'text-3xl'

  return (
    <button
      onClick={() => navigate(`/chat/${game.id}`)}
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
  const { games: recentGames, isLoading: recentLoading } = useRecentGames()
  const { games, isLoading, error, hasMore, loadMore } = useGameList()
  const sentinelRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const sentinel = sentinelRef.current
    if (!sentinel) return

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting && hasMore && !isLoading) {
          loadMore()
        }
      },
      { rootMargin: '200px' },
    )

    observer.observe(sentinel)
    return () => observer.disconnect()
  }, [hasMore, isLoading, loadMore])

  return (
    <div className="h-full bg-background flex flex-col">
      {/* Header */}
      <header className="sticky top-0 z-30 border-b border-border bg-background shrink-0">
        <div className="max-w-3xl mx-auto flex items-center justify-between px-4 pt-4 pb-3">
          <h1 className="text-lg font-semibold text-foreground">Select a Game</h1>
          <div className="w-9 h-9 rounded-full bg-muted flex items-center justify-center shrink-0">
            <span className="text-xs font-semibold text-muted-foreground">{MOCK_USER.initials}</span>
          </div>
        </div>
      </header>

      <div className="flex-1 min-h-0 overflow-y-auto">
        <div className="max-w-3xl mx-auto px-4 pb-8">
          {/* Recently Used */}
          {!recentLoading && recentGames.length > 0 && (
            <section className="mt-6">
              <h2 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-3">
                Recently Used
              </h2>
              <div className="grid grid-cols-4 sm:grid-cols-6 gap-3">
                {recentGames.map((game) => (
                  <GameCard key={game.id} game={game} size="md" />
                ))}
              </div>
            </section>
          )}

          {/* All Games */}
          <section className="mt-8">
            {isLoading && games.length === 0 ? (
              <div className="text-sm text-muted-foreground text-center py-8">Loading games…</div>
            ) : error && games.length === 0 ? (
              <div className="text-sm text-destructive text-center py-8">{error}</div>
            ) : games.length === 0 ? (
              <div className="text-sm text-muted-foreground text-center py-8">No games available.</div>
            ) : (
              <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 gap-4">
                {games.map((game) => (
                  <GameCard key={game.id} game={game} size="md" />
                ))}
              </div>
            )}
            {/* Sentinel for infinite scroll */}
            <div ref={sentinelRef} />
            {isLoading && games.length > 0 && (
              <div className="text-sm text-muted-foreground text-center py-4">Loading more…</div>
            )}
          </section>
        </div>
      </div>
    </div>
  )
}
