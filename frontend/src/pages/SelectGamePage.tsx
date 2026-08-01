import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { RefreshCw, Search, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Skeleton } from '@/components/ui/skeleton'
import { LoadingLabel } from '@/components/LoadingLabel'
import { type Game } from '@/data/games'
import { UserMenu } from '@/components/UserMenu'
import { useGameList } from '@/hooks/useGameList'
import { useRecentGames } from '@/hooks/useRecentGames'

// Shared responsive column layout for the game grid and its loading skeleton,
// so the two stay in sync. Scales from 3 columns on phones up to 7 on wide
// screens, which keeps tiles from ballooning on desktop.
const GAME_GRID = 'grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-6 xl:grid-cols-7 gap-4'

function GameCard({ game }: { game: Game }) {
  const navigate = useNavigate()

  return (
    <button
      onClick={() => navigate(`/chat/${game.id}`)}
      aria-label={game.name}
      className="flex flex-col items-center gap-1.5 group rounded-xl outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
    >
      <div
        className="@container w-full aspect-square rounded-xl flex items-center justify-center shadow-sm ring-1 ring-white/10 transition-transform group-hover:scale-105 active:scale-95"
        style={{ backgroundColor: game.bgColor }}
      >
        {/* Size the emoji relative to the tile (container-query units) so every
            icon fills the same proportion of its card, regardless of how large
            the tile is at the current breakpoint. */}
        <span className="text-[46cqi] leading-none" aria-hidden="true">
          {game.emoji}
        </span>
      </div>
      <span className="text-[13px] font-medium text-foreground/90 text-center leading-tight w-full px-0.5 line-clamp-2">
        {game.shortName}
      </span>
    </button>
  )
}

/** Placeholder grid shown during the initial catalog load (incl. cold start). */
function GameGridSkeleton({ count = 21 }: { count?: number }) {
  return (
    <div className={GAME_GRID} aria-hidden="true">
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} className="flex flex-col items-center gap-1.5">
          <Skeleton className="w-full aspect-square rounded-xl" />
          <Skeleton className="h-3 w-12" />
        </div>
      ))}
    </div>
  )
}

export default function SelectGamePage() {
  const { games: recentGames, isLoading: recentLoading } = useRecentGames()
  const { games, isLoading, isFetchingMore, error, hasMore, loadMore, refetch } = useGameList()
  const [query, setQuery] = useState('')
  const sentinelRef = useRef<HTMLDivElement>(null)

  const q = query.trim().toLowerCase()
  const isSearching = q.length > 0

  // Client-side filter over the already-loaded catalog. `useGameList` fetches
  // the whole catalog in one page (first: 100), so this covers everything today.
  // If the catalog ever grows past one page, move search server-side so matches
  // on not-yet-loaded games aren't missed.
  const filteredGames = useMemo(
    () =>
      isSearching
        ? games.filter(
            (g) =>
              g.name.toLowerCase().includes(q) ||
              g.shortName.toLowerCase().includes(q),
          )
        : games,
    [games, q, isSearching],
  )

  // Recently Used is a shortcut for the default view; while searching we show a
  // single, unified list of matches instead.
  const showRecent = !isSearching && !recentLoading && recentGames.length > 0

  useEffect(() => {
    const sentinel = sentinelRef.current
    if (!sentinel) return

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting && hasMore) {
          loadMore()
        }
      },
      { rootMargin: '200px' },
    )

    observer.observe(sentinel)
    return () => observer.disconnect()
  }, [hasMore, loadMore])

  return (
    <div className="h-full bg-background flex flex-col">
      {/* Header */}
      <header className="sticky top-0 z-30 border-b border-border bg-background shrink-0">
        <div className="max-w-5xl mx-auto px-4 py-3">
          <div className="flex items-center justify-between">
            <h1 className="text-lg font-semibold text-foreground">Select a Game</h1>
            <UserMenu className="w-9 h-9" />
          </div>
          <div className="relative mt-3">
            <Search
              className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 size-4 text-muted-foreground"
              aria-hidden="true"
            />
            <Input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Escape') setQuery('')
              }}
              placeholder="Search games…"
              aria-label="Search games"
              className="pl-9 pr-9"
            />
            {isSearching && (
              <button
                type="button"
                onClick={() => setQuery('')}
                aria-label="Clear search"
                className="absolute right-2 top-1/2 -translate-y-1/2 rounded-md p-1 text-muted-foreground transition-colors hover:text-foreground outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                <X className="size-4" />
              </button>
            )}
          </div>
        </div>
      </header>

      <div className="flex-1 min-h-0 overflow-y-auto">
        <div className="max-w-5xl mx-auto px-4 pb-8">
          {/* Recently Used */}
          {showRecent && (
            <section className="mt-6">
              <h2 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-3">
                Recently Used
              </h2>
              <div className="grid grid-cols-4 sm:grid-cols-6 lg:grid-cols-8 gap-3">
                {recentGames.map((game) => (
                  <GameCard key={game.id} game={game} />
                ))}
              </div>
            </section>
          )}

          {/* All Games / search results */}
          <section className="mt-8">
            <h2 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-3">
              {isSearching ? `Results (${filteredGames.length})` : 'All Games'}
            </h2>
            {isLoading && games.length === 0 ? (
              <GameGridSkeleton />
            ) : error && games.length === 0 ? (
              <div className="flex flex-col items-center gap-3 py-8">
                <p className="text-sm text-destructive text-center">{error}</p>
                <Button variant="outline" size="sm" onClick={() => refetch()}>
                  <RefreshCw />
                  Try again
                </Button>
              </div>
            ) : filteredGames.length === 0 ? (
              <div className="text-sm text-muted-foreground text-center py-8">
                {isSearching ? `No games match "${query.trim()}".` : 'No games available.'}
              </div>
            ) : (
              <div className={GAME_GRID}>
                {filteredGames.map((game) => (
                  <GameCard key={game.id} game={game} />
                ))}
              </div>
            )}
            {/* Sentinel for infinite scroll */}
            <div ref={sentinelRef} />
            {isFetchingMore && (
              <LoadingLabel className="block text-sm text-muted-foreground text-center py-4">
                Loading more…
              </LoadingLabel>
            )}
          </section>
        </div>
      </div>
    </div>
  )
}
