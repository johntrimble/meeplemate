import { useNavigate } from 'react-router-dom'
import { Button } from '@/components/ui/button'

export default function HomePage() {
  const navigate = useNavigate()

  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-3xl mx-auto px-4 flex flex-col">
      {/* Header */}
      <header className="flex items-center justify-end pt-4 pb-2">
        <Button variant="outline" size="sm" onClick={() => navigate('/select-game')}>
          Sign In
        </Button>
      </header>

      {/* Hero */}
      <main className="flex-1 flex items-start gap-4 pt-8">
        {/* Logo card */}
        <div className="flex-shrink-0 w-36 h-36 rounded-2xl border border-border bg-card flex items-center justify-center shadow-sm">
          <div className="flex flex-col items-center gap-1">
            <span className="text-3xl">🎲</span>
            <span className="text-xs font-semibold text-muted-foreground tracking-wide uppercase">
              Boardbarian
            </span>
          </div>
        </div>

        {/* Marketing copy */}
        <div className="flex-1 pt-2">
          <h1 className="text-base font-semibold text-foreground mb-2">
            Rules questions, answered instantly.
          </h1>
          <p className="text-sm text-muted-foreground leading-relaxed">
            Boardbarian is an AI assistant that answers board game rules
            questions so you can spend less time flipping through rulebooks and
            more time playing.
          </p>
        </div>
      </main>
      </div>
    </div>
  )
}
