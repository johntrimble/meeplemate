import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion, useReducedMotion } from 'motion/react'
import { ArrowRightIcon, BadgeCheckIcon, DicesIcon, MessageCircleQuestionIcon } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useAuth } from '@/auth/useAuth'

// ---------------------------------------------------------------------------
// Demo content — a hard-coded sample Q&A used by the animated chat demo.
// ---------------------------------------------------------------------------

const DEMO_QUESTION = "Can I play a Go Up a Level card during combat?"
const DEMO_ANSWER =
  'Yes, you can play a Go Up a Level card during combat. The rules explicitly state: '
const DEMO_QUOTE = 'Go Up a Level cards may be played on yourself or any other player at any time, even during combat. Discard them once they are played. Exception: You cannot play a Go Up a Level card to give a player the winning level!'
const DEMO_CITATION = '(Munchkin Rules, p. 2)'

// ---------------------------------------------------------------------------
// Hero image — swaps in `public/hero.webp` when present, otherwise renders a
// correctly-sized placeholder so the layout never looks broken.
// ---------------------------------------------------------------------------

function HeroImage() {
  const [ok, setOk] = useState(true)

  return (
    <div className="w-full max-w-xl aspect-video rounded-2xl overflow-hidden border border-border bg-card shadow-sm">
      {ok ? (
        <img
          src="/hero.webp"
          alt="Boardbarian — a board game rules assistant"
          className="w-full h-full object-cover"
          onError={() => setOk(false)}
        />
      ) : (
        <div className="w-full h-full flex flex-col items-center justify-center gap-2 text-muted-foreground">
          <span className="text-4xl">🎲</span>
          <span className="text-xs uppercase tracking-wider">Hero image</span>
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Chat demo — a looping sample conversation. Every part stays mounted so the
// card reserves its full height from the first frame and simply fills in,
// rather than growing as each part appears. The reveal loops by resetting
// `phase` in place. Renders the final state statically under reduced motion.
// ---------------------------------------------------------------------------

function ChatDemo() {
  const reduced = useReducedMotion() ?? false
  // phase 0: question only · 1: + answer · 2: + cited, verified quote
  const [phase, setPhase] = useState(reduced ? 2 : 0)

  useEffect(() => {
    if (reduced) return
    let timers: number[] = []
    const run = () => {
      setPhase(0)
      timers = [
        window.setTimeout(() => setPhase(1), 900),
        window.setTimeout(() => setPhase(2), 2100),
        window.setTimeout(run, 18000),
      ]
    }
    run()
    return () => timers.forEach(window.clearTimeout)
  }, [reduced])

  // Fade a part in once its phase is reached, but keep it mounted the whole
  // time so its space is always reserved and the card never resizes. Under
  // reduced motion every part is forced fully visible with no movement — even
  // if the preference toggles on mid-cycle — so the demo is never left
  // partially hidden.
  const reveal = (shown: boolean) => {
    const show = reduced || shown
    return {
      initial: reduced ? false : { opacity: 0, y: 8 },
      animate: { opacity: show ? 1 : 0, y: show ? 0 : 8 },
      transition: { duration: reduced ? 0 : 0.4 },
    }
  }

  return (
    <div className="rounded-2xl border border-border bg-card p-5 shadow-sm">
      <div className="mb-4 flex items-center gap-2">
        <div className="flex size-7 items-center justify-center rounded-lg bg-[#7f1d1d] text-sm">⚔️</div>
        <span className="text-sm font-medium text-foreground">Munchkin</span>
      </div>

      <div className="flex flex-col gap-4">
        {/* User question */}
        <motion.div {...reveal(true)} className="flex justify-end">
          <div className="max-w-[80%] rounded-2xl bg-muted px-3.5 py-2 text-sm text-foreground">
            {DEMO_QUESTION}
          </div>
        </motion.div>

        {/* Assistant answer */}
        <motion.div
          {...reveal(phase >= 1)}
          aria-hidden={phase < 1}
          className="text-sm text-foreground leading-relaxed"
        >
          {DEMO_ANSWER}
        </motion.div>

        {/* Cited, verified quote */}
        <motion.div {...reveal(phase >= 2)} aria-hidden={phase < 2} className="space-y-2">
          <blockquote className="border-l-2 border-border pl-3 text-sm text-muted-foreground italic">
            "{DEMO_QUOTE}"
            <div className="mt-1 not-italic text-xs text-muted-foreground/80">{DEMO_CITATION}</div>
          </blockquote>
          <div className="flex items-center gap-1.5 text-xs font-medium text-emerald-400">
            <BadgeCheckIcon className="size-3.5" />
            Verified against the rulebook
          </div>
        </motion.div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// How it works
// ---------------------------------------------------------------------------

const STEPS = [
  {
    icon: DicesIcon,
    title: 'Pick a game',
    body: 'Choose from the games in the library — from Munchkin to Catan to Gloomhaven.',
  },
  {
    icon: MessageCircleQuestionIcon,
    title: 'Ask in plain English',
    body: 'Type your rules question the way you would ask a friend across the table.',
  },
  {
    icon: BadgeCheckIcon,
    title: 'Get a cited answer',
    body: 'Every answer quotes the actual rulebook, so you can trust it and settle the argument.',
  },
]

function HowItWorks() {
  return (
    <section className="grid gap-4 sm:grid-cols-3">
      {STEPS.map(({ icon: Icon, title, body }) => (
        <div key={title} className="rounded-2xl border border-border bg-card p-5">
          <div className="mb-3 flex size-9 items-center justify-center rounded-lg bg-muted text-foreground">
            <Icon className="size-5" />
          </div>
          <h3 className="mb-1 text-sm font-semibold text-foreground">{title}</h3>
          <p className="text-sm text-muted-foreground leading-relaxed">{body}</p>
        </div>
      ))}
    </section>
  )
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function HomePage() {
  const navigate = useNavigate()
  const { user } = useAuth()

  // Always send the primary CTA to the auth-gated /select-game; CacheGate shows
  // the sign-in screen when needed. Routing here rather than branching on `user`
  // avoids misrouting a signed-in user to /login during the brief window where
  // auth state is still resolving (`user` is momentarily null).
  const start = () => navigate('/select-game')

  return (
    <div className="h-full overflow-y-auto bg-background text-foreground">
      <div className="mx-auto max-w-5xl px-4 pb-16">
        {/* Header */}
        <header className="flex items-center justify-between py-4">
          <div className="flex items-center gap-2">
            <span className="text-xl">🎲</span>
            <span className="text-sm font-semibold tracking-wide">Boardbarian</span>
          </div>
          {user ? (
            <Button variant="ghost" size="sm" onClick={() => navigate('/select-game')}>
              Open app
            </Button>
          ) : (
            <Button variant="outline" size="sm" onClick={() => navigate('/login')}>
              Sign in
            </Button>
          )}
        </header>

        {/* Hero */}
        <section className="grid items-center gap-8 py-10 md:grid-cols-2 md:gap-10 md:py-16">
          <div>
            <h1 className="text-3xl font-semibold tracking-tight text-foreground sm:text-4xl md:text-5xl">
              Rules questions, answered instantly.
            </h1>
            <p className="mt-4 max-w-md text-base text-muted-foreground leading-relaxed">
              Boardbarian is an AI assistant that answers board game rules questions — with
              citations straight from the rulebook — so you spend less time flipping pages and
              more time playing.
            </p>
            <div className="mt-7 flex items-center gap-3">
              <Button size="lg" onClick={start}>
                Get started
                <ArrowRightIcon className="size-4" />
              </Button>
              {!user && (
                <Button size="lg" variant="ghost" onClick={() => navigate('/login')}>
                  Sign in
                </Button>
              )}
            </div>
          </div>
          <div className="flex justify-center md:justify-end">
            <HeroImage />
          </div>
        </section>

        {/* Live demo */}
        <section className="py-6">
          <p className="mb-3 text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
            See it in action
          </p>
          <div className="mx-auto max-w-2xl">
            <ChatDemo />
          </div>
        </section>

        {/* How it works */}
        <div className="py-10 md:py-14">
          <HowItWorks />
        </div>

        {/* Footer. The signed-out entry point to the legal documents - this is
            the path Google's OAuth brand review follows from the homepage. */}
        <footer className="flex flex-col items-center gap-3 border-t border-border pt-8 text-xs text-muted-foreground sm:flex-row sm:justify-between">
          <span>© {new Date().getFullYear()} Boardbarian</span>
          {/* Plain anchors, not react-router Links: these are pre-rendered
              static pages outside the SPA, so a client-side navigation would
              find no route and fall through to the 404. */}
          <nav className="flex items-center gap-4">
            <a href="/terms/" className="hover:text-foreground">
              Terms of Use
            </a>
            <a href="/privacy/" className="hover:text-foreground">
              Privacy Policy
            </a>
            <a href="mailto:support@boardbarian.com" className="hover:text-foreground">
              Contact
            </a>
          </nav>
        </footer>
      </div>
    </div>
  )
}
