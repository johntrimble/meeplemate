import { useEffect, useRef, useState } from 'react'
import { useLocation, useNavigate, useParams } from 'react-router-dom'
import {
  ChainOfThought,
  ChainOfThoughtContent,
  ChainOfThoughtHeader,
  ChainOfThoughtStep,
} from '@/components/ai-elements/chain-of-thought'
import {
  Message,
  MessageAction,
  MessageActions,
  MessageContent,
  MessageResponse,
} from '@/components/ai-elements/message'
import { Button } from '@/components/ui/button'
import { GAMES, type Game } from '@/data/games'
import {
  MOCK_CONVERSATIONS,
  type ConversationMessage,
} from '@/data/conversations'
import { MOCK_USER } from '@/data/user'
import { cn } from '@/lib/utils'
import {
  CopyIcon,
  MenuIcon,
  RefreshCwIcon,
  SendIcon,
  ThumbsDownIcon,
  ThumbsUpIcon,
  XIcon,
} from 'lucide-react'

// ---------------------------------------------------------------------------
// Sidebar
// ---------------------------------------------------------------------------

interface SidebarProps {
  open: boolean
  onClose: () => void
  currentConvId?: string
  gameId?: string
}

function Sidebar({ open, onClose, currentConvId, gameId }: SidebarProps) {
  const navigate = useNavigate()

  const go = (path: string, state?: unknown) => {
    navigate(path, state ? { state } : undefined)
    onClose()
  }

  return (
    <>
      {/* Backdrop */}
      <div
        className={cn(
          'fixed inset-0 bg-black/50 z-40 transition-opacity duration-200',
          open ? 'opacity-100' : 'opacity-0 pointer-events-none',
        )}
        onClick={onClose}
      />

      {/* Drawer */}
      <div
        className={cn(
          'fixed inset-y-0 left-0 w-72 bg-background border-r border-border z-50 flex flex-col transition-transform duration-200',
          open ? 'translate-x-0' : '-translate-x-full',
        )}
      >
        {/* Header row */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-border">
          <span className="text-sm font-semibold text-foreground">Menu</span>
          <Button variant="ghost" size="icon-sm" onClick={onClose} aria-label="Close menu">
            <XIcon className="size-4" />
          </Button>
        </div>

        {/* Primary nav */}
        <div className="flex flex-col gap-1 p-2 border-b border-border">
          <Button
            variant="ghost"
            className="justify-start font-normal"
            onClick={() => go('/select-game')}
          >
            Select game
          </Button>
          <Button
            variant="ghost"
            className="justify-start font-normal"
            onClick={() => go('/chat', { gameId })}
          >
            New Chat
          </Button>
        </div>

        {/* Chat history */}
        <div className="flex-1 overflow-y-auto p-2">
          <p className="px-2 py-1.5 text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
            Your chats
          </p>
          {MOCK_CONVERSATIONS.map((conv) => (
            <Button
              key={conv.id}
              variant={currentConvId === conv.id ? 'secondary' : 'ghost'}
              className="w-full justify-start font-normal text-sm truncate"
              onClick={() => go(`/chat/${conv.id}`)}
            >
              {conv.title}
            </Button>
          ))}
        </div>
      </div>
    </>
  )
}

// ---------------------------------------------------------------------------
// Message renderers
// ---------------------------------------------------------------------------

function UserMsg({ message }: { message: ConversationMessage }) {
  return (
    <Message from="user">
      <MessageContent>{message.content}</MessageContent>
    </Message>
  )
}

function AssistantMsg({ message }: { message: ConversationMessage }) {
  const handleCopy = () => {
    navigator.clipboard.writeText(message.content).catch(() => {})
  }

  return (
    <Message from="assistant">
      {message.thinkingSteps && message.thinkingSteps.length > 0 && (
        <ChainOfThought>
          <ChainOfThoughtHeader>
            {message.thinkingDuration
              ? `Thought for ${message.thinkingDuration}s`
              : 'Thought'}
          </ChainOfThoughtHeader>
          <ChainOfThoughtContent>
            {message.thinkingSteps.map((step, i) => (
              <ChainOfThoughtStep key={i} label={step.label} />
            ))}
          </ChainOfThoughtContent>
        </ChainOfThought>
      )}

      <MessageContent>
        <MessageResponse>{message.content}</MessageResponse>
      </MessageContent>

      <MessageActions>
        <MessageAction tooltip="Copy" onClick={handleCopy}>
          <CopyIcon className="size-4" />
        </MessageAction>
        <MessageAction tooltip="Good response">
          <ThumbsUpIcon className="size-4" />
        </MessageAction>
        <MessageAction tooltip="Bad response">
          <ThumbsDownIcon className="size-4" />
        </MessageAction>
        <MessageAction tooltip="Regenerate">
          <RefreshCwIcon className="size-4" />
        </MessageAction>
      </MessageActions>
    </Message>
  )
}

// ---------------------------------------------------------------------------
// Chat input
// ---------------------------------------------------------------------------

function ChatInput({ onSubmit }: { onSubmit: (text: string) => void }) {
  const [value, setValue] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const submit = () => {
    const text = value.trim()
    if (!text) return
    onSubmit(text)
    setValue('')
    // reset height
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  // Auto-grow textarea
  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setValue(e.target.value)
    const el = e.target
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 128)}px`
  }

  return (
    <div className="border-t border-border bg-background/95 backdrop-blur px-4 py-3">
      <div className="flex items-end gap-2 bg-muted rounded-xl px-3 py-2">
        <textarea
          ref={textareaRef}
          className="flex-1 resize-none bg-transparent text-sm text-foreground placeholder:text-muted-foreground outline-none min-h-[24px] max-h-32 leading-relaxed"
          placeholder="Ask anything"
          value={value}
          rows={1}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
        />
        <Button
          type="button"
          size="icon-sm"
          disabled={!value.trim()}
          onClick={submit}
          aria-label="Send"
        >
          <SendIcon className="size-4" />
        </Button>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Empty state (new chat)
// ---------------------------------------------------------------------------

function EmptyState({
  game,
  onSuggest,
}: {
  game: Game
  onSuggest: (q: string) => void
}) {
  const suggestions: Record<string, string[]> = {
    munchkin: [
      'Can I discard my class card during combat?',
      'What happens when two players want to help with a fight?',
      'How do I loot the room?',
    ],
    warhammer_5th_edition: [
      'Do Grail Knights need to take break tests?',
      'What is the flying high rule?',
      'How does magic work?',
    ],
  }

  const defaultSuggestions = ['What are the basic rules?', 'How does setup work?', 'What is the win condition?']
  const qs = suggestions[game.id] ?? defaultSuggestions

  return (
    <div className="flex flex-col items-center px-4 pt-10 pb-4">
      {/* Game card */}
      <div
        className="w-24 h-24 rounded-2xl flex items-center justify-center mb-4 shadow-md"
        style={{ backgroundColor: game.bgColor }}
      >
        <span className="text-4xl">{game.emoji}</span>
      </div>

      <p className="text-base font-medium text-foreground text-center mb-6">
        What can I tell you about {game.name}?
      </p>

      {/* Suggested questions */}
      <div className="w-full space-y-2">
        <p className="text-[11px] font-medium text-muted-foreground uppercase tracking-wider mb-2">
          Suggested
        </p>
        {qs.map((q) => (
          <button
            key={q}
            className="w-full text-left text-sm text-muted-foreground bg-muted/50 hover:bg-muted rounded-lg px-3 py-2.5 transition-colors"
            onClick={() => onSuggest(q)}
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// ChatPage
// ---------------------------------------------------------------------------

export default function ChatPage() {
  const { id } = useParams<{ id?: string }>()
  const location = useLocation()
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  const locationState = location.state as { gameId?: string } | null

  // Resolve conversation and game
  const conversation = id ? MOCK_CONVERSATIONS.find((c) => c.id === id) : null
  const gameId = conversation?.gameId ?? locationState?.gameId
  const game = GAMES.find((g) => g.id === gameId) ?? GAMES[0]

  // Local messages (initialised from the mock conversation, or empty)
  const [messages, setMessages] = useState<ConversationMessage[]>(
    conversation?.messages ?? [],
  )

  // Re-initialise messages when navigating between conversations
  useEffect(() => {
    setMessages(conversation?.messages ?? [])
  }, [id, conversation])

  // Scroll to bottom when messages grow
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSubmit = (text: string) => {
    setMessages((prev) => [
      ...prev,
      { id: `msg-${Date.now()}`, role: 'user', content: text },
    ])
  }

  return (
    <div className="min-h-screen bg-background flex flex-col max-w-lg mx-auto">
      <Sidebar
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        currentConvId={id}
        gameId={gameId}
      />

      {/* ── Header ─────────────────────────────────────────────────── */}
      <header className="sticky top-0 z-30 bg-background flex items-center gap-3 px-4 pt-4 pb-3 border-b border-border">
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={() => setSidebarOpen(true)}
          aria-label="Open menu"
        >
          <MenuIcon className="size-5" />
        </Button>

        {/* Compact game card */}
        <div
          className="w-9 h-9 rounded-lg flex items-center justify-center shrink-0"
          style={{ backgroundColor: game.bgColor }}
        >
          <span className="text-lg">{game.emoji}</span>
        </div>

        <span className="flex-1 text-sm font-medium text-foreground truncate">
          {game.name}
        </span>

        {/* User avatar */}
        <div className="w-8 h-8 rounded-full bg-muted flex items-center justify-center shrink-0">
          <span className="text-xs font-semibold text-muted-foreground">
            {MOCK_USER.initials}
          </span>
        </div>
      </header>

      {/* ── Scrollable content ─────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto">
        {messages.length === 0 ? (
          <EmptyState game={game} onSuggest={handleSubmit} />
        ) : (
          <div className="flex flex-col gap-6 px-4 py-6">
            {messages.map((msg) =>
              msg.role === 'user' ? (
                <UserMsg key={msg.id} message={msg} />
              ) : (
                <AssistantMsg key={msg.id} message={msg} />
              ),
            )}
            <div ref={bottomRef} />
          </div>
        )}
      </div>

      {/* ── Input ──────────────────────────────────────────────────── */}
      <div className="sticky bottom-0 z-10">
        <ChatInput onSubmit={handleSubmit} />
      </div>
    </div>
  )
}
