import { useChat } from '@ai-sdk/react'
import { DefaultChatTransport, isReasoningUIPart, isTextUIPart } from 'ai'
import type { UIMessage } from 'ai'
import { useCallback, useEffect, useRef, useState } from 'react'
import { useLocation, useNavigate, useParams } from 'react-router-dom'
import {
  Reasoning,
  ReasoningContent,
  ReasoningTrigger,
} from '@/components/ai-elements/reasoning'
import {
  Message,
  MessageAction,
  MessageActions,
  MessageContent,
  MessageResponse,
} from '@/components/ai-elements/message'
import { Button } from '@/components/ui/button'
import { useAuthFetch } from '@/auth/authFetch'
import { useAuth } from '@/auth/useAuth'
import { type Game } from '@/data/games'
import { useGame } from '@/hooks/useGame'
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

interface ChatSummary {
  chat_id: string
  title: string
}

interface SidebarProps {
  open: boolean
  onClose: () => void
  gameId: string
  currentChatId?: string
}

interface ChatsPage {
  pageInfo: { hasNextPage: boolean; endCursor: string | null }
  data: ChatSummary[]
}

function Sidebar({ open, onClose, gameId, currentChatId }: SidebarProps) {
  const navigate = useNavigate()
  const authFetch = useAuthFetch()
  const [chats, setChats] = useState<ChatSummary[]>([])
  const [endCursor, setEndCursor] = useState<string | null>(null)
  const [hasNextPage, setHasNextPage] = useState(false)
  const [loading, setLoading] = useState(false)
  const sentinelRef = useRef<HTMLDivElement>(null)

  const loadChats = useCallback((cursor: string | null) => {
    setLoading(true)
    const params = new URLSearchParams({ first: '20' })
    if (cursor) params.set('cursor', cursor)
    authFetch(`/api/games/${gameId}/chats?${params}`)
      .then((r) => r.json())
      .then((page: ChatsPage) => {
        setChats((prev) => cursor ? [...prev, ...page.data] : page.data)
        setHasNextPage(page.pageInfo.hasNextPage)
        setEndCursor(page.pageInfo.endCursor ?? null)
      })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [gameId, authFetch])

  // Reset and load first page when the sidebar opens or game changes.
  useEffect(() => {
    if (!open) return
    setChats([])
    setEndCursor(null)
    setHasNextPage(false)
    loadChats(null)
  }, [open, gameId, loadChats])

  // Load the next page when the sentinel scrolls into view.
  useEffect(() => {
    const el = sentinelRef.current
    if (!el || !hasNextPage || loading) return
    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) loadChats(endCursor)
    })
    observer.observe(el)
    return () => observer.disconnect()
  }, [hasNextPage, loading, endCursor, loadChats])

  const go = (path: string) => {
    navigate(path)
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
            onClick={() => go(`/chat/${gameId}`)}
          >
            New Chat
          </Button>
        </div>

        {/* Chat history for this game */}
        <div className="flex-1 overflow-y-auto p-2">
          {chats.length > 0 && (
            <p className="px-2 py-1.5 text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
              Your chats
            </p>
          )}
          {chats.map((chat) => (
            <Button
              key={chat.chat_id}
              variant={currentChatId === chat.chat_id ? 'secondary' : 'ghost'}
              className="w-full justify-start font-normal text-sm truncate"
              onClick={() => go(`/chat/${gameId}/${chat.chat_id}`)}
            >
              {chat.title}
            </Button>
          ))}
          <div ref={sentinelRef} className="py-1 flex justify-center">
            {loading && (
              <span className="text-xs text-muted-foreground">Loading…</span>
            )}
          </div>
        </div>
      </div>
    </>
  )
}

// ---------------------------------------------------------------------------
// Message renderers
// ---------------------------------------------------------------------------

function UserMsg({ message }: { message: UIMessage }) {
  const text = message.parts.filter(isTextUIPart).map((p) => p.text).join('')
  return (
    <Message from="user">
      <MessageContent>{text}</MessageContent>
    </Message>
  )
}

function AssistantMsg({
  message,
  isStreaming,
  isRetrying,
  feedback,
  onFeedback,
  onRegenerate,
}: {
  message: UIMessage
  isStreaming: boolean
  isRetrying: boolean
  feedback: 0 | 1 | null
  onFeedback: (v: 0 | 1 | null) => void
  onRegenerate: () => void
}) {
  const authFetch = useAuthFetch()
  const reasoningParts = message.parts.filter(isReasoningUIPart)
  const textParts = message.parts.filter(isTextUIPart)
  const combinedReasoning = reasoningParts.map((p) => p.text).join('\n\n')
  const combinedText = textParts.map((p) => p.text).join('')
  const isReasoningStreaming = isStreaming && combinedText.length === 0

  const handleCopy = () => {
    navigator.clipboard.writeText(combinedText).catch(() => {})
  }

  const handleFeedback = async (clicked: 0 | 1) => {
    if (feedback === clicked) {
      await authFetch(`/api/messages/${message.id}/feedback`, { method: 'DELETE' })
      onFeedback(null)
    } else {
      await authFetch(`/api/messages/${message.id}/feedback`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ value: clicked }),
      })
      onFeedback(clicked)
    }
  }

  return (
    <Message from="assistant">
      {reasoningParts.length > 0 && (
        <Reasoning isStreaming={isReasoningStreaming}>
          <ReasoningTrigger />
          <ReasoningContent>{combinedReasoning}</ReasoningContent>
        </Reasoning>
      )}

      {combinedText && (
        <MessageContent>
          <MessageResponse>{combinedText}</MessageResponse>
        </MessageContent>
      )}

      {combinedText && (
        <MessageActions>
          <MessageAction tooltip="Copy" onClick={handleCopy}>
            <CopyIcon className="size-4" />
          </MessageAction>
          <MessageAction
            tooltip="Good response"
            onClick={() => handleFeedback(1)}
            className={feedback === 1 ? 'text-primary' : undefined}
          >
            <ThumbsUpIcon className="size-4" fill={feedback === 1 ? 'currentColor' : 'none'} />
          </MessageAction>
          <MessageAction
            tooltip="Bad response"
            onClick={() => handleFeedback(0)}
            className={feedback === 0 ? 'text-primary' : undefined}
          >
            <ThumbsDownIcon className="size-4" fill={feedback === 0 ? 'currentColor' : 'none'} />
          </MessageAction>
          <MessageAction
            tooltip="Regenerate"
            onClick={onRegenerate}
            disabled={isStreaming || isRetrying}
          >
            <RefreshCwIcon className={cn('size-4', isRetrying && 'animate-spin')} />
          </MessageAction>
        </MessageActions>
      )}
    </Message>
  )
}

// ---------------------------------------------------------------------------
// Chat input
// ---------------------------------------------------------------------------

function ChatInput({
  onSubmit,
  disabled,
}: {
  onSubmit: (text: string) => void
  disabled?: boolean
}) {
  const [value, setValue] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const submit = () => {
    const text = value.trim()
    if (!text || disabled) return
    onSubmit(text)
    setValue('')
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

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setValue(e.target.value)
    const el = e.target
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 128)}px`
  }

  return (
    <div className="border-t border-border bg-background/95 backdrop-blur px-4 py-3">
      <div className="max-w-3xl mx-auto flex items-end gap-2 bg-muted rounded-xl px-3 py-2">
        <textarea
          ref={textareaRef}
          className="flex-1 resize-none bg-transparent text-sm text-foreground placeholder:text-muted-foreground outline-none min-h-[24px] max-h-48 leading-relaxed"
          placeholder="Ask anything"
          value={value}
          rows={1}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
        />
        <Button
          type="button"
          size="icon-sm"
          disabled={!value.trim() || disabled}
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
    <div className="max-w-3xl mx-auto w-full flex flex-col items-center px-4 pt-10 pb-4">
      <div
        className="w-24 h-24 rounded-2xl flex items-center justify-center mb-4 shadow-md"
        style={{ backgroundColor: game.bgColor }}
      >
        <span className="text-4xl">{game.emoji}</span>
      </div>

      <p className="text-base font-medium text-foreground text-center mb-6">
        What can I tell you about {game.name}?
      </p>

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
// Shared page chrome (header + sidebar wrapper)
// ---------------------------------------------------------------------------

function initials(name: string | null, email: string | null): string {
  if (name) return name.split(' ').map((w) => w[0]).join('').slice(0, 2).toUpperCase()
  if (email) return email[0].toUpperCase()
  return '?'
}

function PageChrome({
  game,
  gameId,
  chatId,
  children,
}: {
  game: Game
  gameId: string
  chatId?: string
  children: React.ReactNode
}) {
  const { user } = useAuth()
  const [sidebarOpen, setSidebarOpen] = useState(false)

  return (
    <div className="fixed inset-0 bg-background flex flex-col">
      <Sidebar
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        gameId={gameId}
        currentChatId={chatId}
      />

      <header className="z-30 bg-background border-b border-border shrink-0">
        <div className="max-w-3xl mx-auto flex items-center gap-3 px-4 py-3">
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={() => setSidebarOpen(true)}
            aria-label="Open menu"
          >
            <MenuIcon className="size-5" />
          </Button>

          <div
            className="w-9 h-9 rounded-lg flex items-center justify-center shrink-0"
            style={{ backgroundColor: game.bgColor }}
          >
            <span className="text-lg">{game.emoji}</span>
          </div>

          <span className="flex-1 text-sm font-medium text-foreground truncate">
            {game.name}
          </span>

          <div className="w-8 h-8 rounded-full bg-muted flex items-center justify-center shrink-0">
            <span className="text-xs font-semibold text-muted-foreground">
              {initials(user?.name ?? null, user?.email ?? null)}
            </span>
          </div>
        </div>
      </header>

      {children}
    </div>
  )
}

// ---------------------------------------------------------------------------
// New chat — no chatId yet, creates chat on first message
// ---------------------------------------------------------------------------

function NewChat({ gameId, game }: { gameId: string; game: Game }) {
  const navigate = useNavigate()
  const authFetch = useAuthFetch()
  const [creating, setCreating] = useState(false)

  const handleSubmit = async (text: string) => {
    setCreating(true)
    try {
      const res = await authFetch(`/api/games/${gameId}/chats`, { method: 'POST' })
      if (!res.ok) throw new Error('Failed to create chat')
      const { chat_id } = (await res.json()) as { chat_id: string }
      // Navigate to the permanent URL, carrying the first message as pending state.
      navigate(`/chat/${gameId}/${chat_id}`, { state: { pendingMessage: text } })
    } finally {
      setCreating(false)
    }
  }

  return (
    <PageChrome game={game} gameId={gameId}>
      <div className="flex-1 min-h-0 overflow-y-auto">
        <EmptyState game={game} onSuggest={handleSubmit} />
      </div>
      <div className="shrink-0">
        <ChatInput onSubmit={handleSubmit} disabled={creating} />
      </div>
    </PageChrome>
  )
}

// ---------------------------------------------------------------------------
// Existing chat — loads history, then uses useChat
// ---------------------------------------------------------------------------

function ExistingChat({
  gameId,
  chatId,
  game,
}: {
  gameId: string
  chatId: string
  game: Game
}) {
  const location = useLocation()
  const authFetch = useAuthFetch()
  const [initialMessages, setInitialMessages] = useState<UIMessage[] | null>(null)

  useEffect(() => {
    authFetch(`/api/chats/${chatId}/messages`)
      .then((r) => { if (!r.ok) throw new Error(`${r.status}`); return r.json() })
      .then((msgs: UIMessage[]) => setInitialMessages(msgs))
      .catch(() => setInitialMessages([]))
  // authFetch identity is stable within a session; chatId is the real dep.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chatId])

  if (initialMessages === null) {
    // Loading history — show minimal chrome so the page doesn't flash blank
    return (
      <PageChrome game={game} gameId={gameId} chatId={chatId}>
        <div className="flex-1 min-h-0 flex items-center justify-center">
          <span className="text-sm text-muted-foreground">Loading…</span>
        </div>
      </PageChrome>
    )
  }

  const pendingMessage =
    (location.state as { pendingMessage?: string } | null)?.pendingMessage ?? null

  return (
    <ChatView
      gameId={gameId}
      chatId={chatId}
      game={game}
      initialMessages={initialMessages}
      pendingMessage={pendingMessage}
    />
  )
}

// ---------------------------------------------------------------------------
// ChatView — renders once history is loaded, drives useChat
// ---------------------------------------------------------------------------

function ChatView({
  gameId,
  chatId,
  game,
  initialMessages,
  pendingMessage,
}: {
  gameId: string
  chatId: string
  game: Game
  initialMessages: UIMessage[]
  pendingMessage: string | null
}) {
  const navigate = useNavigate()
  const location = useLocation()
  const { getIdToken } = useAuth()
  const bottomRef = useRef<HTMLDivElement>(null)
  const pendingSent = useRef(false)
  const [feedbackMap, setFeedbackMap] = useState<Record<string, 0 | 1 | null>>(() => {
    const map: Record<string, 0 | 1 | null> = {}
    for (const msg of initialMessages) {
      const fb = (msg as UIMessage & { feedback?: number | null }).feedback
      if (fb === 0 || fb === 1) map[msg.id] = fb
    }
    return map
  })

  const { messages: chatMessages, sendMessage, regenerate, status } = useChat({
    messages: initialMessages,
    transport: new DefaultChatTransport({
      api: `/api/chats/${chatId}/stream`,
      prepareSendMessagesRequest: async ({ messages, trigger, messageId }) => {
        const last = messages[messages.length - 1]
        const text =
          last?.parts
            .filter((p): p is { type: 'text'; text: string } => p.type === 'text')
            .map((p) => p.text)
            .join('') ?? ''
        const token = await getIdToken()
        return {
          body: {
            message: text,
            game_id: gameId,
            ...(trigger === 'regenerate-message' && messageId ? { retry_message_id: messageId } : {}),
          },
          headers: { Authorization: `Bearer ${token}` },
        }
      },
    }),
  })

  // Send the pending message once on mount (carried from NewChat navigation).
  useEffect(() => {
    if (!pendingMessage || pendingSent.current) return
    pendingSent.current = true
    sendMessage({ text: pendingMessage })
    // Clear pending message from location state so a refresh doesn't re-send it.
    navigate(location.pathname, { replace: true, state: {} })
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const isStreaming = status === 'streaming' || status === 'submitted'

  const displayMessages = chatMessages.filter((m) => m.role === 'user' || m.role === 'assistant')

  // Scroll to bottom on new messages (smooth) or streaming content growth (instant).
  const prevLengthRef = useRef(displayMessages.length)
  useEffect(() => {
    const lengthChanged = displayMessages.length !== prevLengthRef.current
    prevLengthRef.current = displayMessages.length
    bottomRef.current?.scrollIntoView({ behavior: lengthChanged ? 'smooth' : 'instant' })
  }, [displayMessages])

  const handleSubmit = (text: string) => {
    sendMessage({ text })
  }

  return (
    <PageChrome game={game} gameId={gameId} chatId={chatId}>
      <div className="flex-1 min-h-0 overflow-y-auto">
        {displayMessages.length === 0 ? (
          <EmptyState game={game} onSuggest={handleSubmit} />
        ) : (
          <div className="max-w-3xl mx-auto flex flex-col gap-6 px-4 py-6">
            {displayMessages.map((msg, i) =>
              msg.role === 'user' ? (
                <UserMsg key={msg.id} message={msg} />
              ) : (
                <AssistantMsg
                  key={msg.id}
                  message={msg as UIMessage}
                  isStreaming={isStreaming && i === displayMessages.length - 1}
                  isRetrying={false}
                  feedback={feedbackMap[msg.id] ?? null}
                  onFeedback={(v) => setFeedbackMap((prev) => ({ ...prev, [msg.id]: v }))}
                  onRegenerate={() => regenerate({ messageId: msg.id })}
                />
              ),
            )}
            <div ref={bottomRef} />
          </div>
        )}
      </div>

      <div className="shrink-0">
        <ChatInput onSubmit={handleSubmit} />
      </div>
    </PageChrome>
  )
}

// ---------------------------------------------------------------------------
// ChatPage — router entry point
// ---------------------------------------------------------------------------

export default function ChatPage() {
  const { gameId, chatId } = useParams<{ gameId: string; chatId?: string }>()
  const { game, isLoading, error } = useGame(gameId!)

  if (isLoading || !game) {
    return (
      <div className="fixed inset-0 bg-background flex items-center justify-center">
        <span className="text-sm text-muted-foreground">
          {error ?? 'Loading…'}
        </span>
      </div>
    )
  }

  if (!chatId) {
    return <NewChat gameId={gameId!} game={game} />
  }

  return <ExistingChat key={chatId} gameId={gameId!} chatId={chatId} game={game} />
}
