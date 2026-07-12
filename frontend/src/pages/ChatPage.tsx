import { useChat } from '@ai-sdk/react'
import { DefaultChatTransport, isReasoningUIPart, isTextUIPart } from 'ai'
import type { UIMessage } from 'ai'
import { useEffect, useRef, useState } from 'react'
import { useLocation, useNavigate, useParams } from 'react-router-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
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
import { UserMenu } from '@/components/UserMenu'
import { useAuthFetch } from '@/auth/authFetch'
import { useAuth } from '@/auth/useAuth'
import { type Game } from '@/data/games'
import { useGame } from '@/hooks/useGame'
import { useChats } from '@/hooks/useChats'
import { cn } from '@/lib/utils'
import { LoadingLabel } from '@/components/LoadingLabel'
import { fetchWithRetry } from '@/lib/fetchWithRetry'
import {
  CheckIcon,
  CopyIcon,
  MenuIcon,
  RefreshCwIcon,
  SendIcon,
  ThumbsDownIcon,
  ThumbsUpIcon,
  XIcon,
} from 'lucide-react'

// ---------------------------------------------------------------------------
// Rate limit error helpers
// ---------------------------------------------------------------------------

interface RateLimitDetail {
  window: string
  resets_at: string
  message: string
}

function parseRateLimitDetail(error: Error): RateLimitDetail | null {
  try {
    const body = JSON.parse(error.message)
    const detail = body?.detail
    if (detail?.error === 'rate_limit_exceeded') return detail as RateLimitDetail
  } catch {
    // Not a rate-limit error (message wasn't the expected JSON) -- fall through.
  }
  return null
}

function formatResetTime(isoTimestamp: string): string {
  const resetAt = new Date(isoTimestamp)
  const roundedMs = Math.ceil(resetAt.getTime() / 3_600_000) * 3_600_000
  const rounded = new Date(roundedMs)
  return rounded.toLocaleString(undefined, {
    weekday: 'long',
    month: 'long',
    day: 'numeric',
    hour: 'numeric',
    hour12: true,
  })
}

// ---------------------------------------------------------------------------
// Sidebar
// ---------------------------------------------------------------------------

interface SidebarProps {
  open: boolean
  onClose: () => void
  gameId: string
  currentChatId?: string
}

function Sidebar({ open, onClose, gameId, currentChatId }: SidebarProps) {
  const navigate = useNavigate()
  const { chats, hasNextPage, loadMore, isFetching, isPending, isFetchingMore } = useChats(gameId, open)
  const sentinelRef = useRef<HTMLDivElement>(null)

  // Load the next page when the sentinel scrolls into view.
  useEffect(() => {
    const el = sentinelRef.current
    if (!el || !hasNextPage || isFetching) return
    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) loadMore()
    })
    observer.observe(el)
    return () => observer.disconnect()
  }, [hasNextPage, isFetching, loadMore])

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
            {/* Only on first load (no cache) or when paginating — a silent
                background refetch over cached chats shows nothing. */}
            {(isPending || isFetchingMore) && (
              <LoadingLabel className="text-xs text-muted-foreground">Loading…</LoadingLabel>
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
  feedback,
  onFeedback,
  onRegenerate,
}: {
  message: UIMessage
  isStreaming: boolean
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

  const [isCopied, setIsCopied] = useState(false)

  const handleCopy = () => {
    const fallback = () => {
      const el = document.createElement('textarea')
      el.value = combinedText
      document.body.appendChild(el)
      el.select()
      document.execCommand('copy')
      document.body.removeChild(el)
    }

    const finish = () => {
      setIsCopied(true)
      setTimeout(() => setIsCopied(false), 2000)
    }

    if (navigator.clipboard) {
      navigator.clipboard.writeText(combinedText).catch(fallback).finally(finish)
    } else {
      fallback()
      finish()
    }
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
          <MessageAction tooltip={isCopied ? 'Copied!' : 'Copy'} onClick={handleCopy}>
            {isCopied ? <CheckIcon className="size-4" /> : <CopyIcon className="size-4" />}
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
            disabled={isStreaming}
          >
            <RefreshCwIcon className="size-4" />
          </MessageAction>
        </MessageActions>
      )}
    </Message>
  )
}

// ---------------------------------------------------------------------------
// Chat error message
// ---------------------------------------------------------------------------

function ChatError({ error, onDismiss }: { error: Error; onDismiss: () => void }) {
  const detail = parseRateLimitDetail(error)

  if (detail) {
    const resetLabel = formatResetTime(detail.resets_at)
    const isAppWide = detail.message.startsWith('The service')
    return (
      <Message from="assistant">
        <MessageContent>
          <div className="rounded-lg border border-amber-200 bg-amber-50 dark:border-amber-800 dark:bg-amber-950/30 px-4 py-3 text-sm text-amber-900 dark:text-amber-200 space-y-2">
            <p className="font-medium">
              {isAppWide
                ? 'The service is temporarily at capacity.'
                : `You've reached your ${detail.window} usage limit.`}
            </p>
            <p className="text-amber-700 dark:text-amber-400">
              You can try again on {resetLabel}.
            </p>
            <div className="pt-1">
              <Button size="sm" variant="ghost" className="text-amber-700 dark:text-amber-400 hover:text-amber-900 dark:hover:text-amber-200 -ml-2" onClick={onDismiss}>
                Dismiss
              </Button>
            </div>
          </div>
        </MessageContent>
      </Message>
    )
  }

  return (
    <Message from="assistant">
      <MessageContent>
        <div className="rounded-lg border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive space-y-2">
          <p className="font-medium">Something went wrong. Please try again.</p>
          <div className="pt-1">
            <Button size="sm" variant="ghost" className="text-destructive -ml-2" onClick={onDismiss}>
              Dismiss
            </Button>
          </div>
        </div>
      </MessageContent>
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
  const defaultSuggestions = ['What are the basic rules?', 'How does setup work?', 'What is the win condition?']
  const qs = game.exampleQuestions?.length ? game.exampleQuestions : defaultSuggestions

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

          <UserMenu className="w-9 h-9" />
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
  const qc = useQueryClient()

  const createChat = useMutation({
    mutationFn: (text: string) =>
      authFetch(`/api/games/${gameId}/chats`, { method: 'POST' })
        .then((r) => {
          if (!r.ok) throw new Error('Failed to create chat')
          return r.json() as Promise<{ chat_id: string }>
        })
        .then((body) => ({ ...body, text })),
    onSuccess: ({ chat_id, text }) => {
      // Invalidate the chat list so the sidebar is fresh when next opened.
      qc.invalidateQueries({ queryKey: ['chats', gameId] })
      navigate(`/chat/${gameId}/${chat_id}`, { state: { pendingMessage: text } })
    },
  })

  const handleSubmit = (text: string) => createChat.mutate(text)

  return (
    <PageChrome game={game} gameId={gameId}>
      <div className="flex-1 min-h-0 overflow-y-auto">
        <EmptyState game={game} onSuggest={handleSubmit} />
      </div>
      <div className="shrink-0">
        <ChatInput onSubmit={handleSubmit} disabled={createChat.isPending} />
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

  const { data: initialMessages, isPending } = useQuery({
    queryKey: ['messages', chatId],
    queryFn: ({ signal }) =>
      authFetch(`/api/chats/${chatId}/messages`, { signal })
        .then((r) => { if (!r.ok) throw new Error(`${r.status}`); return r.json() as Promise<UIMessage[]> })
        .catch(() => [] as UIMessage[]),
  })

  if (isPending) {
    // Only shown on first visit — cached chats render immediately.
    return (
      <PageChrome game={game} gameId={gameId} chatId={chatId}>
        <div className="flex-1 min-h-0 flex items-center justify-center">
          <LoadingLabel className="text-sm text-muted-foreground">Loading…</LoadingLabel>
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
      initialMessages={initialMessages ?? []}
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

  const { messages: chatMessages, sendMessage, regenerate, status, error, clearError, setMessages } = useChat({
    messages: initialMessages,
    transport: new DefaultChatTransport({
      api: `${import.meta.env.VITE_API_URL ?? ''}/api/chats/${chatId}/stream`,
      // Retry through Cloud Run cold starts. Retries only fire before any stream
      // bytes are written (network abort / non-JSON 5xx), so an in-progress
      // generation is never re-sent; JSON 429 (rate limit) and JSON 5xx (app
      // error) pass straight through to the error handling below.
      fetch: (input, init) => fetchWithRetry(fetch, input, init),
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

  const handleDismiss = () => {
    const last = chatMessages[chatMessages.length - 1]
    if (last?.role === 'user') setMessages(chatMessages.slice(0, -1))
    clearError()
  }

  const handleSubmit = (text: string) => {
    if (error) {
      const last = chatMessages[chatMessages.length - 1]
      if (last?.role === 'user') setMessages(chatMessages.slice(0, -1))
      clearError()
    }
    sendMessage({ text })
  }

  return (
    <PageChrome game={game} gameId={gameId} chatId={chatId}>
      <div className="flex-1 min-h-0 overflow-y-auto">
        {displayMessages.length === 0 && !error ? (
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
                  feedback={feedbackMap[msg.id] ?? null}
                  onFeedback={(v) => setFeedbackMap((prev) => ({ ...prev, [msg.id]: v }))}
                  onRegenerate={() => regenerate({ messageId: msg.id })}
                />
              ),
            )}
            {error && <ChatError error={error} onDismiss={handleDismiss} />}
            <div ref={bottomRef} />
          </div>
        )}
      </div>

      <div className="shrink-0">
        <ChatInput onSubmit={handleSubmit} disabled={isStreaming} />
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
        {error ? (
          <span className="text-sm text-muted-foreground">{error}</span>
        ) : (
          <LoadingLabel className="text-sm text-muted-foreground">Loading…</LoadingLabel>
        )}
      </div>
    )
  }

  if (!chatId) {
    return <NewChat gameId={gameId!} game={game} />
  }

  return <ExistingChat key={chatId} gameId={gameId!} chatId={chatId} game={game} />
}
