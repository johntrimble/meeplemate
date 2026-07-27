import type { ReactNode } from 'react'
import { ExternalLinkIcon } from 'lucide-react'
import { cn } from '@/lib/utils'

/**
 * A link that opens in a new tab, and says so.
 *
 * Opening a new tab without warning is a long-standing accessibility problem
 * (WCAG technique G201): the visual cue people expect is the little
 * arrow-out-of-a-box glyph, but that is invisible to a screen reader, so the
 * icon alone still leaves those users to discover the new tab by finding their
 * back button dead. The `sr-only` text is what actually announces it. Bundling
 * the two here means neither can be added without the other.
 *
 * `rel="noreferrer"` covers `noopener` in every browser that matters, and keeps
 * the opened page from reaching back through `window.opener`.
 */
export function ExternalLink({
  href,
  children,
  className,
  iconClassName,
}: {
  href: string
  children: ReactNode
  className?: string
  iconClassName?: string
}) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noreferrer"
      className={cn('inline-flex items-center gap-1', className)}
    >
      {children}
      <ExternalLinkIcon aria-hidden className={cn('size-3 shrink-0', iconClassName)} />
      <span className="sr-only">(opens in a new tab)</span>
    </a>
  )
}
