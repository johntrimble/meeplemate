type BrandMarkProps = {
  /** Tailwind sizing, e.g. `size-9`. */
  className?: string
}

/**
 * The Boardbarian mascot, used wherever the brand mark appears.
 * Decorative — keep nearby visible text (e.g., the app name or a heading) to provide context.
 */
export function BrandMark({ className }: BrandMarkProps) {
  return (
    <img src="/mascot.png" alt="" aria-hidden draggable={false} className={className} />
  )
}
