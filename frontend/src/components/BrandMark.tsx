type BrandMarkProps = {
  /** Tailwind sizing, e.g. `size-9`. */
  className?: string
}

/**
 * The Boardbarian mascot, used wherever the brand mark appears. Decorative —
 * every placement sits next to the wordmark or a heading that names the app.
 */
export function BrandMark({ className }: BrandMarkProps) {
  return (
    <img src="/mascot.png" alt="" aria-hidden draggable={false} className={className} />
  )
}
