/** Derive up-to-two-character initials for an avatar from a display name or email. */
export function initials(name: string | null, email: string | null): string {
  const trimmedName = name?.trim()
  if (trimmedName) {
    const letters = trimmedName.split(/\s+/).map((w) => w[0]).join('').slice(0, 2)
    if (letters) return letters.toUpperCase()
  }
  const trimmedEmail = email?.trim()
  if (trimmedEmail) return trimmedEmail[0].toUpperCase()
  return '?'
}
