/** Derive up-to-two-character initials for an avatar from a display name or email. */
export function initials(name: string | null, email: string | null): string {
  if (name) return name.split(' ').map((w) => w[0]).join('').slice(0, 2).toUpperCase()
  if (email) return email[0].toUpperCase()
  return '?'
}
