/**
 * Generate a RFC 4122 v4 UUID.
 *
 * `crypto.randomUUID()` is only available in a **secure context** (HTTPS or
 * localhost). Running the dev server over a plain-HTTP LAN hostname
 * (e.g. http://ubuntu-box.local) is not a secure context, so `randomUUID` is
 * undefined there. `crypto.getRandomValues()` has no such restriction, so we
 * fall back to building a v4 UUID from it — still a CSPRNG, so collisions stay a
 * non-issue.
 */
export function randomUUID(): string {
  const c = globalThis.crypto
  if (typeof c?.randomUUID === 'function') return c.randomUUID()

  const bytes = new Uint8Array(16)
  c.getRandomValues(bytes)
  // Per RFC 4122 §4.4: set the version (4) and variant (10xx) bits.
  bytes[6] = (bytes[6] & 0x0f) | 0x40
  bytes[8] = (bytes[8] & 0x3f) | 0x80
  const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, '0'))
  return (
    hex.slice(0, 4).join('') +
    '-' + hex.slice(4, 6).join('') +
    '-' + hex.slice(6, 8).join('') +
    '-' + hex.slice(8, 10).join('') +
    '-' + hex.slice(10, 16).join('')
  )
}
