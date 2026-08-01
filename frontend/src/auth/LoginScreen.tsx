import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { BrandMark } from '@/components/BrandMark'
import { ExternalLink } from '@/components/ExternalLink'
import { Input } from '@/components/ui/input'
import { useAuth } from './useAuth'

/**
 * Presentational sign-in UI (Google, or email/password against the emulator).
 * Contains no navigation — callers decide what to render around it, so it can
 * be used both as the `/login` route and inline by the auth gate without a
 * post-login redirect.
 */
export function LoginScreen() {
  const { login, loginError, emulatorMode } = useAuth()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')

  return (
    <div className="fixed inset-0 bg-background flex flex-col items-center justify-center gap-6">
      <div className="flex flex-col items-center gap-2 text-center">
        <BrandMark className="size-9" />
        <h1 className="text-xl font-semibold text-foreground">Boardbarian</h1>
        <p className="text-sm text-muted-foreground">Ask rules questions, get cited answers.</p>
      </div>
      {emulatorMode ? (
        <form
          className="flex flex-col gap-3 w-64"
          onSubmit={e => { e.preventDefault(); login({ email, password }) }}
        >
          <Input
            type="email"
            placeholder="Email"
            value={email}
            onChange={e => setEmail(e.target.value)}
            required
          />
          <Input
            type="password"
            placeholder="Password"
            value={password}
            onChange={e => setPassword(e.target.value)}
            required
          />
          <Button type="submit">Sign in (emulator)</Button>
        </form>
      ) : (
        <Button onClick={() => login()}>Continue with Google</Button>
      )}
      {loginError && <p className="text-sm text-destructive">{loginError}</p>}
      {/* Notice only - actual acceptance is collected by ConsentScreen after
          sign-in, since that's the first point we have an account to record it
          against. */}
      <p className="max-w-xs px-6 text-center text-xs text-muted-foreground">
        By signing in you agree to our{' '}
        {/* New tab, like the account menu and the consent screen. This screen
            renders inline from CacheGate at whatever gated URL was requested,
            so navigating away in-tab would discard the very route the user was
            trying to reach. */}
        <ExternalLink href="/terms/" className="underline underline-offset-2">
          Terms of Use
        </ExternalLink>{' '}
        and{' '}
        <ExternalLink href="/privacy/" className="underline underline-offset-2">
          Privacy Policy
        </ExternalLink>
        .
      </p>
    </div>
  )
}
