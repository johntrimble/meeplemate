import type { ReactNode } from 'react'
import { Button } from '@/components/ui/button'
import { useAuth } from './useAuth'

function LoginPage() {
  const { login, loginError } = useAuth()

  return (
    <div className="fixed inset-0 bg-background flex flex-col items-center justify-center gap-6">
      <div className="flex flex-col items-center gap-2 text-center">
        <span className="text-4xl">🎲</span>
        <h1 className="text-xl font-semibold text-foreground">Boardbarian</h1>
        <p className="text-sm text-muted-foreground">Sign in to ask rules questions</p>
      </div>
      <Button onClick={login}>Sign in with Google</Button>
      {loginError && <p className="text-sm text-destructive">{loginError}</p>}
    </div>
  )
}

/**
 * Renders children when the user is authenticated.
 * - While loading: renders nothing (avoids flash of login page).
 * - When unauthenticated: shows LoginPage with a button that triggers
 *   signInWithPopup (Google sign-in popup).
 */
export function ProtectedRoute({ children }: { children: ReactNode }) {
  const { user, isLoading } = useAuth()

  if (isLoading) return null
  if (!user) return <LoginPage />

  return <>{children}</>
}
