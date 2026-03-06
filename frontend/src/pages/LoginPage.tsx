import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { useAuth } from '@/auth/useAuth'

export default function LoginPage() {
  const { user, isLoading, login, loginError, emulatorMode } = useAuth()
  const navigate = useNavigate()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')

  useEffect(() => {
    if (!isLoading && user) {
      navigate('/select-game', { replace: true })
    }
  }, [user, isLoading, navigate])

  if (isLoading) return null

  return (
    <div className="fixed inset-0 bg-background flex flex-col items-center justify-center gap-6">
      <div className="flex flex-col items-center gap-2 text-center">
        <span className="text-4xl">🎲</span>
        <h1 className="text-xl font-semibold text-foreground">Boardbarian</h1>
        <p className="text-sm text-muted-foreground">Sign in to ask rules questions</p>
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
        <Button onClick={() => login()}>Sign in with Google</Button>
      )}
      {loginError && <p className="text-sm text-destructive">{loginError}</p>}
    </div>
  )
}
