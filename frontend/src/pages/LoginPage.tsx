import { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '@/auth/useAuth'
import { LoginScreen } from '@/auth/LoginScreen'

export default function LoginPage() {
  const { user, isLoading } = useAuth()
  const navigate = useNavigate()

  useEffect(() => {
    if (!isLoading && user) {
      navigate('/select-game', { replace: true })
    }
  }, [user, isLoading, navigate])

  if (isLoading) return null

  return <LoginScreen />
}
