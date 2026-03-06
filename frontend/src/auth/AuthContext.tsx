import { createContext } from 'react'

export interface AuthUser {
  uid: string
  email: string | null
  name: string | null
  photoURL: string | null
}

export interface AuthContextValue {
  user: AuthUser | null
  isLoading: boolean
  /** Set when a redirect sign-in attempt fails (e.g. sign-up disabled). */
  loginError: string | null
  /** Returns a valid ID token, refreshing silently if needed. */
  getIdToken: () => Promise<string>
  login: () => Promise<void>
  logout: () => void
}

export const AuthContext = createContext<AuthContextValue>({
  user: null,
  isLoading: true,
  loginError: null,
  getIdToken: () => Promise.reject(new Error('AuthContext not initialised')),
  login: () => Promise.resolve(),
  logout: () => {},
})
