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
  /** When true, the app is connected to the Firebase Auth Emulator. */
  emulatorMode: boolean
  /** In emulator mode, pass { email, password } credentials. In production, redirects to Google OAuth. */
  login: (credentials?: { email: string; password: string }) => Promise<void>
  logout: () => void
}

export const AuthContext = createContext<AuthContextValue>({
  user: null,
  isLoading: true,
  loginError: null,
  emulatorMode: false,
  getIdToken: () => Promise.reject(new Error('AuthContext not initialised')),
  login: () => Promise.resolve(),
  logout: () => {},
})
