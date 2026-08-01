/**
 * AuthProvider
 *
 * Switches between two implementations based on env vars:
 *
 *   VITE_AUTH_BYPASS=true     →  MockAuthProvider  (no Firebase, no login redirect)
 *   VITE_FIREBASE_EMULATOR=true  →  FirebaseAuthProvider connected to local emulator
 *   (default)                 →  FirebaseAuthProvider connected to real Firebase
 *
 * Both expose the same AuthContext shape so the rest of the app is unaware
 * of which mode is active.
 */

import { useEffect, useRef, useState, type ReactNode } from 'react'
import { initializeApp, type FirebaseApp } from 'firebase/app'
import {
  connectAuthEmulator,
  getAuth,
  GoogleAuthProvider,
  onAuthStateChanged,
  signInWithEmailAndPassword,
  signInWithPopup,
  signOut,
  type User,
} from 'firebase/auth'
import { AuthContext, type AuthUser } from './AuthContext'
import { clearUserCache } from '@/lib/userCache'

const EMULATOR_MODE = import.meta.env.VITE_FIREBASE_EMULATOR === 'true'
// If no explicit host is configured, use the same hostname the browser is using.
// This means localhost:5173 → localhost:9099, ubuntu-box.local:5173 → ubuntu-box.local:9099, etc.
const EMULATOR_HOST = import.meta.env.VITE_FIREBASE_EMULATOR_HOST ?? `${window.location.hostname}:9099`
// Allow VITE_FIREBASE_EMULATOR_HOST to optionally include a scheme (e.g. for an
// HTTPS-terminating proxy in front of the emulator); default to http:// otherwise.
const EMULATOR_URL = /^https?:\/\//.test(EMULATOR_HOST) ? EMULATOR_HOST : `http://${EMULATOR_HOST}`

// ---------------------------------------------------------------------------
// Firebase config — these values are NOT secrets; safe to include in client code.
// ---------------------------------------------------------------------------

const firebaseConfig = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
}

// ---------------------------------------------------------------------------
// Real Firebase provider
// ---------------------------------------------------------------------------

function FirebaseAuthProvider({ children }: { children: ReactNode }) {
  const appRef = useRef<FirebaseApp | null>(null)
  const [user, setUser] = useState<AuthUser | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [loginError, setLoginError] = useState<string | null>(null)

  if (!appRef.current) {
    appRef.current = initializeApp(firebaseConfig)
    if (EMULATOR_MODE) {
      connectAuthEmulator(getAuth(appRef.current), EMULATOR_URL, { disableWarnings: true })
    }
  }
  const auth = getAuth(appRef.current)

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (firebaseUser: User | null) => {
      if (firebaseUser) {
        setUser({
          uid: firebaseUser.uid,
          email: firebaseUser.email,
          name: firebaseUser.displayName,
          photoURL: firebaseUser.photoURL,
        })
      } else {
        setUser(null)
      }
      setIsLoading(false)
    })
    return unsubscribe
  }, [auth])

  const getIdToken = () => {
    const firebaseUser = auth.currentUser
    if (!firebaseUser) return Promise.reject(new Error('Not authenticated'))
    return firebaseUser.getIdToken()
  }

  const login = (credentials?: { email: string; password: string }) => {
    if (EMULATOR_MODE && credentials) {
      return signInWithEmailAndPassword(auth, credentials.email, credentials.password)
        .then(() => {})
        .catch((err: { code?: string }) => {
          setLoginError('Sign-in failed. Check your credentials.')
          console.error(err)
        })
    }
    return signInWithPopup(auth, new GoogleAuthProvider()).then(() => {}).catch((err: { code?: string }) => {
      if (err?.code === 'auth/admin-restricted-operation') {
        setLoginError('Sign-up is currently disabled. Contact the app administrator.')
      } else if (err?.code && err.code !== 'auth/popup-closed-by-user' && err.code !== 'auth/cancelled-popup-request') {
        setLoginError('Sign-in failed. Please try again.')
      }
    })
  }

  const logout = () => {
    // Clear cached data on explicit logout - privacy at rest, and the next user
    // starts clean. (A lapsed session is handled by reconcile, not here.)
    // Fire-and-forget: the async IndexedDB delete needn't block sign-out.
    void clearUserCache().catch((err) => console.error('cache clear failed', err))
    signOut(auth).catch(console.error)
  }

  return (
    <AuthContext.Provider value={{ user, isLoading, loginError, emulatorMode: EMULATOR_MODE, getIdToken, login, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

// ---------------------------------------------------------------------------
// Bypass (mock) provider — no Firebase required
// ---------------------------------------------------------------------------

function MockAuthProvider({ children }: { children: ReactNode }) {
  const raw = import.meta.env.VITE_AUTH_BYPASS_USER ?? '{}'
  const parsed = JSON.parse(raw) as Partial<AuthUser>
  const user: AuthUser = {
    uid: parsed.uid ?? 'local-dev',
    email: parsed.email ?? 'dev@local',
    name: parsed.name ?? 'Dev User',
    photoURL: parsed.photoURL ?? null,
  }

  return (
    <AuthContext.Provider
      value={{
        user,
        isLoading: false,
        loginError: null,
        emulatorMode: false,
        getIdToken: () => Promise.resolve('bypass-token'),
        login: () => Promise.resolve(),
        logout: () => {},
      }}
    >
      {children}
    </AuthContext.Provider>
  )
}

// ---------------------------------------------------------------------------
// Exported provider — picks the right implementation
// ---------------------------------------------------------------------------

export function AuthProvider({ children }: { children: ReactNode }) {
  if (import.meta.env.VITE_AUTH_BYPASS === 'true') {
    return <MockAuthProvider>{children}</MockAuthProvider>
  }
  return <FirebaseAuthProvider>{children}</FirebaseAuthProvider>
}
