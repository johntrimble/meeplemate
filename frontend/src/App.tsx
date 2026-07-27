import { Routes, Route } from 'react-router-dom'
import { PersistQueryClientProvider } from '@tanstack/react-query-persist-client'
import { AuthProvider } from '@/auth/AuthProvider'
import { TooltipProvider } from '@/components/ui/tooltip'
import { CacheGate } from '@/auth/CacheGate'
import { queryClient, persister } from '@/lib/queryClient'
import { CACHE_SCHEMA_VERSION, RETENTION_MS, makeShouldDehydrateQuery } from '@/lib/cachePersist'
import { seedGamesFromStatic } from '@/lib/seedGames'
import HomePage from './pages/HomePage'
import LoginPage from './pages/LoginPage'
import SelectGamePage from './pages/SelectGamePage'
import ChatPage from './pages/ChatPage'
import NotFoundPage from './pages/NotFoundPage'
import NotAuthorizedPage from './pages/NotAuthorizedPage'
import MaintenancePage from './pages/MaintenancePage'

function App() {
  return (
    <PersistQueryClientProvider
      client={queryClient}
      persistOptions={{
        persister,
        maxAge: RETENTION_MS,
        buster: CACHE_SCHEMA_VERSION,
        dehydrateOptions: { shouldDehydrateQuery: makeShouldDehydrateQuery(queryClient) },
      }}
      // Once the persisted cache has restored, seed the game catalog from the
      // static CDN snapshot if it's still cold (new user). Runs after restore so
      // it never clobbers a returning user's persisted list. See seedGames.ts.
      onSuccess={() => {
        void seedGamesFromStatic(queryClient)
      }}
    >
    <Routes>
      <Route path="/maintenance" element={<MaintenancePage />} />
      <Route path="*" element={
        <AuthProvider>
          <TooltipProvider>
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/login" element={<LoginPage />} />
              {/* No /terms or /privacy route: those are pre-rendered static
                  HTML served straight by the host, never routed through this
                  app. See scripts/legal-pages.ts and docs/legal.md. */}
              <Route path="/select-game" element={<CacheGate><SelectGamePage /></CacheGate>} />
              <Route path="/chat/:gameId" element={<CacheGate><ChatPage /></CacheGate>} />
              <Route path="/chat/:gameId/:chatId" element={<CacheGate><ChatPage /></CacheGate>} />
              <Route path="/not-authorized" element={<NotAuthorizedPage />} />
              <Route path="*" element={<NotFoundPage />} />
            </Routes>
          </TooltipProvider>
        </AuthProvider>
      } />
    </Routes>
    </PersistQueryClientProvider>
  )
}

export default App
