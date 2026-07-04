import { Routes, Route } from 'react-router-dom'
import { PersistQueryClientProvider } from '@tanstack/react-query-persist-client'
import { AuthProvider } from '@/auth/AuthProvider'
import { TooltipProvider } from '@/components/ui/tooltip'
import { CacheGate } from '@/auth/CacheGate'
import { queryClient, persister } from '@/lib/queryClient'
import { CACHE_SCHEMA_VERSION, RETENTION_MS, makeShouldDehydrateQuery } from '@/lib/cachePersist'
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
    >
    <Routes>
      <Route path="/maintenance" element={<MaintenancePage />} />
      <Route path="*" element={
        <AuthProvider>
          <TooltipProvider>
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/login" element={<LoginPage />} />
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
