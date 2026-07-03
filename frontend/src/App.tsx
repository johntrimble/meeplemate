import { Routes, Route } from 'react-router-dom'
import { PersistQueryClientProvider } from '@tanstack/react-query-persist-client'
import { AuthProvider } from '@/auth/AuthProvider'
import { TooltipProvider } from '@/components/ui/tooltip'
import { ProtectedRoute } from '@/auth/ProtectedRoute'
import { queryClient, localStoragePersister } from '@/lib/queryClient'
import { ColdStartBanner } from '@/components/ColdStartBanner'
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
      persistOptions={{ persister: localStoragePersister }}
    >
    <ColdStartBanner />
    <Routes>
      <Route path="/maintenance" element={<MaintenancePage />} />
      <Route path="*" element={
        <AuthProvider>
          <TooltipProvider>
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/login" element={<LoginPage />} />
              <Route path="/select-game" element={<SelectGamePage />} />
              <Route path="/chat/:gameId" element={<ProtectedRoute><ChatPage /></ProtectedRoute>} />
              <Route path="/chat/:gameId/:chatId" element={<ProtectedRoute><ChatPage /></ProtectedRoute>} />
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
