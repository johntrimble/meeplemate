import { Routes, Route } from 'react-router-dom'
import { ProtectedRoute } from '@/auth/ProtectedRoute'
import HomePage from './pages/HomePage'
import LoginPage from './pages/LoginPage'
import SelectGamePage from './pages/SelectGamePage'
import ChatPage from './pages/ChatPage'

function App() {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/login" element={<LoginPage />} />
      <Route path="/select-game" element={<SelectGamePage />} />
      <Route path="/chat/:gameId" element={<ProtectedRoute><ChatPage /></ProtectedRoute>} />
      <Route path="/chat/:gameId/:chatId" element={<ProtectedRoute><ChatPage /></ProtectedRoute>} />
    </Routes>
  )
}

export default App
