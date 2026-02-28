import { Routes, Route } from 'react-router-dom'
import HomePage from './pages/HomePage'
import SelectGamePage from './pages/SelectGamePage'
import ChatPage from './pages/ChatPage'

function App() {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/select-game" element={<SelectGamePage />} />
      <Route path="/chat/:gameId" element={<ChatPage />} />
      <Route path="/chat/:gameId/:chatId" element={<ChatPage />} />
    </Routes>
  )
}

export default App
