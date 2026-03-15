import { useNavigate } from 'react-router-dom'
import { Button } from '@/components/ui/button'

export default function NotAuthorizedPage() {
  const navigate = useNavigate()

  return (
    <div className="fixed inset-0 bg-background flex flex-col items-center justify-center gap-6">
      <div className="flex flex-col items-center gap-2 text-center">
        <span className="text-4xl">🔒</span>
        <h1 className="text-xl font-semibold text-foreground">Access denied</h1>
        <p className="text-sm text-muted-foreground">
          You don't have permission to view this page.
        </p>
      </div>
      <div className="flex gap-2">
        <Button onClick={() => navigate('/login')}>Sign In</Button>
        <Button variant="outline" onClick={() => navigate('/')}>Go Home</Button>
      </div>
    </div>
  )
}
