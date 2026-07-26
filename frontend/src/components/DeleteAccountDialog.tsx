import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '@/auth/useAuth'
import { useAuthFetch } from '@/auth/authFetch'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Spinner } from '@/components/ui/spinner'

/**
 * Confirmation dialog for deleting the signed-in account.
 *
 * The backend soft-deletes, so signing back in with the same Google account
 * restores everything for a while. The copy says that without naming a
 * duration: the window is a backend constant and the actual erasure is a manual
 * operator job, so a number here would be a promise the UI can't keep.
 *
 * The typed-email confirmation is here because sessions persist indefinitely in
 * localStorage - without it, anyone at an unlocked browser is two clicks from
 * wiping someone's history.
 */
export function DeleteAccountDialog({
  open,
  onOpenChange,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
}) {
  const { user, logout } = useAuth()
  const authFetch = useAuthFetch()
  const navigate = useNavigate()

  const [confirmation, setConfirmation] = useState('')
  const [isDeleting, setIsDeleting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const email = user?.email ?? ''
  const canDelete =
    email.length > 0 && confirmation.trim().toLowerCase() === email.toLowerCase()

  // Reset on close so reopening never starts out pre-confirmed or showing a
  // stale error from a previous attempt.
  useEffect(() => {
    if (!open) {
      setConfirmation('')
      setError(null)
    }
  }, [open])

  const handleDelete = async () => {
    if (!canDelete || isDeleting) return
    setIsDeleting(true)
    setError(null)
    try {
      const res = await authFetch('/api/account', { method: 'DELETE' })
      if (!res.ok) {
        throw new Error(`delete failed with ${res.status}`)
      }
      // logout() clears the cached data as well as signing out.
      logout()
      navigate('/', { replace: true })
    } catch (err) {
      console.error('account deletion failed', err)
      setError("We couldn't delete your account. Please try again.")
      setIsDeleting(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Delete account</DialogTitle>
          <DialogDescription asChild>
            <div className="space-y-2 text-left">
              <p>
                You'll be signed out and your account will be removed.
              </p>
              <p>
                If you change your mind, signing in again with the same Google account
                soon afterwards will restore your chat history.
              </p>
            </div>
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-2">
          <label htmlFor="delete-account-confirm" className="text-sm text-muted-foreground">
            Type <span className="font-medium text-foreground">{email}</span> to confirm:
          </label>
          <Input
            id="delete-account-confirm"
            value={confirmation}
            onChange={(e) => setConfirmation(e.target.value)}
            autoComplete="off"
            autoCapitalize="none"
            spellCheck={false}
            disabled={isDeleting}
            aria-invalid={confirmation.length > 0 && !canDelete}
          />
          {error && (
            <p role="alert" className="text-sm text-destructive">
              {error}
            </p>
          )}
        </div>

        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={isDeleting}
          >
            Cancel
          </Button>
          <Button variant="destructive" onClick={handleDelete} disabled={!canDelete || isDeleting}>
            {isDeleting && <Spinner />}
            Delete account
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
