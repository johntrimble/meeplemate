import { useState } from 'react'
import { useAuth } from './useAuth'
import { useAuthFetch } from './authFetch'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Spinner } from '@/components/ui/spinner'
import { DeleteAccountDialog } from '@/components/DeleteAccountDialog'
import { ExternalLink } from '@/components/ExternalLink'
import { useSlowLoading } from '@/hooks/useSlowLoading'
import { TERMS_VERSION, PRIVACY_VERSION, writeAcceptance } from '@/lib/legal'
import type { AcceptanceState } from '@/lib/legal'

/**
 * Blocking consent screen shown before an account may use the app.
 *
 * A full screen rather than a Dialog, and deliberately non-dismissible: there
 * is no "later" here, so an overlay with an X would only offer a way to appear
 * to decline while still getting in. The ways out are real ones - sign out, or
 * delete the account.
 *
 * Presentational and navigation-free, in the same style as LoginScreen, so the
 * gate can render it inline without a redirect.
 */
export function ConsentScreen({
  state,
  onAccepted,
}: {
  state: Exclude<AcceptanceState, 'current'>
  /** Called once the server has recorded acceptance and localStorage is written. */
  onAccepted: () => void
}) {
  const { user, logout } = useAuth()
  const authFetch = useAuthFetch()

  const [agreed, setAgreed] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [deleteOpen, setDeleteOpen] = useState(false)

  const isReconsent = state === 'outdated'

  const handleAccept = async () => {
    if (!agreed || isSubmitting || !user) return
    setIsSubmitting(true)
    setError(null)
    try {
      const res = await authFetch('/api/account/legal-acceptance', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          terms_version: TERMS_VERSION,
          privacy_version: PRIVACY_VERSION,
        }),
      })
      if (res.status === 409) {
        // This bundle is stale: the documents on the server have moved on, so
        // what the user just read is not what they'd be agreeing to. Reload
        // rather than record it.
        setError('These documents have been updated. Reload the page to continue.')
        setIsSubmitting(false)
        return
      }
      if (!res.ok) throw new Error(`acceptance failed with ${res.status}`)
      // Only now - see writeAcceptance's note on why this must not be optimistic.
      writeAcceptance(user.uid)
      // Stay in the submitting state: `onAccepted` unmounts this screen, and
      // clearing it first would flash an enabled button on the way out.
      onAccepted()
    } catch (err) {
      console.error('legal acceptance failed', err)
      setError("We couldn't save that. Please try again.")
      setIsSubmitting(false)
    }
  }

  return (
    <>
      <div className="fixed inset-0 overflow-y-auto bg-background">
        <div className="mx-auto flex min-h-full max-w-md flex-col justify-center gap-6 px-6 py-10">
          <div className="flex flex-col items-center gap-2 text-center">
            <span className="text-4xl">🎲</span>
            <h1 className="text-xl font-semibold text-foreground">
              {isReconsent ? "We've updated our terms" : 'Before you start'}
            </h1>
            <p className="text-sm text-muted-foreground">
              {isReconsent
                ? 'Please review and accept the updated documents to keep using Boardbarian.'
                : 'Please review these before using Boardbarian.'}
            </p>
          </div>

          <ul className="space-y-2 text-sm text-muted-foreground">
            <li className="flex gap-2">
              <span aria-hidden>•</span>
              <span>
                Boardbarian&apos;s answers are AI-generated and can be wrong. Check
                anything that matters against the rulebook.
              </span>
            </li>
            <li className="flex gap-2">
              <span aria-hidden>•</span>
              <span>
                Your questions and conversations are stored, and are used to test and
                improve the service.
              </span>
            </li>
            <li className="flex gap-2">
              <span aria-hidden>•</span>
              <span>
                Questions are sent to third-party AI providers to generate an answer.
                Don&apos;t share sensitive or private information.
              </span>
            </li>
          </ul>

          <label
            htmlFor="legal-accept"
            className="flex cursor-pointer items-start gap-3 rounded-md border border-border p-3 text-sm text-foreground"
          >
            <Checkbox
              id="legal-accept"
              checked={agreed}
              onCheckedChange={(checked) => setAgreed(checked === true)}
              disabled={isSubmitting}
              className="mt-0.5"
            />
            <span>
              I am at least 13 years old and I agree to the{' '}
              {/* New tab so opening a document doesn't tear down this screen
                  and lose the checkbox state. */}
              <ExternalLink href="/terms/" className="font-medium underline underline-offset-2">
                Terms of Use
              </ExternalLink>{' '}
              and{' '}
              <ExternalLink href="/privacy/" className="font-medium underline underline-offset-2">
                Privacy Policy
              </ExternalLink>
              .
            </span>
          </label>

          {error && (
            <p role="alert" className="text-sm text-destructive">
              {error}
            </p>
          )}

          <div className="flex flex-col gap-2">
            <Button onClick={handleAccept} disabled={!agreed || isSubmitting}>
              {isSubmitting && <Spinner />}
              <AcceptLabel isSubmitting={isSubmitting} />
            </Button>
            <Button variant="ghost" onClick={logout} disabled={isSubmitting}>
              Sign out
            </Button>
          </div>

          <button
            type="button"
            onClick={() => setDeleteOpen(true)}
            disabled={isSubmitting}
            className="text-xs text-muted-foreground underline underline-offset-2 disabled:opacity-50"
          >
            Delete my account instead
          </button>
        </div>
      </div>

      <DeleteAccountDialog open={deleteOpen} onOpenChange={setDeleteOpen} />
    </>
  )
}

/**
 * For a new user this POST is often the first request to the backend, so it can
 * legitimately sit through a full Cloud Run cold start (~26s, retried up to 90s
 * by authFetch). Say so rather than leaving a silent spinner.
 */
function AcceptLabel({ isSubmitting }: { isSubmitting: boolean }) {
  const isSlow = useSlowLoading()
  if (!isSubmitting) return <>Agree and continue</>
  return <>{isSlow ? 'Still working, hang tight…' : 'Saving…'}</>
}
