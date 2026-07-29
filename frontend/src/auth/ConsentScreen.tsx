import { useState } from 'react'
import { useAuth } from './useAuth'
import { useAuthFetch } from './authFetch'
import { postAcceptance } from './useAcceptanceSync'
import { Button } from '@/components/ui/button'
import { BrandMark } from '@/components/BrandMark'
import { Checkbox } from '@/components/ui/checkbox'
import { DeleteAccountDialog } from '@/components/DeleteAccountDialog'
import { ExternalLink } from '@/components/ExternalLink'
import { writeAcceptance } from '@/lib/legal'
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
  /** Called once acceptance is recorded locally. Does not wait for the server. */
  onAccepted: () => void
}) {
  const { user, logout } = useAuth()
  const authFetch = useAuthFetch()

  const [agreed, setAgreed] = useState(false)
  const [deleteOpen, setDeleteOpen] = useState(false)

  const isReconsent = state === 'outdated'

  /**
   * Record locally, open the app, and let the POST land whenever it lands.
   *
   * Nothing here awaits the network. For a new user this POST is the *first*
   * request the backend ever sees - every other call site is inside CacheGate's
   * children - so it eats the entire Cloud Run cold start by construction.
   * Waiting on it meant a 26-90s spinner before the app appeared, sitting on top
   * of a game list that had already been seeded from the CDN snapshot and was
   * ready to paint. `useAcceptanceSync` retries if it never arrives.
   */
  const handleAccept = () => {
    if (!agreed || !user) return
    writeAcceptance(user.uid)
    void postAcceptance(authFetch, user.uid)
    onAccepted()
  }

  return (
    <>
      <div className="fixed inset-0 overflow-y-auto bg-background">
        <div className="mx-auto flex min-h-full max-w-md flex-col justify-center gap-6 px-6 py-10">
          <div className="flex flex-col items-center gap-2 text-center">
            <BrandMark className="size-9" />
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

          <div className="flex flex-col gap-2">
            {/* No pending state: accepting is a local write, so this screen is
                gone by the next frame. There is nothing to spin on and nothing
                that can fail in front of the user. */}
            <Button onClick={handleAccept} disabled={!agreed}>
              Agree and continue
            </Button>
            <Button variant="ghost" onClick={logout}>
              Sign out
            </Button>
          </div>

          {/* Signing out just leaves the account sitting here un-accepted, so
              deletion is the only way to actually withdraw. */}
          <button
            type="button"
            onClick={() => setDeleteOpen(true)}
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
