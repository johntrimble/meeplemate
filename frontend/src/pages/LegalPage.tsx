import { useEffect, type ComponentProps } from 'react'
import { useNavigate } from 'react-router-dom'
import { ArrowLeftIcon } from 'lucide-react'
import { Streamdown } from 'streamdown'
import { Button } from '@/components/ui/button'
import termsMarkdown from '@/content/terms.md?raw'
import privacyMarkdown from '@/content/privacy.md?raw'

/**
 * Renders the Terms of Use or Privacy Policy.
 *
 * The markdown is imported with `?raw` rather than fetched, so the document a
 * user reads is always the one the bundle was built from - the same build whose
 * `legal.ts` constants decide whether they need to accept it. A runtime fetch
 * could serve a newer document than the version the consent screen records.
 *
 * `Streamdown` is used bare here rather than via `MessageResponse`: these are
 * static prose documents, and the chat wrapper pulls in the mermaid and syntax
 * highlighting plugins, which have no business in a page anonymous visitors
 * (and Google's OAuth reviewer) load.
 *
 * Deliberately routed outside `CacheGate` - the documents must be readable
 * without signing in, both because it would be perverse to gate the terms
 * behind agreeing to them and because OAuth brand verification requires the
 * privacy policy to be publicly reachable.
 */
const DOCS = {
  terms: { markdown: termsMarkdown, title: 'Terms of Use' },
  privacy: { markdown: privacyMarkdown, title: 'Privacy Policy' },
} as const

/**
 * Render links as real anchors.
 *
 * Streamdown's default link renderer emits an inert `<button>` with no `href` -
 * reasonable for streaming model output, where a URL may still be half-written
 * and shouldn't be clickable yet. Here it's actively wrong: every contact route
 * in both documents is a `mailto:support@boardbarian.com`, and the Privacy
 * Policy directs users there to exercise access, correction, deletion and
 * objection requests. Underlined text that does nothing when clicked would make
 * the only stated way to exercise those rights a dead end.
 */
const components = {
  a: ({ href, children, ...props }: ComponentProps<'a'>) => (
    <a
      href={href}
      // External destinations only ever appear as mailto: today, but treat any
      // off-site link as untrusted rather than assuming that stays true.
      {...(href?.startsWith('http') ? { target: '_blank', rel: 'noreferrer' } : {})}
      className="font-medium text-primary underline underline-offset-2 wrap-anywhere"
      {...props}
    >
      {children}
    </a>
  ),
}

export default function LegalPage({ doc }: { doc: keyof typeof DOCS }) {
  const navigate = useNavigate()
  const { markdown, title } = DOCS[doc]

  // The documents get linked and bookmarked directly, so the tab should say
  // which one it is rather than inheriting the app's default title.
  useEffect(() => {
    const previous = document.title
    document.title = `${title} · Boardbarian`
    return () => {
      document.title = previous
    }
  }, [title])

  return (
    // `h-full overflow-y-auto`, not `min-h-dvh`: index.css pins
    // html/body/#root to height:100% with overflow:hidden, so the document
    // never scrolls and every page has to own its scroll container. Same shape
    // as HomePage. With `min-h-dvh` these documents simply overflow a clipped
    // root and can't be scrolled at all.
    <div className="h-full overflow-y-auto bg-background">
      <div className="mx-auto max-w-3xl px-4 py-8 sm:px-6 sm:py-12">
        <Button
          variant="ghost"
          size="sm"
          className="-ml-2 mb-6 text-muted-foreground"
          onClick={() => navigate('/')}
        >
          <ArrowLeftIcon className="size-4" />
          Back to Boardbarian
        </Button>

        <Streamdown
          // Long-form prose needs more breathing room between blocks than the
          // chat bubble it's normally rendered in.
          className={
            'text-foreground [&_h1]:mb-2 [&_h1]:text-2xl [&_h1]:font-semibold ' +
            '[&_h2]:mt-8 [&_h2]:mb-3 [&_h2]:text-lg [&_h2]:font-semibold ' +
            '[&_h3]:mt-6 [&_h3]:mb-2 [&_h3]:font-medium ' +
            '[&_p]:my-3 [&_p]:leading-relaxed [&_li]:my-1 [&_li]:leading-relaxed ' +
            '[&_ul]:my-3 [&_ul]:list-disc [&_ul]:pl-6 ' +
            '[&_a]:underline [&_a]:underline-offset-2'
          }
          components={components}
        >
          {markdown}
        </Streamdown>
      </div>
    </div>
  )
}
