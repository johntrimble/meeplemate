import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  ExternalLinkIcon,
  FileTextIcon,
  LogOutIcon,
  ShieldIcon,
  Trash2Icon,
} from 'lucide-react'
import { useAuth } from '@/auth/useAuth'
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar'
import { DeleteAccountDialog } from '@/components/DeleteAccountDialog'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { initials } from '@/lib/initials'
import { cn } from '@/lib/utils'

/**
 * Account avatar with a dropdown menu. Shows the signed-in user's Google
 * profile photo (falling back to their initials) and offers Log out plus
 * Delete account.
 */
export function UserMenu({ className }: { className?: string }) {
  const { user, logout } = useAuth()
  const navigate = useNavigate()
  const [deleteOpen, setDeleteOpen] = useState(false)

  const handleLogout = () => {
    logout()
    navigate('/', { replace: true })
  }

  const label = user?.name?.trim() || user?.email?.trim() || 'Account'

  return (
    <>
    <DropdownMenu>
      <DropdownMenuTrigger
        aria-label="Account menu"
        className="rounded-full outline-hidden focus-visible:ring-2 focus-visible:ring-ring shrink-0"
      >
        <Avatar className={cn('size-8', className)}>
          {user?.photoURL && <AvatarImage src={user.photoURL} alt={user.name ?? ''} />}
          <AvatarFallback className="text-xs font-semibold">
            {initials(user?.name ?? null, user?.email ?? null)}
          </AvatarFallback>
        </Avatar>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="min-w-48">
        <DropdownMenuLabel className="truncate font-normal">{label}</DropdownMenuLabel>
        <DropdownMenuSeparator />
        {/* New tab, and real anchors rather than navigate() or window.open.
            The menu is only ever rendered over the chat and select-game views,
            and ChatInput holds the unsent question in local state - navigating
            away in-tab unmounts it and silently discards whatever the user was
            part-way through typing. A tab also keeps their scroll position and
            costs nothing to close. Anchors (not window.open) so middle-click
            and "open in new window" behave, and no popup blocker is involved. */}
        <DropdownMenuItem asChild>
          <a href="/terms/" target="_blank" rel="noreferrer">
            <FileTextIcon />
            Terms of Use
            {/* Trailing and right-aligned: the leading slot is the item's own
                icon, and a new-tab hint reads as a property of the row rather
                than part of its label. */}
            <ExternalLinkIcon aria-hidden className="ml-auto !size-3" />
            <span className="sr-only">(opens in a new tab)</span>
          </a>
        </DropdownMenuItem>
        <DropdownMenuItem asChild>
          <a href="/privacy/" target="_blank" rel="noreferrer">
            <ShieldIcon />
            Privacy Policy
            <ExternalLinkIcon aria-hidden className="ml-auto !size-3" />
            <span className="sr-only">(opens in a new tab)</span>
          </a>
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem onSelect={handleLogout}>
          <LogOutIcon />
          Log out
        </DropdownMenuItem>
        <DropdownMenuItem variant="destructive" onSelect={() => setDeleteOpen(true)}>
          <Trash2Icon />
          Delete account
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
    {/* Sibling of the menu, not a child: Radix unmounts DropdownMenuContent on
        select, which would tear the dialog down with it. */}
    <DeleteAccountDialog open={deleteOpen} onOpenChange={setDeleteOpen} />
    </>
  )
}
