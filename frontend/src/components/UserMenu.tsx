import { useNavigate } from 'react-router-dom'
import { LogOutIcon } from 'lucide-react'
import { useAuth } from '@/auth/useAuth'
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar'
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
 * profile photo (falling back to their initials) and offers a Log out action
 * that signs the user out and returns them to the home screen.
 */
export function UserMenu({ className }: { className?: string }) {
  const { user, logout } = useAuth()
  const navigate = useNavigate()

  const handleLogout = () => {
    logout()
    navigate('/')
  }

  const label = user?.name || user?.email || 'Account'

  return (
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
        <DropdownMenuItem variant="destructive" onSelect={handleLogout}>
          <LogOutIcon />
          Log out
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
