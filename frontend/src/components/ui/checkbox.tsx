import * as React from "react"
import { Checkbox as CheckboxPrimitive } from "radix-ui"
import { CheckIcon } from "lucide-react"

import { cn } from "@/lib/utils"

function Checkbox({
  className,
  ...props
}: React.ComponentProps<typeof CheckboxPrimitive.Root>) {
  return (
    <CheckboxPrimitive.Root
      data-slot="checkbox"
      className={cn(
        // `dark:data-[state=unchecked]:bg-input/30`, not upstream shadcn's bare
        // `dark:bg-input/30`. Tailwind sorts the `dark:` variant after
        // `data-[state=checked]:`, so the bare form wins in dark mode and the
        // checked background stays `input/30` while the tick still picks up
        // `text-primary-foreground` - which in this theme is near-black
        // (oklch 0.205). The result is a black check on a black box: invisible,
        // and the box merely looks slightly brighter when ticked. Scoping the
        // dark background to the unchecked state removes the collision. Same
        // fix switch.tsx already uses.
        "peer border-input dark:data-[state=unchecked]:bg-input/30 data-[state=checked]:bg-primary data-[state=checked]:text-primary-foreground data-[state=checked]:border-primary focus-visible:border-ring focus-visible:ring-ring/50 aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive size-4 shrink-0 rounded-[4px] border shadow-xs transition-shadow outline-none focus-visible:ring-[3px] disabled:cursor-not-allowed disabled:opacity-50",
        className
      )}
      {...props}
    >
      <CheckboxPrimitive.Indicator
        data-slot="checkbox-indicator"
        className="flex items-center justify-center text-current transition-none"
      >
        <CheckIcon className="size-3.5" />
      </CheckboxPrimitive.Indicator>
    </CheckboxPrimitive.Root>
  )
}

export { Checkbox }
