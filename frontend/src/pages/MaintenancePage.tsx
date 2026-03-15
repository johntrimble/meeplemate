export default function MaintenancePage() {
  return (
    <div className="fixed inset-0 bg-background flex flex-col items-center justify-center gap-6">
      <div className="flex flex-col items-center gap-2 text-center">
        <span className="text-4xl">🔧</span>
        <h1 className="text-xl font-semibold text-foreground">Down for maintenance</h1>
        <p className="text-sm text-muted-foreground">
          Boardbarian is temporarily unavailable. Check back soon.
        </p>
      </div>
    </div>
  )
}
