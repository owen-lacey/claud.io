export function LoadingCard() {
  return <div className="bg-card border border-border/50 animate-pulse h-full rounded-xl">
    <div role="status" aria-live="polite" className="h-full bg-muted/50 rounded-xl flex items-center justify-center">
      <div className="flex items-center gap-2 text-muted-foreground">
        <div className="w-4 h-4 border-2 border-primary/20 border-t-primary rounded-full animate-spin"></div>
        <span className="text-sm">Loading...</span>
      </div>
      <span className="sr-only">Loading...</span>
    </div>
  </div>
}