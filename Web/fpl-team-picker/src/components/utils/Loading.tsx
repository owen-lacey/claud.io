export function LoadingCard() {
  return <div className="bg-card border border-border animate-pulse h-full">
    <div role="status" aria-live="polite" className="h-full bg-gray-200 rounded-lg">
      <span className="sr-only">Loading...</span>
    </div>
  </div>
}