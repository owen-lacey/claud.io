export function LoadingCard() {
  return <div role="status" className="animate-pulse h-full">
    <div className="h-full bg-gray-200 rounded-lg"></div>
    <span className="sr-only">Loading...</span>
  </div>
}