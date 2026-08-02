import Link from "next/link"

export default function NotFound() {
  return (
    <main className="flex min-h-screen items-center justify-center bg-background px-workspace py-workspace">
      <section className="w-full max-w-xl rounded-panel border border-cardBorder bg-card p-card-lg text-center shadow-elevation-card">
        <p className="text-eyebrow uppercase text-primary">MausamSetu मौसम सेतु</p>
        <h1 className="mt-3 font-mono text-hero text-textPrimary">Page not found</h1>
        <p className="mx-auto mt-3 max-w-md text-body text-textSecondary">
          This route is not part of the current Phase 3 backend integration. Open the live component workspace to verify the connected system.
        </p>
        <Link
          href="/kitchen-sink"
          className="mt-6 inline-flex min-h-target items-center justify-center rounded-button bg-primary px-4 text-label text-primary-foreground transition-colors duration-fast hover:bg-primary-strong focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
        >
          Open live workspace
        </Link>
      </section>
    </main>
  )
}
