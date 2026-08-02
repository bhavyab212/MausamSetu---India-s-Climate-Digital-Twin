const colors = [
  ["background", "#F7F8FA", "bg-background"],
  ["workspace", "rgba(249,250,250,.92)", "bg-workspace"],
  ["card", "rgba(255,255,255,.88)", "bg-card"],
  ["cardSolid", "#FEFFFE", "bg-card-solid"],
  ["cardMuted", "#F1F6FA", "bg-card-muted"],
  ["cardBorder", "#E1E4E8", "bg-cardBorder"],
  ["divider", "#D8DCE2", "bg-divider"],
  ["textPrimary", "#111318", "bg-textPrimary"],
  ["textSecondary", "#56606B", "bg-textSecondary"],
  ["textTertiary", "#7A8390", "bg-textTertiary"],
  ["textInverse", "#FFFFFF", "bg-textInverse"],
  ["primary", "#0061FE", "bg-primary"],
  ["primaryStrong", "#0854FE", "bg-primary-strong"],
  ["primarySoft", "#EAF2FE", "bg-primary-soft"],
  ["positive", "#07945B", "bg-positive"],
  ["positiveSoft", "#E5FCED", "bg-positive-soft"],
  ["warning", "#F57602", "bg-warning"],
  ["warningSoft", "#FEF7EA", "bg-warning-soft"],
  ["critical", "#ED3335", "bg-critical"],
  ["criticalStrong", "#EC3736", "bg-critical-strong"],
  ["criticalSoft", "#FDEBEC", "bg-critical-soft"],
  ["rain-0", "#DDEBFF", "bg-rain-0"],
  ["rain-25", "#ABD2FF", "bg-rain-25"],
  ["rain-50", "#8FBEFD", "bg-rain-50"],
  ["rain-75", "#3D91FB", "bg-rain-75"],
  ["rain-100", "#0061FE", "bg-rain-100"],
  ["temperature-low", "#FEF3C7", "bg-temperature-low"],
  ["temperature-high", "#B91C1C", "bg-temperature-high"],
  ["anomaly-negative", "#E56604", "bg-anomaly-negative"],
  ["anomaly-neutral", "#FFFFFF", "bg-anomaly-neutral"],
  ["anomaly-positive", "#143CB9", "bg-anomaly-positive"],
  ["missing", "#E5E7EB", "bg-missing"],
] as const;

const typeSpecimens = [
  ["hero", "text-hero", "48 / 600"],
  ["page", "text-page", "24 / 650"],
  ["title", "text-title", "16 / 600"],
  ["label", "text-label", "13 / 600"],
  ["body", "text-body", "14 / 400"],
  ["caption", "text-caption", "11 / 500"],
  ["eyebrow", "text-eyebrow uppercase", "11 / 600 / tracked"],
] as const;

const radii = [
  ["control", "rounded-control"],
  ["card", "rounded-card"],
  ["panel", "rounded-panel"],
  ["shell", "rounded-shell"],
  ["pill", "rounded-pill"],
] as const;

export default function TokensTestPage() {
  return (
    <main className="mx-auto max-w-7xl space-y-8 p-8">
      <header>
        <p className="text-eyebrow uppercase text-primary">Phase 2 · token verification</p>
        <h1 className="mt-2 text-page">MausamSetu मौसम सेतु</h1>
        <p className="mt-2 text-body text-textSecondary">
          Locked design tokens extracted from seven approved references.
        </p>
      </header>

      <section>
        <h2 className="mb-4 text-title">Color system</h2>
        <div className="grid grid-cols-6 gap-column-gap">
          {colors.map(([name, value, className]) => (
            <article key={name} className="overflow-hidden rounded-card border bg-card-solid shadow-elevation-card">
              <div className={`h-20 ${className}`} />
              <div className="space-y-1 p-card">
                <p className="text-label">{name}</p>
                <p className="font-mono text-caption text-textSecondary">{value}</p>
                <p className="font-mono text-caption text-textTertiary">{className}</p>
              </div>
            </article>
          ))}
        </div>
      </section>

      <section className="grid grid-cols-2 gap-column-gap">
        <article className="rounded-panel border bg-card p-card-lg shadow-elevation-card">
          <h2 className="mb-5 text-title">Typography</h2>
          <div className="space-y-5">
            {typeSpecimens.map(([name, className, metric]) => (
              <div key={name} className="border-b pb-4 last:border-0 last:pb-0">
                <p className={`${className}`}>Monsoon signal · {name}</p>
                <p className="mt-1 font-mono text-caption text-textTertiary">{className} · {metric}</p>
              </div>
            ))}
            <p className="font-mono text-body tabular-nums">JetBrains Mono · T-25:48 · 32.4 mm</p>
          </div>
        </article>

        <article className="rounded-panel border bg-card p-card-lg shadow-elevation-card">
          <h2 className="mb-5 text-title">Radius & elevation</h2>
          <div className="grid grid-cols-2 gap-column-gap">
            {radii.map(([name, className]) => (
              <div key={name} className={`${className} border bg-card-muted p-card shadow-elevation-card`}>
                <p className="text-label">{name}</p>
                <p className="mt-1 font-mono text-caption text-textTertiary">{className}</p>
              </div>
            ))}
            <div className="rounded-panel border bg-card p-card shadow-elevation-floating">
              <p className="text-label">floating</p>
              <p className="mt-1 font-mono text-caption text-textTertiary">shadow-elevation-floating</p>
            </div>
          </div>
        </article>
      </section>
    </main>
  );
}
