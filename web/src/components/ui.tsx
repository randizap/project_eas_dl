import { useEffect, useId, useRef, useState, type ReactNode } from "react";

export function cx(...c: (string | false | null | undefined)[]) {
  return c.filter(Boolean).join(" ");
}

export function Eyebrow({ n, children }: { n?: string; children: ReactNode }) {
  return (
    <div className="eyebrow flex items-center gap-3">
      <span aria-hidden>◆</span>
      {n && <span>{n}</span>}
      {n && <span aria-hidden>·</span>}
      <span>{children}</span>
    </div>
  );
}

export function PageHeader({ n, eyebrow, title, lead }: { n?: string; eyebrow: string; title: ReactNode; lead?: ReactNode }) {
  return (
    <header className="pt-14 pb-10 sm:pt-20 fade-in">
      <Eyebrow n={n}>{eyebrow}</Eyebrow>
      <h1 className="mt-5 text-4xl sm:text-5xl font-extrabold tracking-tight leading-[1.05] max-w-4xl">{title}</h1>
      {lead && <p className="mt-5 text-lg text-ink-2 max-w-3xl leading-relaxed">{lead}</p>}
    </header>
  );
}

export function Section({ id, title, kicker, children, aside }: {
  id?: string; title: ReactNode; kicker?: ReactNode; children: ReactNode; aside?: ReactNode;
}) {
  return (
    <section id={id} className="py-10 scroll-mt-24">
      <div className="flex flex-wrap items-end justify-between gap-4 mb-6">
        <div className="max-w-3xl">
          <h2 className="text-2xl sm:text-[1.7rem] font-bold tracking-tight">{title}</h2>
          {kicker && <p className="mt-2 text-ink-2 leading-relaxed">{kicker}</p>}
        </div>
        {aside}
      </div>
      {children}
    </section>
  );
}

export function Card({ children, className, tone }: { children: ReactNode; className?: string; tone?: "coral" | "lime" }) {
  return (
    <div
      className={cx(
        "rounded-2xl border bg-surface/80 backdrop-blur-sm",
        tone === "coral" ? "border-coral/40 bg-gradient-to-br from-coral/10 to-surface/80" :
        tone === "lime" ? "border-lime/30 bg-gradient-to-br from-lime/8 to-surface/80" : "border-line",
        className,
      )}
    >
      {children}
    </div>
  );
}

export function Pill({ children, active, onClick, color }: { children: ReactNode; active?: boolean; onClick?: () => void; color?: string }) {
  const Tag = onClick ? "button" : "span";
  return (
    <Tag
      onClick={onClick}
      aria-pressed={onClick ? !!active : undefined}
      className={cx(
        "inline-flex items-center gap-2 rounded-full border px-3 py-1 font-mono text-xs transition-colors",
        active ? "border-lime bg-lime text-bg font-semibold" : "border-line-strong text-ink-2",
        onClick && !active && "hover:border-lime/60 hover:text-ink cursor-pointer",
      )}
    >
      {color && <span className="h-2 w-2 rounded-full" style={{ background: color }} aria-hidden />}
      {children}
    </Tag>
  );
}

/** Kontrol segmen: satu pilihan dari beberapa. */
export function Segmented<T extends string>({ value, onChange, options, label }: {
  value: T; onChange: (v: T) => void; options: { value: T; label: ReactNode; color?: string }[]; label?: string;
}) {
  return (
    <div className="flex flex-wrap items-center gap-2" role="radiogroup" aria-label={label}>
      {label && <span className="font-mono text-[11px] uppercase tracking-widest text-muted mr-1">{label}</span>}
      <div className="inline-flex rounded-xl border border-line bg-bg/60 p-1">
        {options.map((o) => (
          <button
            key={o.value}
            role="radio"
            aria-checked={value === o.value}
            onClick={() => onChange(o.value)}
            className={cx(
              "flex items-center gap-2 rounded-lg px-3 py-1.5 text-sm transition-colors",
              value === o.value ? "bg-surface-2 text-ink shadow-[inset_0_0_0_1px_var(--color-line-strong)]" : "text-muted hover:text-ink",
            )}
          >
            {o.color && <span className="h-2 w-2 rounded-full" style={{ background: o.color }} aria-hidden />}
            {o.label}
          </button>
        ))}
      </div>
    </div>
  );
}

export function Callout({ tone = "info", title, children }: { tone?: "info" | "warn" | "good"; title?: ReactNode; children: ReactNode }) {
  const map = {
    info: ["border-line-strong", "text-ink-2", "ⓘ"],
    warn: ["border-amber/40 bg-amber/5", "text-amber", "!"],
    good: ["border-lime/40 bg-lime/5", "text-lime", "✓"],
  } as const;
  const [b, t, icon] = map[tone];
  return (
    <div className={cx("rounded-xl border px-4 py-3 text-sm leading-relaxed", b)}>
      <div className="flex gap-3">
        <span className={cx("font-mono font-bold", t)} aria-hidden>{icon}</span>
        <div className="text-ink-2">
          {title && <div className={cx("font-semibold mb-0.5", t)}>{title}</div>}
          {children}
        </div>
      </div>
    </div>
  );
}

/** Istilah teknis dengan penjelasan singkat (hover / fokus / ketuk). */
export function Term({ children, tip }: { children: ReactNode; tip: ReactNode }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  const ref = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    if (!open) return;
    const h = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false); };
    document.addEventListener("click", h);
    return () => document.removeEventListener("click", h);
  }, [open]);
  return (
    <span ref={ref} className="relative inline-block">
      <button
        type="button"
        aria-describedby={open ? id : undefined}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onClick={() => setOpen((v) => !v)}
        className="underline decoration-dotted decoration-lime/70 underline-offset-4 text-inherit cursor-help"
      >
        {children}
      </button>
      {open && (
        <span
          id={id}
          role="tooltip"
          className="absolute left-1/2 top-full z-40 mt-2 w-72 -translate-x-1/2 rounded-xl border border-line-strong bg-surface-2 p-3 text-left text-[13px] font-normal leading-relaxed text-ink-2 shadow-2xl normal-case tracking-normal"
        >
          {tip}
        </span>
      )}
    </span>
  );
}

export function StatTile({ label, value, sub, tone, children }: {
  label: ReactNode; value: ReactNode; sub?: ReactNode; tone?: "coral" | "lime"; children?: ReactNode;
}) {
  return (
    <Card tone={tone} className="p-6 sm:p-8">
      <div className={cx("font-mono text-xs uppercase tracking-[0.2em]", tone === "coral" ? "text-coral" : "text-muted")}>{label}</div>
      <div className={cx("mt-4 text-6xl sm:text-7xl font-extrabold tracking-tight tnum leading-none", tone === "coral" ? "text-coral" : "text-ink")}>
        {value}
      </div>
      {sub && <div className="mt-4 text-ink-2">{sub}</div>}
      {children}
    </Card>
  );
}

export function DataTable({ head, rows, dense, caption }: {
  head: ReactNode[]; rows: ReactNode[][]; dense?: boolean; caption?: string;
}) {
  return (
    <div className="overflow-x-auto rounded-xl border border-line">
      <table className="w-full text-sm tnum">
        {caption && <caption className="sr-only">{caption}</caption>}
        <thead className="bg-surface-2/70 text-left">
          <tr>
            {head.map((h, i) => (
              <th key={i} className={cx("font-mono text-[11px] uppercase tracking-widest text-muted font-medium whitespace-nowrap", dense ? "px-3 py-2" : "px-4 py-3")}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-t border-line hover:bg-white/[0.02]">
              {r.map((c, j) => (
                <td key={j} className={cx("align-top text-ink-2", dense ? "px-3 py-2" : "px-4 py-3", j === 0 && "text-ink")}>{c}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Bingkai grafik: judul, legenda, dan tombol tampilan tabel (aksesibilitas). */
export function ChartFrame({ title, note, legend, table, children, controls }: {
  title: ReactNode; note?: ReactNode; legend?: ReactNode; table?: ReactNode; children: ReactNode; controls?: ReactNode;
}) {
  const [showTable, setShowTable] = useState(false);
  return (
    <Card className="min-w-0 p-5 sm:p-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="font-semibold text-ink">{title}</h3>
          {note && <p className="mt-1 text-sm text-muted max-w-2xl">{note}</p>}
        </div>
        {table && (
          <button
            onClick={() => setShowTable((v) => !v)}
            className="font-mono text-[11px] uppercase tracking-widest text-muted hover:text-lime border border-line rounded-lg px-2.5 py-1.5"
            aria-pressed={showTable}
          >
            {showTable ? "◧ grafik" : "▦ tabel"}
          </button>
        )}
      </div>
      {controls && <div className="mt-4 flex flex-wrap gap-4">{controls}</div>}
      {legend && <div className="mt-4 flex flex-wrap gap-x-5 gap-y-2 text-sm text-ink-2">{legend}</div>}
      <div className="mt-4">{showTable && table ? table : children}</div>
    </Card>
  );
}

export function LegendItem({ color, children, shape = "dot" }: { color: string; children: ReactNode; shape?: "dot" | "line" | "ring" | "rect" }) {
  return (
    <span className="inline-flex items-center gap-2">
      {shape === "line" ? (
        <span className="inline-block h-0.5 w-4 rounded" style={{ background: color }} />
      ) : shape === "ring" ? (
        <span className="inline-block h-2.5 w-2.5 rounded-full border-2" style={{ borderColor: color }} />
      ) : shape === "rect" ? (
        <span className="inline-block h-2.5 w-3 rounded-sm" style={{ background: color }} />
      ) : (
        <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: color }} />
      )}
      {children}
    </span>
  );
}

export function Verdict({ ok, children }: { ok: boolean | null; children: ReactNode }) {
  return (
    <span
      className={cx(
        "inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 font-mono text-[11px] uppercase tracking-wider",
        ok === null ? "bg-white/5 text-muted" : ok ? "bg-lime/15 text-lime" : "bg-white/5 text-ink-2",
      )}
    >
      <span aria-hidden>{ok === null ? "·" : ok ? "●" : "○"}</span>
      {children}
    </span>
  );
}
