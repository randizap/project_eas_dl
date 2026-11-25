import { useState, type ReactNode } from "react";
import { inkOn, linear, niceTicks, seqBlue, TipRow, TooltipBox, useWidth, XAxis, type Tip } from "./core";
import { num, pct, signedPct } from "../lib/data";

// =========================================================================== Dumbbell (ID vs OOD)
export interface DumbbellRow {
  id: string;
  label: ReactNode;
  short?: string;
  color: string;
  a: { v: number | null; lo: number | null; hi: number | null; label: string };
  b: { v: number | null; lo: number | null; hi: number | null; label: string };
}

/** Satu baris per konfigurasi: titik penuh = ID, cincin = OOD, garis = celah. Whisker = CI 95%. */
export function Dumbbell({ rows, domain = [0, 1] }: { rows: DumbbellRow[]; domain?: [number, number] }) {
  const [ref, w] = useWidth<HTMLDivElement>();
  const [tip, setTip] = useState<Tip>(null);
  const labelW = w < 560 ? 0 : 190;
  const m = { l: labelW + 12, r: 24, t: 8, b: 30 };
  const rowH = w < 560 ? 64 : 52;
  const h = m.t + m.b + rows.length * rowH;
  const x = linear(domain[0], domain[1], m.l, w - m.r);
  const ticks = niceTicks(domain[0], domain[1], w < 500 ? 4 : 6);

  const point = (r: DumbbellRow, p: DumbbellRow["a"], cy: number, filled: boolean, key: string) => {
    if (p.v === null) return null;
    const show = (e: React.PointerEvent | React.FocusEvent) => {
      const rect = (e.currentTarget as SVGElement).ownerSVGElement!.getBoundingClientRect();
      const cx = x(p.v!);
      setTip({
        x: "clientX" in e ? e.clientX - rect.left : cx, y: "clientY" in e ? e.clientY - rect.top : cy,
        content: (
          <>
            <div className="mb-1 font-semibold text-ink">{p.label}</div>
            <TipRow color={r.color} shape="dot" label="rerata 3 seed" value={`${pct(p.v)}%`} />
            <TipRow label="CI 95%" value={`${pct(p.lo)} – ${pct(p.hi)}`} />
          </>
        ),
      });
    };
    return (
      <g key={key}>
        {p.lo !== null && p.hi !== null && (
          <line x1={x(p.lo)} x2={x(p.hi)} y1={cy} y2={cy} stroke={r.color} strokeWidth={2} strokeLinecap="round" opacity={0.55} />
        )}
        <circle cx={x(p.v)} cy={cy} r={6.5} fill={filled ? r.color : "var(--color-surface)"} stroke={r.color} strokeWidth={2.5} />
        <circle
          cx={x(p.v)} cy={cy} r={14} fill="transparent" tabIndex={0} aria-label={`${p.label}: ${pct(p.v)}%`}
          onPointerMove={show} onFocus={show} onPointerLeave={() => setTip(null)} onBlur={() => setTip(null)}
          className="cursor-pointer outline-none"
        />
      </g>
    );
  };

  return (
    <div ref={ref} className="relative w-full min-w-0">
      <svg width={w} height={h} role="img" aria-label="Perbandingan in-distribution dan out-of-distribution">
        <XAxis ticks={ticks} x={x} y0={m.t} y1={h - m.b} fmt={(v) => `${Math.round(v * 100)}`} />
        {rows.map((r, i) => {
          const cy = m.t + i * rowH + rowH / 2 + (w < 560 ? 8 : 0);
          const { a, b } = r;
          return (
            <g key={r.id}>
              {labelW > 0 ? (
                <foreignObject x={0} y={cy - 18} width={labelW} height={36}>
                  <div className="flex h-full items-center text-[13px] leading-tight text-ink-2">{r.label}</div>
                </foreignObject>
              ) : (
                <text x={m.l} y={cy - 20} fontSize={12} fill="var(--color-ink-2)">{typeof r.label === "string" ? r.label : r.short ?? r.id}</text>
              )}
              {a.v !== null && b.v !== null && (
                <line x1={x(a.v)} x2={x(b.v)} y1={cy} y2={cy} stroke={r.color} strokeWidth={6} opacity={0.18} strokeLinecap="round" />
              )}
              {point(r, a, cy, true, "a")}
              {point(r, b, cy, false, "b")}
              {a.v !== null && b.v !== null && w >= 420 && (
                <text
                  x={(x(a.v) + x(b.v)) / 2} y={cy - 11} textAnchor="middle" fontSize={11}
                  fill="var(--color-ink-2)" fontFamily="var(--font-mono)"
                >
                  Δ {signedPct(b.v - a.v)}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      <TooltipBox tip={tip} width={w} />
    </div>
  );
}

// =========================================================================== Forest plot (selisih + CI)
export interface ForestRow {
  id: string;
  label: ReactNode;
  sub?: ReactNode;
  v: number | null;
  lo: number | null;
  hi: number | null;
  color?: string;
}

/** Selisih dengan CI terhadap garis nol. CI yang tidak menyentuh nol ditandai titik penuh. */
export function Forest({ rows, unit = "poin %" }: { rows: ForestRow[]; unit?: string }) {
  const [ref, w] = useWidth<HTMLDivElement>();
  const [tip, setTip] = useState<Tip>(null);
  const vals = rows.flatMap((r) => [r.lo, r.hi, r.v]).filter((v): v is number => v !== null);
  const ext = Math.max(0.05, ...vals.map(Math.abs)) * 1.15;
  const labelW = w < 560 ? 0 : 220;
  const m = { l: labelW + 12, r: 24, t: 8, b: 30 };
  const rowH = w < 560 ? 58 : 46;
  const h = m.t + m.b + rows.length * rowH;
  const x = linear(-ext, ext, m.l, w - m.r);
  const ticks = niceTicks(-ext, ext, w < 500 ? 4 : 6);
  return (
    <div ref={ref} className="relative w-full min-w-0">
      <svg width={w} height={h} role="img" aria-label="Selisih dan interval kepercayaan">
        <XAxis ticks={ticks} x={x} y0={m.t} y1={h - m.b} fmt={(v) => signedPct(v, 0)} />
        <line x1={x(0)} x2={x(0)} y1={m.t} y2={h - m.b} stroke="var(--color-ink-2)" strokeWidth={1.5} strokeDasharray="3 3" />
        <text x={w - m.r} y={h - 4} textAnchor="end" fontSize={10} fill="var(--color-muted)" fontFamily="var(--font-mono)">{unit}</text>
        {rows.map((r, i) => {
          const cy = m.t + i * rowH + rowH / 2 + (w < 560 ? 8 : 0);
          const sig = r.lo !== null && r.hi !== null && (r.lo > 0 || r.hi < 0);
          const c = r.color ?? "var(--color-lime)";
          const show = (e: React.PointerEvent | React.FocusEvent) => {
            const rect = (e.currentTarget as SVGElement).ownerSVGElement!.getBoundingClientRect();
            setTip({
              x: "clientX" in e ? e.clientX - rect.left : x(r.v ?? 0), y: "clientY" in e ? e.clientY - rect.top : cy,
              content: (
                <>
                  <div className="mb-1 font-semibold text-ink">{r.label}</div>
                  <TipRow label="selisih" value={`${signedPct(r.v)} ${unit}`} />
                  <TipRow label="CI 95%" value={`${signedPct(r.lo)} … ${signedPct(r.hi)}`} />
                  <div className="mt-1 text-muted">{sig ? "CI tidak memuat 0" : "CI memuat 0 → belum bisa dibedakan dari nol"}</div>
                </>
              ),
            });
          };
          return (
            <g key={r.id}>
              {labelW > 0 ? (
                <foreignObject x={0} y={cy - 20} width={labelW} height={40}>
                  <div className="flex h-full flex-col justify-center text-[13px] leading-tight text-ink-2">
                    <span>{r.label}</span>
                    {r.sub && <span className="font-mono text-[10px] text-muted">{r.sub}</span>}
                  </div>
                </foreignObject>
              ) : (
                <text x={m.l} y={cy - 16} fontSize={12} fill="var(--color-ink-2)">{typeof r.label === "string" ? r.label : r.id}</text>
              )}
              {r.lo !== null && r.hi !== null && (
                <line x1={x(r.lo)} x2={x(r.hi)} y1={cy} y2={cy} stroke={c} strokeWidth={2.5} strokeLinecap="round" />
              )}
              {r.v !== null && (
                <rect x={x(r.v) - 6} y={cy - 6} width={12} height={12} rx={2} transform={`rotate(45 ${x(r.v)} ${cy})`}
                  fill={sig ? c : "var(--color-surface)"} stroke={c} strokeWidth={2.5} />
              )}
              <rect x={m.l} y={cy - rowH / 2} width={w - m.l - m.r} height={rowH} fill="transparent" tabIndex={0}
                aria-label={`${typeof r.label === "string" ? r.label : r.id}: ${signedPct(r.v)}`}
                onPointerMove={show} onFocus={show} onPointerLeave={() => setTip(null)} onBlur={() => setTip(null)}
                className="outline-none" />
            </g>
          );
        })}
      </svg>
      <TooltipBox tip={tip} width={w} />
    </div>
  );
}

// =========================================================================== Heatmap
export function Heatmap({ rows, cols, values, fmt = (v) => pct(v, 0), cellMin = 44, rowLabelW = 170, tipLabel }: {
  rows: ReactNode[]; cols: string[]; values: (number | null)[][]; fmt?: (v: number | null) => string;
  cellMin?: number; rowLabelW?: number; tipLabel?: (r: number, c: number) => ReactNode;
}) {
  const [ref, w] = useWidth<HTMLDivElement>();
  const [tip, setTip] = useState<Tip>(null);
  const lw = w < 480 ? Math.min(110, rowLabelW) : rowLabelW;
  const cw = Math.max(cellMin, (w - lw - 4) / cols.length);
  const ch = Math.min(52, Math.max(34, cw * 0.62));
  const h = 26 + rows.length * ch;
  const width = lw + cols.length * cw;
  return (
    <div ref={ref} className="relative overflow-x-auto">
      <svg width={Math.max(width, 0)} height={h} role="img" aria-label="Peta panas">
        {cols.map((c, j) => (
          <text key={c} x={lw + j * cw + cw / 2} y={14} textAnchor="middle" fontSize={11} fill="var(--color-muted)" fontFamily="var(--font-mono)">{c}</text>
        ))}
        {rows.map((r, i) => (
          <g key={i}>
            <foreignObject x={0} y={26 + i * ch} width={lw - 8} height={ch}>
              <div className="flex h-full items-center text-[12px] leading-tight text-ink-2">{r}</div>
            </foreignObject>
            {cols.map((_, j) => {
              const v = values[i][j];
              const fill = v === null ? "#26272d" : seqBlue(v);
              const show = (e: React.PointerEvent | React.FocusEvent) => {
                const rect = (e.currentTarget as SVGElement).ownerSVGElement!.getBoundingClientRect();
                setTip({
                  x: "clientX" in e ? e.clientX - rect.left : lw + j * cw, y: "clientY" in e ? e.clientY - rect.top : 26 + i * ch,
                  content: tipLabel ? tipLabel(i, j) : <TipRow label={cols[j]} value={fmt(v)} />,
                });
              };
              return (
                <g key={j}>
                  <rect x={lw + j * cw + 1} y={26 + i * ch + 1} width={cw - 2} height={ch - 2} rx={4} fill={fill}
                    tabIndex={0} onPointerMove={show} onFocus={show} onPointerLeave={() => setTip(null)} onBlur={() => setTip(null)}
                    className="outline-none hover:opacity-85 focus:stroke-lime" strokeWidth={2} aria-label={`${cols[j]}: ${fmt(v)}`} />
                  <text x={lw + j * cw + cw / 2} y={26 + i * ch + ch / 2 + 4} textAnchor="middle" fontSize={12} fontWeight={600}
                    pointerEvents="none" fill={v === null ? "var(--color-muted)" : inkOn(v)} className="tnum">
                    {fmt(v)}
                  </text>
                </g>
              );
            })}
          </g>
        ))}
      </svg>
      <TooltipBox tip={tip} width={w} />
    </div>
  );
}

// =========================================================================== Line chart (crosshair)
export interface LineSeries {
  id: string;
  label: string;
  color: string;
  dashed?: boolean;
  points: { x: number; y: number | null }[];
}

export function LineChart({ series, xTicks, xLabel, yLabel, yDomain, yFmt = (v) => pct(v, 0), xFmt = (v) => String(v), height = 260, tipFmt }: {
  series: LineSeries[]; xTicks: number[]; xLabel?: string; yLabel?: string; yDomain?: [number, number];
  yFmt?: (v: number) => string; xFmt?: (v: number) => string; height?: number; tipFmt?: (v: number | null) => string;
}) {
  const [ref, w] = useWidth<HTMLDivElement>();
  const [hx, setHx] = useState<number | null>(null);
  const m = { l: 46, r: 16, t: yLabel ? 28 : 12, b: xLabel ? 44 : 30 };
  const all = series.flatMap((s) => s.points.map((p) => p.y)).filter((v): v is number => v !== null);
  const [y0, y1] = yDomain ?? [Math.min(0, ...all), Math.max(1e-6, ...all) * 1.08];
  const xs = series.flatMap((s) => s.points.map((p) => p.x));
  const [x0, x1] = [Math.min(...xTicks, ...xs), Math.max(...xTicks, ...xs)];
  const x = linear(x0, x1, m.l, w - m.r);
  const y = linear(y0, y1, height - m.b, m.t);
  const yt = niceTicks(y0, y1, 4);
  const xVals = [...new Set(xs)].sort((a, b) => a - b);

  const onMove = (e: React.PointerEvent<SVGRectElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - rect.left + m.l;
    let best = xVals[0];
    for (const v of xVals) if (Math.abs(x(v) - px) < Math.abs(x(best) - px)) best = v;
    setHx(best);
  };
  const tipContent = hx === null ? null : (
    <>
      <div className="mb-1 font-semibold text-ink">{xLabel ? `${xLabel}: ` : ""}{xFmt(hx)}</div>
      {series.map((s) => {
        const p = s.points.find((q) => q.x === hx);
        return <TipRow key={s.id} color={s.color} label={s.label} value={tipFmt ? tipFmt(p?.y ?? null) : p?.y == null ? "–" : yFmt(p.y)} />;
      })}
    </>
  );
  return (
    <div ref={ref} className="relative w-full min-w-0">
      <svg width={w} height={height} role="img" aria-label={yLabel ?? "grafik garis"}>
        {yt.map((t) => (
          <g key={t}>
            <line x1={m.l} x2={w - m.r} y1={y(t)} y2={y(t)} stroke="var(--grid)" />
            <text x={m.l - 8} y={y(t) + 4} textAnchor="end" fontSize={11} fill="var(--color-muted)" fontFamily="var(--font-mono)">{yFmt(t)}</text>
          </g>
        ))}
        <line x1={m.l} x2={w - m.r} y1={height - m.b} y2={height - m.b} stroke="var(--axis)" />
        {xTicks.map((t) => (
          <text key={t} x={x(t)} y={height - m.b + 16} textAnchor="middle" fontSize={11} fill="var(--color-muted)" fontFamily="var(--font-mono)">{xFmt(t)}</text>
        ))}
        {xLabel && <text x={(m.l + w - m.r) / 2} y={height - 6} textAnchor="middle" fontSize={11} fill="var(--color-muted)">{xLabel}</text>}
        {yLabel && <text x={4} y={12} fontSize={11} fill="var(--color-muted)">{yLabel}</text>}
        {hx !== null && <line x1={x(hx)} x2={x(hx)} y1={m.t} y2={height - m.b} stroke="var(--color-ink-2)" strokeWidth={1} opacity={0.5} />}
        {series.map((s) => {
          const pts = s.points.filter((p) => p.y !== null) as { x: number; y: number }[];
          const d = pts.map((p, i) => `${i ? "L" : "M"}${x(p.x)},${y(p.y)}`).join(" ");
          return (
            <g key={s.id}>
              <path d={d} fill="none" stroke={s.color} strokeWidth={2} strokeDasharray={s.dashed ? "6 4" : undefined} strokeLinejoin="round" strokeLinecap="round" />
              {pts.map((p) => (
                <circle key={p.x} cx={x(p.x)} cy={y(p.y)} r={hx === p.x ? 5 : 3.5} fill={s.dashed ? "var(--color-surface)" : s.color} stroke={s.color} strokeWidth={2} />
              ))}
            </g>
          );
        })}
        <rect x={m.l} y={m.t} width={Math.max(0, w - m.l - m.r)} height={height - m.t - m.b} fill="transparent"
          onPointerMove={onMove} onPointerLeave={() => setHx(null)} />
      </svg>
      <TooltipBox tip={hx === null ? null : { x: x(hx), y: m.t, content: tipContent }} width={w} />
    </div>
  );
}

// =========================================================================== Bar mini (probabilitas)
export function ProbBars({ labels, values, color, highlight }: { labels: string[]; values: number[]; color: string; highlight?: number }) {
  return (
    <div className="space-y-2">
      {labels.map((l, i) => (
        <div key={l} className="grid grid-cols-[3.2rem_1fr_3.6rem] items-center gap-3 text-sm">
          <span className={i === highlight ? "font-semibold text-ink font-mono" : "text-muted font-mono"}>{l}</span>
          <div className="h-3 rounded-full bg-white/5 overflow-hidden">
            <div className="h-full rounded-full transition-[width] duration-500" style={{ width: `${Math.max(1, values[i] * 100)}%`, background: color, opacity: i === highlight ? 1 : 0.55 }} />
          </div>
          <span className="text-right tnum text-ink-2">{pct(values[i])}%</span>
        </div>
      ))}
    </div>
  );
}

export { num };
