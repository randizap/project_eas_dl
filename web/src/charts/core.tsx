import { useLayoutEffect, useRef, useState, type ReactNode } from "react";

/** Lebar kontainer yang responsif (SVG digambar ulang mengikuti lebar). */
export function useWidth<T extends HTMLElement>(initial = 320) {
  const ref = useRef<T>(null);
  const [w, setW] = useState(initial);
  useLayoutEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(260, Math.floor(e.contentRect.width))));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, w] as const;
}

export type Tip = { x: number; y: number; content: ReactNode } | null;

/** Tooltip mengambang dalam kontainer relatif. */
export function TooltipBox({ tip, width }: { tip: Tip; width: number }) {
  if (!tip) return null;
  const left = Math.min(Math.max(tip.x + 14, 4), width - 230);
  return (
    <div
      role="status"
      className="pointer-events-none absolute z-30 min-w-[150px] max-w-[240px] rounded-lg border border-line-strong bg-[#202127]/95 px-3 py-2 text-xs shadow-2xl backdrop-blur"
      style={{ left, top: Math.max(tip.y - 12, 0) }}
    >
      {tip.content}
    </div>
  );
}

export function TipRow({ color, label, value, shape = "line" }: { color?: string; label: ReactNode; value: ReactNode; shape?: "line" | "dot" }) {
  return (
    <div className="flex items-center justify-between gap-4 py-0.5">
      <span className="flex items-center gap-2 text-muted">
        {color && (shape === "line"
          ? <span className="inline-block h-0.5 w-3 rounded" style={{ background: color }} />
          : <span className="inline-block h-2 w-2 rounded-full" style={{ background: color }} />)}
        {label}
      </span>
      <span className="font-semibold text-ink tnum">{value}</span>
    </div>
  );
}

export const linear = (d0: number, d1: number, r0: number, r1: number) => (v: number) =>
  r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0);

export function niceTicks(min: number, max: number, count = 5): number[] {
  const span = max - min || 1;
  const step0 = span / count;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= count) ?? step0;
  const out: number[] = [];
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

/** Sumbu x numerik + garis grid vertikal tipis. */
export function XAxis({ ticks, x, y0, y1, fmt }: { ticks: number[]; x: (v: number) => number; y0: number; y1: number; fmt: (v: number) => string }) {
  return (
    <g>
      {ticks.map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={y0} y2={y1} stroke="var(--grid)" strokeWidth={1} />
          <text x={x(t)} y={y1 + 16} textAnchor="middle" fontSize={11} fill="var(--color-muted)" fontFamily="var(--font-mono)">
            {fmt(t)}
          </text>
        </g>
      ))}
    </g>
  );
}

// sequential biru (100 -> 650) untuk heatmap
const BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281"];
export function seqBlue(t: number): string {
  if (t === null || Number.isNaN(t)) return "#26272d";
  const v = Math.max(0, Math.min(1, t));
  return BLUES[Math.min(BLUES.length - 1, Math.floor(v * BLUES.length))];
}
export function inkOn(t: number): string {
  return t >= 3 / 9 ? "#ffffff" : "#0b0b0b";
}
