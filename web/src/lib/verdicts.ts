// Kalimat kesimpulan dibuat otomatis dari angka + CI, supaya teks tidak pernah
// bertentangan dengan data. Aturan: klaim "berbeda" hanya bila CI 95% tidak memuat nol.
import { ciExcludesZero, ciText, pct, signedPct } from "./data";
import type { Results } from "./types";

export interface Verdict {
  headline: string;
  detail: string;
  significant: boolean | null;
}

export function rq1(r: Results): Verdict {
  const c = r.summary.configs.main;
  if (!c) return { headline: "Belum ada data", detail: "", significant: null };
  const g = c.gap.bal_acc;
  const sig = ciExcludesZero(g);
  const id = (c.test.bal_acc as { mean: number }).mean;
  const ood = (c.ood.bal_acc as { mean: number }).mean;
  const drop = g.mean ?? 0;
  return {
    headline: sig
      ? drop > 0
        ? `Turun ${pct(drop)} poin saat pindah ke foto ponsel`
        : `Naik ${pct(-drop)} poin saat pindah ke foto ponsel`
      : `Selisih ${signedPct(-drop)} poin, belum bisa dibedakan dari nol`,
    detail: `Model utama: balanced accuracy ${pct(id)}% di HAM10000 test (ID) vs ${pct(ood)}% di PAD-UFES-20 (OOD). Celah ID−OOD ${pct(drop)} poin, CI 95% ${ciText(g)}.`,
    significant: sig,
  };
}

export function rq2(r: Results): Verdict {
  const p = r.summary.paired_phone_minus_main;
  if (!p) return { headline: "Belum ada data", detail: "", significant: null };
  const gc = p.gap_change_bal_acc;
  const ood = p.ood.bal_acc;
  const id = p.test.bal_acc;
  const sig = ciExcludesZero(gc);
  const narrowed = (gc.mean ?? 0) < 0;
  const head = sig
    ? narrowed
      ? `Augmentasi ponsel mempersempit celah ${pct(-(gc.mean ?? 0))} poin`
      : `Augmentasi ponsel justru memperlebar celah ${pct(gc.mean ?? 0)} poin`
    : `Perubahan celah ${signedPct(gc.mean)} poin, CI memuat nol`;
  return {
    headline: head,
    detail: `Selisih berpasangan (intervensi − utama): OOD ${signedPct(ood.mean)} poin ${ciText(ood, true)}, ID ${signedPct(id.mean)} poin ${ciText(id, true)}. Hasil negatif tetap dilaporkan.`,
    significant: sig,
  };
}

export function rq3(r: Results): Verdict {
  const m = r.summary.rq3?.main ?? r.summary.rq3?.baseline_lp;
  if (!m) return { headline: "Belum ada data", detail: "", significant: null };
  const inf = m.inflation_bal_acc;
  const sig = ciExcludesZero(inf);
  const which = r.summary.rq3?.main ? "model utama" : "baseline";
  return {
    headline: sig
      ? (inf.mean ?? 0) > 0
        ? `Split per citra menggelembungkan skor ${pct(inf.mean)} poin`
        : `Split per citra justru ${pct(-(inf.mean ?? 0))} poin lebih rendah`
      : `Inflasi ${signedPct(inf.mean)} poin, CI memuat nol`,
    detail: `Balanced accuracy ${which}: ${pct(m.image.bal_acc.mean)}% (split acak per citra) vs ${pct(m.lesion.bal_acc.mean)}% (split per lesi). Inflasi CI 95% ${ciText(inf, true)}.`,
    significant: sig,
  };
}
