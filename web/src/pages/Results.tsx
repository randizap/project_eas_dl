import { useMemo, useState } from "react";
import { Dumbbell, Forest, Heatmap, LineChart, type ForestRow } from "../charts/charts";
import { TipRow } from "../charts/core";
import { Callout, Card, ChartFrame, DataTable, LegendItem, PageHeader, Section, Segmented, Term, Verdict } from "../components/ui";
import { ciText, CONFIG_META, CONFIGS, METRIC_LABEL, num, PART_LABEL, pct, signedPct, useResults } from "../lib/data";
import type { ConfigId, Part, Stat } from "../lib/types";
import { rq1, rq2, rq3 } from "../lib/verdicts";

const st = (v: unknown) => v as Stat;

export default function ResultsPage() {
  const r = useResults();
  return (
    <div>
      <PageHeader
        n="02"
        eyebrow="Hasil"
        title="Apa yang terjadi saat model keluar dari klinik"
        lead={<>Semua angka: rerata 3 seed dengan interval kepercayaan {Math.round(r.summary.ci_level * 100)}% dari bootstrap klaster (B = {r.summary.B}). Arahkan kursor ke titik untuk detail, atau klik <span className="font-mono text-sm">▦ tabel</span> untuk melihat angka mentah.</>}
      />
      <RQ1 />
      <PerClass />
      <RQ2 />
      <Corruption />
      <RQ3 />
    </div>
  );
}

function VerdictBar({ v }: { v: { headline: string; detail: string; significant: boolean | null } }) {
  return (
    <Card tone="lime" className="mb-5 p-5">
      <div className="flex flex-wrap items-center gap-3">
        <Verdict ok={v.significant}>{v.significant ? "CI tidak memuat 0" : "CI memuat 0"}</Verdict>
        <span className="text-lg font-semibold">{v.headline}</span>
      </div>
      <p className="mt-2 text-sm text-ink-2">{v.detail}</p>
    </Card>
  );
}

// ======================================================================= RQ1
function RQ1() {
  const r = useResults();
  const [metric, setMetric] = useState<"bal_acc" | "macro_f1">("bal_acc");
  const rows = CONFIGS.map((c) => {
    const e = r.summary.configs[c];
    const a = st(e.test[metric]);
    const b = st(e.ood[metric]);
    return {
      id: c,
      label: (
        <span>
          <span className="block text-ink">{CONFIG_META[c].short}</span>
          <span className="block text-[11px] text-muted">{CONFIG_META[c].name.split(" · ")[1]}</span>
        </span>
      ),
      short: CONFIG_META[c].short,
      color: CONFIG_META[c].color,
      a: { v: a.mean, lo: a.ci[0], hi: a.ci[1], label: `${CONFIG_META[c].short} · ${PART_LABEL.test}` },
      b: { v: b.mean, lo: b.ci[0], hi: b.ci[1], label: `${CONFIG_META[c].short} · ${PART_LABEL.ood}` },
    };
  });
  const table = (
    <DataTable
      dense
      head={["Konfigurasi", "ID", "OOD", "Celah ID−OOD", "sd antar seed (ID/OOD)"]}
      rows={CONFIGS.map((c) => {
        const e = r.summary.configs[c];
        const a = st(e.test[metric]), b = st(e.ood[metric]), g = e.gap[metric];
        return [CONFIG_META[c].short, `${pct(a.mean)} ${ciText(a)}`, `${pct(b.mean)} ${ciText(b)}`, `${pct(g.mean)} ${ciText(g)}`,
          `${pct(a.sd_seeds)} / ${pct(b.sd_seeds)}`];
      })}
    />
  );
  return (
    <Section id="rq1" title="RQ1 · Seberapa besar penurunan dari dermatoskop ke ponsel?" kicker="Titik penuh = HAM10000 test (in-distribution). Cincin = PAD-UFES-20 (out-of-distribution). Garis tipis = CI 95%. Label Δ = perubahan dari ID ke OOD.">
      <VerdictBar v={rq1(r)} />
      <ChartFrame
        title={`${METRIC_LABEL[metric]} · ID vs OOD`}
        note="Skala 0–100%. Tebakan acak untuk balanced accuracy = 25%."
        controls={<Segmented label="Metrik" value={metric} onChange={setMetric} options={[{ value: "bal_acc", label: "Balanced acc." }, { value: "macro_f1", label: "Macro-F1" }]} />}
        legend={<>
          {CONFIGS.map((c) => <LegendItem key={c} color={CONFIG_META[c].color}>{CONFIG_META[c].short}</LegendItem>)}
          <span className="text-muted">·</span>
          <LegendItem color="var(--color-ink-2)">ID (penuh)</LegendItem>
          <LegendItem color="var(--color-ink-2)" shape="ring">OOD (cincin)</LegendItem>
        </>}
        table={table}
      >
        <Dumbbell rows={rows} />
      </ChartFrame>
    </Section>
  );
}

// ======================================================================= per-class & confusion
function PerClass() {
  const r = useResults();
  const classes = r.meta.classes;
  const [cfg, setCfg] = useState<ConfigId>("main");
  const [part, setPart] = useState<Part>("ood");
  const nSeeds = r.summary.configs[cfg][part].per_seed.length;
  const [seed, setSeed] = useState(0);
  const rowsDef = CONFIGS.flatMap((c) => (["test", "ood"] as Part[]).map((p) => ({ c, p })));
  const recallVals = rowsDef.map(({ c, p }) => classes.map((k) => st(r.summary.configs[c][p][`recall_${k}`]).mean));
  const conf = r.summary.configs[cfg][part].per_seed[Math.min(seed, nSeeds - 1)]?.confusion ?? [];
  const confN = conf.map((row) => {
    const s = row.reduce((a, b) => a + b, 0);
    return row.map((v) => (s ? v / s : null));
  });
  return (
    <Section id="per-kelas" title="Recall per kelas: siapa yang paling dirugikan?" kicker="Recall melanoma paling penting secara klinis: melanoma yang terlewat jauh lebih berbahaya daripada nevus yang salah dicurigai.">
      <div className="grid gap-5 xl:grid-cols-[1.1fr_1fr]">
        <ChartFrame
          title="Recall per kelas · rerata 3 seed"
          note="Biru lebih gelap = recall lebih tinggi. Arahkan ke sel untuk CI."
          table={
            <DataTable dense head={["Konfigurasi · domain", ...classes, "ECE"]}
              rows={rowsDef.map(({ c, p }, i) => [`${CONFIG_META[c].short} · ${p === "test" ? "ID" : "OOD"}`,
                ...recallVals[i].map((v, j) => `${pct(v)} ${ciText(st(r.summary.configs[c][p][`recall_${classes[j]}`]))}`),
                num(st(r.summary.configs[c][p].ece).mean, 3)])} />
          }
        >
          <Heatmap
            rows={rowsDef.map(({ c, p }) => (
              <span className="flex items-center gap-2">
                <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: CONFIG_META[c].color }} />
                {CONFIG_META[c].short} · <span className={p === "ood" ? "text-coral" : ""}>{p === "test" ? "ID" : "OOD"}</span>
              </span>
            ))}
            cols={classes}
            values={recallVals}
            tipLabel={(i, j) => {
              const { c, p } = rowsDef[i];
              const s = st(r.summary.configs[c][p][`recall_${classes[j]}`]);
              return (<>
                <div className="mb-1 font-semibold text-ink">{CONFIG_META[c].short} · {p === "test" ? "ID" : "OOD"} · {classes[j]}</div>
                <TipRow label="recall" value={`${pct(s.mean)}%`} />
                <TipRow label="CI 95%" value={`${pct(s.ci[0])} – ${pct(s.ci[1])}`} />
              </>);
            }}
          />
        </ChartFrame>
        <ChartFrame
          title="Confusion matrix (dinormalisasi per baris)"
          note="Baris = label benar, kolom = prediksi. Diagonal = recall."
          controls={<>
            <Segmented value={cfg} onChange={setCfg} options={CONFIGS.map((c) => ({ value: c, label: CONFIG_META[c].short, color: CONFIG_META[c].color }))} />
            <Segmented value={part} onChange={setPart} options={[{ value: "test", label: "ID" }, { value: "ood", label: "OOD" }]} />
            {nSeeds > 1 && <Segmented label="seed" value={String(seed)} onChange={(v) => setSeed(+v)} options={Array.from({ length: nSeeds }, (_, i) => ({ value: String(i), label: String(i) }))} />}
          </>}
          table={<DataTable dense head={["benar \\ prediksi", ...classes]} rows={conf.map((row, i) => [classes[i], ...row.map(String)])} />}
        >
          <Heatmap
            rows={classes.map((c) => <span className="font-mono">{c}</span>)}
            cols={classes}
            values={confN}
            rowLabelW={60}
            tipLabel={(i, j) => (<>
              <div className="mb-1 font-semibold text-ink">benar {classes[i]} → prediksi {classes[j]}</div>
              <TipRow label="jumlah citra" value={conf[i][j]} />
              <TipRow label="porsi baris" value={`${pct(confN[i][j])}%`} />
            </>)}
          />
        </ChartFrame>
      </div>
    </Section>
  );
}

// ======================================================================= RQ2
function RQ2() {
  const r = useResults();
  const p = r.summary.paired_phone_minus_main;
  const classes = r.meta.classes;
  const row = (part: Part, key: string, label: string): ForestRow => {
    const s = p[part][key];
    return { id: `${part}-${key}`, label, sub: part === "test" ? "HAM test · ID" : "PAD-UFES-20 · OOD", v: s?.mean ?? null, lo: s?.ci[0] ?? null, hi: s?.ci[1] ?? null,
      color: part === "ood" ? "var(--color-coral)" : "var(--color-ink-2)" };
  };
  const rows: ForestRow[] = [
    row("ood", "bal_acc", "Balanced accuracy"),
    row("ood", "macro_f1", "Macro-F1"),
    row("ood", "recall_MEL", "Recall melanoma"),
    row("test", "bal_acc", "Balanced accuracy"),
    row("test", "macro_f1", "Macro-F1"),
    { id: "gap", label: "Perubahan celah ID−OOD", sub: "negatif = celah menyempit", v: p.gap_change_bal_acc.mean, lo: p.gap_change_bal_acc.ci[0], hi: p.gap_change_bal_acc.ci[1], color: "var(--color-lime)" },
  ];
  const all = (["ood", "test"] as Part[]).flatMap((part) => ["bal_acc", "macro_f1", ...classes.map((c) => `recall_${c}`)].map((k) => ({ part, k, s: p[part][k] })));
  return (
    <Section id="rq2" title="RQ2 · Apakah augmentasi ala ponsel menolong?" kicker="Selisih berpasangan intervensi − model utama, dengan resample bootstrap dan seed yang sama. Belah ketupat penuh = CI tidak memuat nol.">
      <VerdictBar v={rq2(r)} />
      <ChartFrame
        title="Intervensi − utama (poin persentase)"
        note="Ke kanan = intervensi lebih baik. Untuk ‘perubahan celah’, ke kiri = celah menyempit (yang diharapkan)."
        legend={<>
          <LegendItem color="var(--color-coral)">OOD</LegendItem>
          <LegendItem color="var(--color-ink-2)">ID (biaya di data asal)</LegendItem>
          <LegendItem color="var(--color-lime)">perubahan celah</LegendItem>
        </>}
        table={<DataTable dense head={["Metrik", "Domain", "Selisih", "CI 95%", "CI ≠ 0?"]}
          rows={[...all.map(({ part, k, s }) => [k.replace("recall_", "recall "), part === "test" ? "ID" : "OOD", signedPct(s?.mean), ciText(s, true), s?.ci_excludes_zero ? "ya" : "tidak"]),
            ...(["bal_acc", "macro_f1"] as const).map((m) => [`perubahan celah ${m}`, "ID−OOD", signedPct(p[`gap_change_${m}`].mean), ciText(p[`gap_change_${m}`], true), "–"])]} />}
      >
        <Forest rows={rows} />
      </ChartFrame>
    </Section>
  );
}

// ======================================================================= Korupsi
function Corruption() {
  const r = useResults();
  const [group, setGroup] = useState<"non" | "ov">("non");
  // baris "clean" adalah acuan (severity 0) dan sudah ada di summary.corruption.clean_bal_acc
  const detail = useMemo(() => r.corruption_detail.filter((d) => d.corruption !== "clean"), [r.corruption_detail]);
  const corrNames = useMemo(() => [...new Set(detail.map((d) => d.corruption))], [detail]);
  const sevs = useMemo(() => [...new Set(detail.map((d) => d.severity))].sort((a, b) => a - b), [detail]);
  const [corr, setCorr] = useState<string>("all");
  if (!detail.length || !r.summary.corruption) return null;
  const clean = r.summary.corruption.clean_bal_acc;
  const ov = group === "ov";
  const pick = detail.filter((d) => d.overlap_with_phone_aug === ov && (corr === "all" || d.corruption === corr));
  const series = CONFIGS.map((c) => ({
    id: c, label: CONFIG_META[c].short, color: CONFIG_META[c].color,
    points: [{ x: 0, y: clean[c] ?? null }, ...sevs.map((s) => {
      const v = pick.filter((d) => d.config === c && d.severity === s).map((d) => d.mean);
      return { x: s, y: v.length ? v.reduce((a, b) => a + b, 0) / v.length : null };
    })],
  }));
  const groupCorrs = corrNames.filter((n) => detail.some((d) => d.corruption === n && d.overlap_with_phone_aug === ov));
  const heatRows = CONFIGS.flatMap((c) => groupCorrs.map((n) => ({ c, n })));
  return (
    <Section id="korupsi" title="Ketahanan terhadap korupsi sintetis" kicker="Citra HAM test dirusak dengan 9 korupsi bergaya ImageNet-C pada severity 1–5. Korupsi yang juga dipakai sebagai augmentasi intervensi dipisahkan, karena ketahanan terhadapnya ‘dilatih langsung’.">
      <div className="grid gap-5 xl:grid-cols-2">
        <ChartFrame
          title="Balanced accuracy vs tingkat korupsi"
          note="Severity 0 = citra bersih. Rerata atas jenis korupsi dan seed (tanpa CI; lihat Keterbatasan)."
          controls={<>
            <Segmented value={group} onChange={(v) => { setGroup(v); setCorr("all"); }} options={[{ value: "non", label: "Non-overlap (informatif)" }, { value: "ov", label: "Overlap dgn aug. ponsel" }]} />
            <select value={corr} onChange={(e) => setCorr(e.target.value)} aria-label="Jenis korupsi"
              className="rounded-lg border border-line bg-bg px-3 py-1.5 text-sm text-ink-2">
              <option value="all">semua jenis</option>
              {groupCorrs.map((n) => <option key={n} value={n}>{n.replace(/_/g, " ")}</option>)}
            </select>
          </>}
          legend={CONFIGS.map((c) => <LegendItem key={c} color={CONFIG_META[c].color} shape="line">{CONFIG_META[c].short}</LegendItem>)}
          table={<DataTable dense head={["Konfigurasi", "bersih", ...sevs.map((s) => `sev ${s}`)]} rows={series.map((s) => [s.label, ...s.points.map((p) => pct(p.y))])} />}
        >
          <LineChart series={series} xTicks={[0, ...sevs]} xLabel="severity" yDomain={[0, 1]} xFmt={(v) => (v === 0 ? "bersih" : String(v))} tipFmt={(v) => `${pct(v)}%`} />
        </ChartFrame>
        <ChartFrame title="Rincian per jenis korupsi (rerata severity)" note="Baris dikelompokkan per konfigurasi.">
          <Heatmap
            rowLabelW={190}
            rows={heatRows.map(({ c, n }) => (
              <span className="flex items-center gap-2"><span className="h-2 w-2 shrink-0 rounded-full" style={{ background: CONFIG_META[c].color }} />{n.replace(/_/g, " ")}</span>
            ))}
            cols={sevs.map((s) => `sev ${s}`)}
            values={heatRows.map(({ c, n }) => sevs.map((s) => detail.find((d) => d.config === c && d.corruption === n && d.severity === s)?.mean ?? null))}
          />
        </ChartFrame>
      </div>
    </Section>
  );
}

// ======================================================================= RQ3
function RQ3() {
  const r = useResults();
  const entries = (["baseline_lp", "main"] as ConfigId[]).filter((c) => r.summary.rq3[c]);
  const rows: ForestRow[] = entries.flatMap((c) => {
    const e = r.summary.rq3[c]!;
    return (["bal_acc", "macro_f1"] as const).map((m) => ({
      id: `${c}-${m}`, label: `${CONFIG_META[c].short} · ${METRIC_LABEL[m]}`, sub: `per citra ${pct(e.image[m].mean)} vs per lesi ${pct(e.lesion[m].mean)}`,
      v: e[`inflation_${m}`].mean, lo: e[`inflation_${m}`].ci[0], hi: e[`inflation_${m}`].ci[1], color: CONFIG_META[c].color,
    }));
  });
  return (
    <Section id="rq3" title="RQ3 · Seberapa menyesatkan split acak per citra?" kicker="Model yang sama dilatih dua kali: dengan split per lesi (benar) dan split acak per citra (bocor). Selisih skor test-nya adalah ‘inflasi’ akibat kebocoran.">
      <VerdictBar v={rq3(r)} />
      <div className="grid gap-5 xl:grid-cols-[1.3fr_1fr]">
        <ChartFrame title="Inflasi skor: split per citra − split per lesi" note="Ke kanan = split per citra tampak lebih bagus dari seharusnya.">
          <Forest rows={rows} />
        </ChartFrame>
        <Card className="p-6">
          <h3 className="font-semibold">Mekanismenya: lesi yang sudah ‘dilihat’</h3>
          <p className="mt-1 text-sm text-muted">Pada split per citra (seed pertama), citra test dipisah menurut apakah lesinya juga muncul di train.</p>
          <div className="mt-5 space-y-5">
            {entries.map((c) => {
              const e = r.summary.rq3[c]!.image;
              const seen = e.bal_acc_lesion_seen_in_train, uns = e.bal_acc_lesion_unseen;
              return (
                <div key={c}>
                  <div className="flex items-center gap-2 text-sm"><span className="h-2 w-2 rounded-full" style={{ background: CONFIG_META[c].color }} />{CONFIG_META[c].short}</div>
                  <div className="mt-2 grid grid-cols-2 gap-3">
                    <div className="rounded-xl border border-line p-3">
                      <div className="font-mono text-[10px] uppercase tracking-widest text-coral">lesi ada di train</div>
                      <div className="mt-1 text-2xl font-bold tnum">{pct(seen?.value)}%</div>
                      <div className="text-xs text-muted">n = {seen?.n ?? 0} citra</div>
                    </div>
                    <div className="rounded-xl border border-line p-3">
                      <div className="font-mono text-[10px] uppercase tracking-widest text-lime">lesi baru</div>
                      <div className="mt-1 text-2xl font-bold tnum">{pct(uns?.value)}%</div>
                      <div className="text-xs text-muted">n = {uns?.n ?? 0} citra</div>
                    </div>
                  </div>
                  <div className="mt-2 text-xs text-muted">Model split-per-citra di OOD: <span className="text-ink-2 tnum">{pct(r.summary.rq3[c]!.image_ood_bal_acc)}%</span> — skor yang ‘bocor’ tidak menolong di lapangan.</div>
                </div>
              );
            })}
          </div>
        </Card>
      </div>
      <div className="mt-5">
        <Callout title="Catatan statistik">
          Split per citra dan per lesi punya test set yang berbeda, sehingga CI inflasi dihitung dari dua bootstrap
          independen per <Term tip="Satu lesi bisa difoto beberapa kali. Lesi adalah unit independen terkecil di HAM10000.">lesi</Term>, bukan berpasangan.
        </Callout>
      </div>
    </Section>
  );
}
