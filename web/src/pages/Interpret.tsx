import { useEffect, useMemo, useRef, useState } from "react";
import { LineChart } from "../charts/charts";
import { Callout, Card, ChartFrame, DataTable, LegendItem, PageHeader, Section, Segmented, Term } from "../components/ui";
import { decodeB64Map, drawCamOverlay, loadImage } from "../lib/cam";
import { CONFIG_META, num, pct, useResults } from "../lib/data";
import type { GradcamExample, Part } from "../lib/types";

const runToCfg = (rid: string) => (rid.includes("phone") ? "phone" : "main") as "main" | "phone";

export default function Interpret() {
  const r = useResults();
  const rand = r.summary.gradcam_randomization ?? [];
  const energy = r.summary.gradcam_energy ?? [];
  const steps = useMemo(() => {
    const m = new Map<number, string>();
    rand.forEach((d) => m.set(d.step, d.randomized_upto));
    return [...m.entries()].sort((a, b) => a[0] - b[0]);
  }, [rand]);
  const [metric, setMetric] = useState<"spearman" | "ssim">("spearman");
  const series = useMemo(() => {
    const keys = [...new Set(rand.map((d) => `${d.run_id}|${d.domain}`))];
    return keys.map((k) => {
      const [rid, dom] = k.split("|");
      const c = runToCfg(rid);
      return {
        id: k, label: `${CONFIG_META[c].short} · ${dom === "test" ? "ID" : "OOD"}`, color: CONFIG_META[c].color, dashed: dom === "ood",
        points: rand.filter((d) => d.run_id === rid && d.domain === dom).sort((a, b) => a.step - b.step).map((d) => ({ x: d.step, y: d[metric] })),
      };
    });
  }, [rand, metric]);

  return (
    <div>
      <PageHeader
        n="03"
        eyebrow="Interpretasi"
        title="Model melihat lesinya, atau alatnya?"
        lead="Grad-CAM menyorot area citra yang paling mendorong prediksi. Tapi peta yang indah belum tentu bermakna, jadi setiap peta diuji dengan mengacak bobot model."
      />

      <Section id="acak" title="Uji pengacakan parameter (Adebayo dkk. 2018)" kicker="Lapisan model diacak bertahap dari kepala ke lapisan awal. Jika Grad-CAM benar-benar bergantung pada apa yang dipelajari model, kemiripan dengan peta asli harus turun mendekati nol.">
        {rand.length ? (
          <ChartFrame
            title="Kemiripan dengan peta asli saat bobot diacak bertahap"
            note="Kurva yang cepat turun ke ~0 = peta lolos uji (bergantung pada bobot terlatih). Kurva yang tetap tinggi = peta tidak layak ditafsirkan."
            controls={<Segmented label="Ukuran" value={metric} onChange={setMetric} options={[{ value: "spearman", label: "Spearman ρ" }, { value: "ssim", label: "SSIM" }]} />}
            legend={series.map((s) => <LegendItem key={s.id} color={s.color} shape="line">{s.label}{s.dashed ? " (putus-putus)" : ""}</LegendItem>)}
            table={<DataTable dense head={["Langkah", "Diacak sampai", ...series.map((s) => s.label)]}
              rows={steps.map(([st, name]) => [st, name, ...series.map((s) => num(s.points.find((p) => p.x === st)?.y ?? null, 3))])} />}
          >
            <LineChart series={series} xTicks={steps.map(([s]) => s)} xFmt={(v) => steps.find(([s]) => s === v)?.[1].replace("features.", "f") ?? String(v)}
              xLabel="diacak sampai lapisan" yDomain={[-0.2, 1]} yFmt={(v) => num(v, 1)} tipFmt={(v) => num(v, 3)} />
          </ChartFrame>
        ) : (
          <Callout tone="warn">Belum ada hasil uji pengacakan.</Callout>
        )}
        {energy.length > 0 && (
          <div className="mt-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {energy.map((e) => {
              const c = runToCfg(e.run_id);
              return (
                <Card key={`${e.run_id}-${e.domain}`} className="p-5">
                  <div className="flex items-center gap-2 font-mono text-[11px] uppercase tracking-widest text-muted">
                    <span className="h-2 w-2 rounded-full" style={{ background: CONFIG_META[c].color }} />
                    {CONFIG_META[c].short} · {e.domain === "test" ? "ID" : "OOD"}
                  </div>
                  <div className="mt-2 text-3xl font-bold tnum">{pct(e.border_energy)}%</div>
                  <div className="text-sm text-muted">
                    energi peta di <Term tip="Pita 15% di tepi citra. Di sana ada sudut gelap vignette dermatoskop, penggaris, dan latar. Area tepi menempati ±51% luas citra; nilai jauh di atas itu mengisyaratkan model melihat artefak.">pita tepi</Term>
                  </div>
                  {e.lesion_energy != null && <div className="mt-1 text-xs text-ink-2">di dalam masker lesi: {pct(e.lesion_energy)}%</div>}
                </Card>
              );
            })}
          </div>
        )}
      </Section>

      <Gallery />
    </div>
  );
}

function Gallery() {
  const r = useResults();
  const ex = r.gradcam_examples;
  const [cfg, setCfg] = useState<"main" | "phone">("main");
  const [dom, setDom] = useState<Part>("ood");
  const [step, setStep] = useState(0);
  const [alpha, setAlpha] = useState(0.6);
  const items = ex.filter((e) => runToCfg(e.run_id) === cfg && e.domain === dom);
  const stepNames = items[0]?.steps ?? [];
  if (!ex.length) return null;
  return (
    <Section id="galeri" title="Galeri: geser untuk mengacak model" kicker="Peta asli di posisi kiri slider. Geser ke kanan untuk melihat peta setelah lapisan demi lapisan diacak. Peta yang bermakna akan ‘hancur’.">
      <Card className="p-5 sm:p-6">
        <div className="flex flex-wrap items-center gap-4">
          <Segmented value={cfg} onChange={setCfg} options={(["main", "phone"] as const).map((c) => ({ value: c, label: CONFIG_META[c].short, color: CONFIG_META[c].color }))} />
          <Segmented value={dom} onChange={setDom} options={[{ value: "test", label: "HAM · ID" }, { value: "ood", label: "PAD · OOD" }]} />
          <label className="flex items-center gap-3 text-sm text-muted">
            opasitas
            <input type="range" min={0} max={1} step={0.05} value={alpha} onChange={(e) => setAlpha(+e.target.value)} className="w-28" />
          </label>
        </div>
        <div className="mt-5">
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted">Diacak sampai: <b className="font-mono text-ink">{step === 0 ? "— (peta asli)" : stepNames[step]}</b></span>
            <span className="font-mono text-xs text-muted">{step}/{Math.max(0, stepNames.length - 1)}</span>
          </div>
          <input type="range" min={0} max={Math.max(0, stepNames.length - 1)} value={step} onChange={(e) => setStep(+e.target.value)}
            className="mt-2 w-full" aria-label="Tingkat pengacakan lapisan" />
          <div className="mt-1 flex justify-between font-mono text-[10px] text-muted"><span>asli</span><span>kepala</span><span>semua lapisan</span></div>
        </div>
        <div className="mt-6 grid gap-4 grid-cols-2 sm:grid-cols-3 lg:grid-cols-6">
          {items.map((e) => <CamCard key={e.image_id + e.run_id} e={e} step={Math.min(step, e.maps.length - 1)} alpha={alpha} />)}
        </div>
        {r.meta.is_demo && <p className="mt-4 text-xs text-amber">Mode demo: model belum terlatih dan citra sintetis, jadi peta tampak acak.</p>}
      </Card>
    </Section>
  );
}

function CamCard({ e, step, alpha }: { e: GradcamExample; step: number; alpha: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const [img, setImg] = useState<HTMLImageElement | null>(null);
  useEffect(() => { loadImage("/" + e.image).then(setImg).catch(() => setImg(null)); }, [e.image]);
  useEffect(() => {
    if (!ref.current) return;
    const { data, size } = decodeB64Map(e.maps[step]);
    drawCamOverlay(ref.current, img, data, size, alpha);
  }, [img, step, alpha, e.maps]);
  const ok = e.pred === e.label;
  return (
    <figure className="rounded-xl border border-line bg-bg/50 p-2">
      <canvas ref={ref} width={224} height={224} className="w-full aspect-square rounded-lg" aria-label={`Grad-CAM untuk ${e.image_id}`} />
      <figcaption className="mt-2 px-1 text-xs">
        <div className="flex justify-between">
          <span className="text-muted">label</span><span className="font-mono text-ink">{e.label}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-muted">prediksi</span>
          <span className={ok ? "font-mono text-lime" : "font-mono text-coral"}>{e.pred ?? "–"} {e.pred ? (ok ? "✓" : "✗") : ""}</span>
        </div>
      </figcaption>
    </figure>
  );
}
