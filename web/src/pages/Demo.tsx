import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { LineChart, ProbBars } from "../charts/charts";
import { Callout, Card, cx, LegendItem, PageHeader, Segmented } from "../components/ui";
import { drawCamOverlay, loadImage, normalizeCam } from "../lib/cam";
import { CONFIG_META, pct, useResults } from "../lib/data";
import { loadModel, predict, type Prediction } from "../lib/infer";
import { centerCrop, DEFAULT_PARAMS, randomParams, simulate, type PhoneParams } from "../lib/phone";
import type { ModelInfo } from "../lib/types";

type ModelState = { status: "idle" | "loading" | "ready" | "error"; progress: number; error?: string };

const SLIDERS: { key: keyof PhoneParams; label: string; min: number; max: number; step: number; fmt: (v: number) => string; hint: string }[] = [
  { key: "zoom", label: "Jarak (zoom-out)", min: 0.4, max: 1, step: 0.01, fmt: (v) => (v >= 0.999 ? "mati" : `${Math.round(v * 100)}%`), hint: "Lesi tampak lebih kecil karena kamera lebih jauh" },
  { key: "brightness", label: "Kecerahan", min: 0.6, max: 1.4, step: 0.01, fmt: (v) => `×${v.toFixed(2)}`, hint: "Cahaya ruangan" },
  { key: "contrast", label: "Kontras", min: 0.6, max: 1.4, step: 0.01, fmt: (v) => `×${v.toFixed(2)}`, hint: "" },
  { key: "saturation", label: "Saturasi", min: 0.6, max: 1.4, step: 0.01, fmt: (v) => `×${v.toFixed(2)}`, hint: "" },
  { key: "warmth", label: "White balance", min: -0.15, max: 0.15, step: 0.01, fmt: (v) => (v === 0 ? "netral" : v > 0 ? `hangat +${v.toFixed(2)}` : `dingin ${v.toFixed(2)}`), hint: "Gain kanal merah/biru" },
  { key: "blur", label: "Blur (σ px)", min: 0, max: 3, step: 0.1, fmt: (v) => (v < 0.05 ? "mati" : v.toFixed(1)), hint: "Tangan goyang / fokus meleset" },
  { key: "jpeg", label: "Kualitas JPEG", min: 10, max: 100, step: 1, fmt: (v) => (v >= 100 ? "mati" : String(v)), hint: "Kompresi saat disimpan/dikirim" },
];

const SWEEPS: { key: keyof PhoneParams; label: string; values: number[] }[] = [
  { key: "blur", label: "Blur", values: [0, 0.5, 1, 1.5, 2, 2.5, 3] },
  { key: "zoom", label: "Zoom-out", values: [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4] },
  { key: "jpeg", label: "JPEG", values: [100, 80, 60, 40, 25, 15, 10] },
  { key: "brightness", label: "Kecerahan", values: [0.6, 0.75, 0.9, 1, 1.1, 1.25, 1.4] },
];

export default function Demo() {
  const r = useResults();
  const classes = r.meta.classes;
  const models = r.models;
  const [ms, setMs] = useState<Record<string, ModelState>>({});
  const [source, setSource] = useState<HTMLImageElement | null>(null);
  const [sourceName, setSourceName] = useState("");
  const [params, setParams] = useState<PhoneParams>(DEFAULT_PARAMS);
  const [sim, setSim] = useState<HTMLCanvasElement | null>(null);
  const [base, setBase] = useState<HTMLCanvasElement | null>(null);
  const [preds, setPreds] = useState<Record<string, Prediction>>({});
  const [camClass, setCamClass] = useState<number | "top">("top");
  const [alpha, setAlpha] = useState(0.55);
  const [dragOver, setDragOver] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const samples = useMemo(() => {
    const seen = new Set<string>();
    return r.gradcam_examples.filter((e) => (seen.has(e.image) ? false : (seen.add(e.image), true))).slice(0, 8);
  }, [r.gradcam_examples]);

  // ----------------------------------------------------- model
  const ensureModels = useCallback(() => {
    models.forEach((m) => {
      setMs((s) => (s[m.id]?.status === "ready" || s[m.id]?.status === "loading" ? s : { ...s, [m.id]: { status: "loading", progress: 0 } }));
      loadModel("/" + m.file, (f) => setMs((s) => ({ ...s, [m.id]: { ...(s[m.id] ?? { status: "loading" }), status: s[m.id]?.status === "ready" ? "ready" : "loading", progress: f } })))
        .then(() => setMs((s) => ({ ...s, [m.id]: { status: "ready", progress: 1 } })))
        .catch((e) => setMs((s) => ({ ...s, [m.id]: { status: "error", progress: 0, error: String(e) } })));
    });
  }, [models]);
  const allReady = models.length > 0 && models.every((m) => ms[m.id]?.status === "ready");

  // ----------------------------------------------------- input
  const pickImage = useCallback((im: HTMLImageElement, name: string) => {
    setSource(im);
    setSourceName(name);
    setBase(centerCrop(im));
    ensureModels();
  }, [ensureModels]);

  const onFile = (f: File | undefined) => {
    if (!f || !f.type.startsWith("image/")) return;
    const url = URL.createObjectURL(f);
    loadImage(url).then((im) => pickImage(im, f.name));
  };

  // ----------------------------------------------------- simulasi + prediksi
  useEffect(() => {
    if (!base) return;
    let alive = true;
    const t = setTimeout(async () => {
      const s = await simulate(base, params);
      if (!alive) return;
      setSim(s);
    }, 60);
    return () => { alive = false; clearTimeout(t); };
  }, [base, params]);

  useEffect(() => {
    if (!sim || !allReady) return;
    let alive = true;
    (async () => {
      const out: Record<string, Prediction> = {};
      for (const m of models) out[m.id] = await predict(await loadModel("/" + m.file), sim);
      if (alive) setPreds(out);
    })();
    return () => { alive = false; };
  }, [sim, allReady, models]);

  const isDefault = JSON.stringify(params) === JSON.stringify(DEFAULT_PARAMS);

  return (
    <div>
      <PageHeader
        n="04"
        eyebrow="Coba model"
        title={<>Ubah foto dermatoskop jadi <span className="text-lime">“foto ponsel”</span>, lihat prediksinya bergeser</>}
        lead="Kedua EfficientNet-B0 (utama dan intervensi) berjalan langsung di browser Anda lewat ONNX Runtime Web. Citra yang Anda pilih tidak pernah dikirim ke server mana pun."
      />
      <Callout tone="warn" title="Bukan alat diagnosis">
        Model ini adalah artefak penelitian tentang pergeseran domain, dilatih pada 4 kelas saja dan tidak divalidasi klinis.
        Hasilnya tidak boleh dipakai untuk menilai lesi kulit sungguhan.
      </Callout>

      {models.length === 0 ? (
        <div className="mt-6"><Callout>Model ONNX belum diekspor. Jalankan tahap <code className="font-mono">export</code> di notebook Colab.</Callout></div>
      ) : (
        <div className="mt-8 grid gap-6 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.05fr)]">
          {/* ================================================= kiri: input + kontrol */}
          <div className="space-y-6">
            <Card className="p-5">
              <div className="flex items-center justify-between">
                <h2 className="font-semibold">1 · Pilih citra</h2>
                {sourceName && <span className="truncate pl-3 font-mono text-xs text-muted">{sourceName}</span>}
              </div>
              <div
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={(e) => { e.preventDefault(); setDragOver(false); onFile(e.dataTransfer.files[0]); }}
                className={cx("mt-4 grid place-items-center rounded-xl border-2 border-dashed px-4 py-6 text-center transition-colors",
                  dragOver ? "border-lime bg-lime/5" : "border-line-strong")}
              >
                <p className="text-sm text-ink-2">Seret citra ke sini, atau</p>
                <button onClick={() => fileRef.current?.click()} className="mt-2 rounded-full bg-lime px-4 py-1.5 text-sm font-semibold text-bg hover:brightness-110">
                  Unggah citra
                </button>
                <input ref={fileRef} type="file" accept="image/*" className="hidden" onChange={(e) => onFile(e.target.files?.[0])} />
                <p className="mt-2 text-xs text-muted">Diproses di perangkat Anda: resize 256 → crop tengah 224 px.</p>
              </div>
              {samples.length > 0 && (
                <>
                  <div className="mt-4 font-mono text-[11px] uppercase tracking-widest text-muted">atau contoh dari test set</div>
                  <div className="mt-2 grid grid-cols-4 gap-2 sm:grid-cols-8 xl:grid-cols-4 2xl:grid-cols-8">
                    {samples.map((s) => (
                      <button key={s.image} title={`${s.image_id} · label ${s.label}`}
                        onClick={() => loadImage("/" + s.image).then((im) => pickImage(im, `${s.image_id} (label ${s.label}, ${s.domain === "test" ? "HAM" : "PAD"})`))}
                        className="group relative overflow-hidden rounded-lg border border-line hover:border-lime">
                        <img src={"/" + s.image} alt={`contoh ${s.label}`} className="aspect-square w-full object-cover" loading="lazy" />
                        <span className={cx("absolute left-1 top-1 rounded px-1 font-mono text-[9px]", s.domain === "test" ? "bg-bg/80 text-ink-2" : "bg-coral/90 text-bg")}>
                          {s.domain === "test" ? "HAM" : "PAD"}
                        </span>
                        <span className="absolute bottom-1 right-1 rounded bg-bg/80 px-1 font-mono text-[9px] text-ink">{s.label}</span>
                      </button>
                    ))}
                  </div>
                </>
              )}
            </Card>

            <Card className="p-5">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <h2 className="font-semibold">2 · Simulasikan kamera ponsel</h2>
                <div className="flex gap-2">
                  <button onClick={() => setParams(randomParams(r.config))} className="rounded-full border border-lime/60 px-3 py-1 font-mono text-xs text-lime hover:bg-lime/10">⟳ acak seperti augmentasi</button>
                  <button onClick={() => setParams(DEFAULT_PARAMS)} disabled={isDefault} className="rounded-full border border-line px-3 py-1 font-mono text-xs text-ink-2 hover:text-ink disabled:opacity-40">reset</button>
                </div>
              </div>
              <p className="mt-1 text-xs text-muted">Rentang mengikuti <code className="font-mono">augment.phone</code> di konfigurasi, diperlebar sedikit agar efeknya terlihat.</p>
              <div className="mt-4 grid gap-x-6 gap-y-4 sm:grid-cols-2">
                {SLIDERS.map((s) => (
                  <label key={s.key} className="block">
                    <span className="flex justify-between text-sm">
                      <span className="text-ink-2">{s.label}</span>
                      <span className={cx("font-mono text-xs tnum", params[s.key] !== DEFAULT_PARAMS[s.key] ? "text-lime" : "text-muted")}>{s.fmt(params[s.key])}</span>
                    </span>
                    <input type="range" min={s.min} max={s.max} step={s.step} value={params[s.key]} className="mt-1.5 w-full"
                      onChange={(e) => setParams((p) => ({ ...p, [s.key]: +e.target.value }))} aria-label={s.label} />
                    {s.hint && <span className="text-[11px] text-muted">{s.hint}</span>}
                  </label>
                ))}
              </div>
            </Card>

            <Card className="p-5">
              <h2 className="font-semibold">Citra yang dilihat model</h2>
              <div className="mt-4 grid grid-cols-2 gap-4">
                <Preview canvas={base} label="asli (crop 224)" />
                <Preview canvas={sim} label={isDefault ? "tanpa simulasi" : "setelah simulasi ponsel"} accent={!isDefault} />
              </div>
            </Card>
          </div>

          {/* ================================================= kanan: hasil */}
          <div className="space-y-6">
            <Card className="p-5">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <h2 className="font-semibold">3 · Prediksi kedua model</h2>
                <div className="flex flex-wrap items-center gap-3">
                  <Segmented value={String(camClass)} onChange={(v) => setCamClass(v === "top" ? "top" : +v)}
                    options={[{ value: "top", label: "CAM: teratas" }, ...classes.map((c, i) => ({ value: String(i), label: c }))]} />
                  <label className="flex items-center gap-2 text-xs text-muted">opasitas
                    <input type="range" min={0} max={1} step={0.05} value={alpha} onChange={(e) => setAlpha(+e.target.value)} className="w-20" />
                  </label>
                </div>
              </div>
              {!source && <p className="mt-6 text-sm text-muted">Pilih atau unggah citra untuk mulai. Model (±16 MB per model) diunduh sekali lalu disimpan di cache browser.</p>}
              <div className="mt-5 grid gap-5 md:grid-cols-2">
                {models.map((m) => (
                  <ModelPanel key={m.id} m={m} state={ms[m.id]} pred={preds[m.id]} sim={sim} classes={classes} camClass={camClass} alpha={alpha} started={!!source} />
                ))}
              </div>
            </Card>
            {allReady && base && <Sweep base={base} params={params} models={models} classes={classes} preds={preds} />}
          </div>
        </div>
      )}
    </div>
  );
}

function Preview({ canvas, label, accent }: { canvas: HTMLCanvasElement | null; label: string; accent?: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const ctx = c.getContext("2d")!;
    ctx.clearRect(0, 0, c.width, c.height);
    if (canvas) ctx.drawImage(canvas, 0, 0);
  }, [canvas]);
  return (
    <figure>
      <canvas ref={ref} width={224} height={224} className={cx("w-full aspect-square rounded-xl border bg-bg", accent ? "border-lime/50" : "border-line")} />
      <figcaption className={cx("mt-2 text-center font-mono text-[11px]", accent ? "text-lime" : "text-muted")}>{label}</figcaption>
    </figure>
  );
}

function ModelPanel({ m, state, pred, sim, classes, camClass, alpha, started }: {
  m: ModelInfo; state?: ModelState; pred?: Prediction; sim: HTMLCanvasElement | null; classes: string[];
  camClass: number | "top"; alpha: number; started: boolean;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const meta = CONFIG_META[m.id];
  const top = pred ? pred.probs.indexOf(Math.max(...pred.probs)) : -1;
  const k = camClass === "top" ? top : camClass;
  useEffect(() => {
    if (!ref.current) return;
    drawCamOverlay(ref.current, sim, pred && k >= 0 ? normalizeCam(pred.cams[k]) : null, pred?.camSize ?? 7, alpha);
  }, [sim, pred, k, alpha]);
  return (
    <div className="rounded-xl border border-line bg-bg/40 p-4">
      <div className="flex items-center gap-2">
        <span className="h-2.5 w-2.5 rounded-full" style={{ background: meta.color }} />
        <span className="font-semibold">{meta.short}</span>
        <span className="ml-auto font-mono text-[10px] text-muted">{m.run_id}</span>
      </div>
      {state?.status === "loading" && (
        <div className="mt-4">
          <div className="h-1.5 rounded-full bg-white/5"><div className="h-full rounded-full bg-lime transition-[width]" style={{ width: `${state.progress * 100}%` }} /></div>
          <p className="mt-2 font-mono text-[11px] text-muted">mengunduh model… {Math.round(state.progress * 100)}% dari {m.size_mb} MB</p>
        </div>
      )}
      {state?.status === "error" && <p className="mt-3 text-sm text-coral">Gagal memuat model: {state.error}</p>}
      <canvas ref={ref} width={224} height={224} className="mt-4 w-full aspect-square rounded-lg bg-bg" aria-label={`Grad-CAM ${meta.short}`} />
      {pred ? (
        <>
          <div className="mt-4 flex items-baseline justify-between">
            <span className="text-sm text-muted">prediksi</span>
            <span className="text-2xl font-bold">{classes[top]} <span className="text-base font-normal text-ink-2 tnum">{pct(pred.probs[top])}%</span></span>
          </div>
          <div className="mt-3"><ProbBars labels={classes} values={pred.probs} color={meta.color} highlight={top} /></div>
          <p className="mt-3 font-mono text-[10px] text-muted">inferensi {pred.ms.toFixed(0)} ms · CAM kelas {classes[k]}</p>
        </>
      ) : started && state?.status === "ready" ? (
        <p className="mt-3 text-sm text-muted">menghitung…</p>
      ) : null}
    </div>
  );
}

/** Sapuan: jalankan model pada satu parameter yang divariasikan, plot probabilitas kelas teratas. */
function Sweep({ base, params, models, classes, preds }: { base: HTMLCanvasElement; params: PhoneParams; models: ModelInfo[]; classes: string[]; preds: Record<string, Prediction> }) {
  const [which, setWhich] = useState<keyof PhoneParams>("blur");
  const [cls, setCls] = useState<number | null>(null);
  const [data, setData] = useState<Record<string, number[][]> | null>(null);
  const [busy, setBusy] = useState(false);
  const sw = SWEEPS.find((s) => s.key === which)!;
  const ref0 = preds.main ?? preds[models[0].id];
  const target = cls ?? (ref0 ? ref0.probs.indexOf(Math.max(...ref0.probs)) : 0);

  const run = async () => {
    setBusy(true);
    const out: Record<string, number[][]> = {};
    for (const m of models) out[m.id] = [];
    for (const v of sw.values) {
      const c = await simulate(base, { ...params, [which]: v });
      for (const m of models) out[m.id].push((await predict(await loadModel("/" + m.file), c)).probs);
    }
    setData(out);
    setBusy(false);
  };
  useEffect(() => { setData(null); }, [which, base]);

  const series = data ? models.map((m) => ({
    id: m.id, label: CONFIG_META[m.id].short, color: CONFIG_META[m.id].color,
    points: sw.values.map((_, i) => ({ x: i, y: data[m.id][i][target] })),
  })) : [];
  return (
    <Card className="p-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="font-semibold">4 · Uji sapuan: seberapa tahan?</h2>
          <p className="mt-1 text-sm text-muted">Satu efek dinaikkan bertahap, efek lain tetap seperti slider. Model yang tahan punya kurva datar.</p>
        </div>
        <button onClick={run} disabled={busy} className="rounded-full bg-lime px-4 py-1.5 text-sm font-semibold text-bg disabled:opacity-50">
          {busy ? "menjalankan…" : "▶ jalankan sapuan"}
        </button>
      </div>
      <div className="mt-4 flex flex-wrap gap-3">
        <Segmented value={which} onChange={setWhich} options={SWEEPS.map((s) => ({ value: s.key, label: s.label }))} />
        <Segmented value={String(target)} onChange={(v) => setCls(+v)} label="prob. kelas" options={classes.map((c, i) => ({ value: String(i), label: c }))} />
      </div>
      {data ? (
        <div className="mt-4">
          <div className="mb-2 flex gap-4 text-sm text-ink-2">{series.map((s) => <LegendItem key={s.id} color={s.color} shape="line">{s.label}</LegendItem>)}</div>
          <LineChart series={series} xTicks={sw.values.map((_, i) => i)} xFmt={(i) => String(sw.values[i])} xLabel={sw.label}
            yDomain={[0, 1]} yLabel={`P(${classes[target]})`} tipFmt={(v) => `${pct(v)}%`} height={230} />
        </div>
      ) : (
        <p className="mt-4 text-sm text-muted">Tekan ▶ untuk menjalankan {sw.values.length} × {models.length} inferensi.</p>
      )}
    </Card>
  );
}
