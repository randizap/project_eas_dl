import { useMemo, useState } from "react";
import { LineChart } from "../charts/charts";
import { Callout, Card, ChartFrame, cx, DataTable, LegendItem, PageHeader, Section, Segmented } from "../components/ui";
import { num, pct, repoUrl, useResults } from "../lib/data";

export default function Experiments() {
  const r = useResults();
  const log = r.runs_log;
  const [status, setStatus] = useState<"all" | "ok" | "failed">("all");
  const [q, setQ] = useState("");
  const rows = useMemo(
    () => log.filter((x) => (status === "all" || x.status === status) && (!q || x.run_id.toLowerCase().includes(q.toLowerCase()))),
    [log, status, q],
  );
  const failed = log.filter((x) => x.status === "failed").length;
  const minutes = log.reduce((a, x) => a + (x.train_minutes ?? 0), 0);
  const gpus = [...new Set(log.map((x) => x.gpu).filter(Boolean))];

  return (
    <div>
      <PageHeader
        n="05"
        eyebrow="Bisa diulang"
        title="Setiap run tercatat, termasuk yang gagal"
        lead="Log ini ditulis otomatis oleh pipeline (runs/log.csv). Tidak ada run yang dihapus, dan pemilihan model hanya memakai skor validasi."
      />
      <div className="grid gap-4 grid-cols-2 lg:grid-cols-4">
        {[
          ["Total run", log.length],
          ["Run gagal", failed],
          ["Waktu latih total", `${num(minutes / 60, 1)} jam`],
          ["Perangkat", gpus.join(", ") || "–"],
        ].map(([t, v]) => (
          <Card key={t as string} className="p-5">
            <div className="font-mono text-[11px] uppercase tracking-widest text-muted">{t}</div>
            <div className="mt-2 text-2xl font-bold tnum truncate" title={String(v)}>{v}</div>
          </Card>
        ))}
      </div>

      <Curves />

      <Section id="log" title="Log run" kicker="Status, waktu latih, epoch terbaik, dan skor validasi setiap run.">
        <div className="mb-3 flex flex-wrap items-center gap-3">
          <Segmented value={status} onChange={setStatus} options={[{ value: "all", label: `semua · ${log.length}` }, { value: "ok", label: "ok" }, { value: "failed", label: `gagal · ${failed}` }]} />
          <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="cari run_id…" aria-label="Cari run"
            className="rounded-lg border border-line bg-bg px-3 py-1.5 text-sm text-ink placeholder:text-muted focus:border-lime focus:outline-none" />
        </div>
        <DataTable
          dense
          head={["run_id", "tahap", "status", "seed", "split", "menit", "epoch terbaik", "val bal. acc", "val macro-F1", "waktu (UTC)", "catatan"]}
          rows={rows.map((x) => [
            <span className="font-mono text-xs">{x.run_id}</span>,
            x.stage,
            <span className={cx("font-mono text-xs", x.status === "ok" ? "text-lime" : "text-coral")}>{x.status}</span>,
            x.seed ?? "–",
            x.split_mode ?? "–",
            num(x.train_minutes, 1),
            x.best_epoch ?? "–",
            pct(x.val_bal_acc),
            pct(x.val_macro_f1),
            <span className="whitespace-nowrap font-mono text-xs">{x.timestamp_utc?.replace("T", " ").replace("Z", "")}</span>,
            <span className="block max-w-[22rem] truncate text-xs" title={x.notes ?? ""}>{x.notes}</span>,
          ])}
        />
      </Section>

      <Section id="tuning" title="Tuning learning rate" kicker="Satu seed, model utama saja, dipilih dari balanced accuracy validasi HAM10000. Intervensi memakai lr yang sama.">
        {r.tuned ? (
          <div className="grid gap-4 sm:grid-cols-3">
            {Object.entries(r.tuned.val_bal_acc_per_lr).map(([lr, v]) => (
              <Card key={lr} tone={+lr === r.tuned!.lr ? "lime" : undefined} className="p-5">
                <div className="font-mono text-xs text-muted">lr = {lr}</div>
                <div className="mt-2 text-3xl font-bold tnum">{pct(v)}%</div>
                <div className="text-xs text-muted">val balanced accuracy {+lr === r.tuned!.lr && <span className="text-lime">· terpilih</span>}</div>
              </Card>
            ))}
          </div>
        ) : <Callout>Belum ada hasil tuning.</Callout>}
      </Section>

      <Section id="integritas" title="Integritas: test dibuka sekali, bobot ber-hash" kicker="Berkas TEST_OPENED.json ditulis sebelum prediksi test pertama. SHA-256 setiap checkpoint memungkinkan penilai memverifikasi bobot yang dipakai.">
        <div className="grid gap-5 lg:grid-cols-2">
          <Card className="p-6">
            <div className="font-mono text-xs text-lime">TEST SET</div>
            {r.meta.test_opened ? (
              <>
                <p className="mt-2 text-sm text-ink-2">Dibuka pada <b className="text-ink font-mono">{r.meta.test_opened.opened_utc}</b>
                  {r.meta.test_opened.forced_reopen ? <span className="text-coral"> · dibuka ulang paksa (tercatat)</span> : " · pembukaan pertama"}</p>
                <div className="mt-4 max-h-72 overflow-auto rounded-lg border border-line">
                  {Object.entries(r.meta.test_opened.checkpoints).map(([k, v]) => (
                    <div key={k} className="flex justify-between gap-4 border-b border-line px-3 py-1.5 text-xs last:border-0">
                      <span className="font-mono text-ink-2">{k}</span>
                      <span className="font-mono text-muted truncate" title={v}>{v.slice(0, 16)}…</span>
                    </div>
                  ))}
                </div>
              </>
            ) : <p className="mt-2 text-sm text-muted">Test set belum dibuka.</p>}
          </Card>
          <Card className="p-6">
            <div className="font-mono text-xs text-lime">MODEL DI SITUS INI (ONNX)</div>
            <div className="mt-3 space-y-3">
              {r.models.map((m) => (
                <div key={m.id} className="rounded-lg border border-line p-3 text-xs">
                  <div className="flex justify-between"><span className="text-ink">{m.label}</span><span className="text-muted">{m.size_mb} MB</span></div>
                  <div className="mt-1 font-mono text-muted break-all">onnx sha256 {m.sha256}</div>
                  <div className="font-mono text-muted break-all">ckpt sha256 {m.checkpoint_sha256}</div>
                  <div className="mt-1 text-muted">selisih maks. vs PyTorch: <span className="font-mono text-ink-2">{m.max_abs_diff_vs_torch.toExponential(1)}</span></div>
                </div>
              ))}
            </div>
            <div className="mt-5 font-mono text-xs text-lime">LINGKUNGAN</div>
            <div className="mt-2 grid grid-cols-[6rem_1fr] gap-y-1 text-xs">
              {Object.entries(r.meta.environment).map(([k, v]) => (
                <div key={k} className="contents"><span className="text-muted">{k}</span><span className="font-mono text-ink-2 break-all">{v}</span></div>
              ))}
            </div>
            {repoUrl() && <a href={repoUrl("/blob/main/weights.txt")} target="_blank" rel="noreferrer" className="mt-4 inline-block text-sm text-lime">weights.txt di GitHub ↗</a>}
          </Card>
        </div>
      </Section>
    </div>
  );
}

function Curves() {
  const r = useResults();
  const ids = Object.keys(r.histories).sort();
  const [sel, setSel] = useState<string[]>(() => ids.filter((i) => /lesion_s0$/.test(i)).slice(0, 3));
  const palette = ["var(--s1)", "var(--s2)", "var(--s3)"];
  const chosen = sel.filter((s) => r.histories[s]).slice(0, 3);
  const maxEp = Math.max(1, ...chosen.flatMap((s) => r.histories[s].map((h) => h.epoch)));
  const ticks = Array.from({ length: maxEp + 1 }, (_, i) => i).filter((i) => maxEp < 16 || i % 2 === 0);
  const mk = (key: "val_bal_acc" | "train_loss") => chosen.map((s, i) => ({
    id: s, label: s, color: palette[i], points: r.histories[s].map((h) => ({ x: h.epoch, y: h[key] })),
  }));
  const toggle = (id: string) => setSel((cur) => (cur.includes(id) ? cur.filter((c) => c !== id) : [...cur, id].slice(-3)));
  if (!ids.length) return null;
  return (
    <Section id="kurva" title="Kurva training" kicker="Pilih hingga 3 run untuk dibandingkan. Checkpoint terbaik = epoch dengan val balanced accuracy tertinggi.">
      <div className="mb-4 flex flex-wrap gap-2">
        {ids.map((id) => {
          const i = chosen.indexOf(id);
          return (
            <button key={id} onClick={() => toggle(id)} aria-pressed={i >= 0}
              className={cx("flex items-center gap-2 rounded-full border px-3 py-1 font-mono text-xs", i >= 0 ? "border-line-strong text-ink bg-white/5" : "border-line text-muted hover:text-ink")}>
              <span className="h-2 w-2 rounded-full" style={{ background: i >= 0 ? palette[i] : "var(--axis)" }} />{id}
            </button>
          );
        })}
      </div>
      <div className="grid gap-5 lg:grid-cols-2">
        <ChartFrame title="Val balanced accuracy per epoch" legend={chosen.map((s, i) => <LegendItem key={s} color={palette[i]} shape="line">{s}</LegendItem>)}>
          <LineChart series={mk("val_bal_acc")} xTicks={ticks} xLabel="epoch" yDomain={[0, 1]} tipFmt={(v) => `${pct(v)}%`} />
        </ChartFrame>
        <ChartFrame title="Train loss per epoch" note="Grafik terpisah, bukan sumbu ganda.">
          <LineChart series={mk("train_loss")} xTicks={ticks} xLabel="epoch" yFmt={(v) => num(v, 2)} tipFmt={(v) => num(v, 3)} />
        </ChartFrame>
      </div>
    </Section>
  );
}
