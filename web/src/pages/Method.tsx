import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { Callout, Card, DataTable, PageHeader, Section, Segmented, Term } from "../components/ui";
import { CONFIG_META, CONFIGS, num, useResults } from "../lib/data";

const TOC = [
  ["data", "Data"],
  ["kebocoran", "Cek kebocoran"],
  ["split", "Split per lesi"],
  ["model", "Tiga konfigurasi"],
  ["augmentasi", "Augmentasi ponsel"],
  ["evaluasi", "Evaluasi"],
];

export default function Method() {
  const r = useResults();
  const cfg = r.config;
  return (
    <div>
      <PageHeader
        n="01"
        eyebrow="Metode"
        title="Bagaimana kami memastikan angkanya jujur"
        lead="Setiap keputusan di bawah dibuat sebelum test set dibuka, dan semuanya bisa ditelusuri ke kode serta konfigurasi di repositori."
      />
      <nav aria-label="Daftar isi" className="sticky top-16 z-30 -mx-5 sm:-mx-8 mb-4 border-y border-line bg-bg/85 px-5 sm:px-8 py-3 backdrop-blur-xl">
        <div className="flex gap-2 overflow-x-auto">
          {TOC.map(([id, t], i) => (
            <a key={id} href={`#${id}`} className="shrink-0 rounded-full border border-line px-3 py-1 font-mono text-xs text-ink-2 hover:border-lime hover:text-lime">
              {String(i + 1).padStart(2, "0")} {t}
            </a>
          ))}
        </div>
      </nav>

      <DataSection />

      {/* -------------------------------------------------- kebocoran */}
      <LeakageSection />

      {/* -------------------------------------------------- split */}
      <SplitSection />

      {/* -------------------------------------------------- model */}
      <Section id="model" title="Tiga konfigurasi yang dibandingkan" kicker="Semuanya dilatih hanya pada HAM10000. Satu-satunya perbedaan antara model utama dan intervensi adalah augmentasinya; learning rate sama.">
        <div className="grid gap-5 lg:grid-cols-3">
          {CONFIGS.map((c) => (
            <Card key={c} className="p-6">
              <div className="flex items-center gap-2">
                <span className="h-3 w-3 rounded-full" style={{ background: CONFIG_META[c].color }} />
                <span className="font-mono text-xs uppercase tracking-widest text-muted">{CONFIG_META[c].short}</span>
              </div>
              <h3 className="mt-3 font-semibold">{CONFIG_META[c].name.split(" · ")[1]}</h3>
              <p className="mt-2 text-sm text-ink-2 leading-relaxed">{CONFIG_META[c].desc}</p>
            </Card>
          ))}
        </div>
        <div className="mt-5 grid gap-5 lg:grid-cols-2">
          <DataTable
            head={["Hiperparameter CNN", "Nilai"]}
            dense
            rows={[
              ["Arsitektur", `${cfg.cnn.arch} (torchvision, ${cfg.cnn.weights})`],
              ["Resolusi", `${cfg.data.img_size} px, cache sisi pendek ${cfg.data.cache_short_side} px`],
              ["Optimizer", `AdamW, weight decay ${cfg.cnn.weight_decay}`],
              ["Learning rate", r.tuned ? `${r.tuned.lr} (dipilih dari grid ${cfg.tune_lr_grid.join(", ")})` : cfg.cnn.lr],
              ["Jadwal", `warmup ${cfg.cnn.warmup_epochs} epoch + cosine, maks ${cfg.cnn.epochs} epoch`],
              ["Early stopping", `patience ${cfg.cnn.patience} pada val balanced accuracy`],
              ["Loss", "Cross-entropy berbobot invers frekuensi kelas"],
              ["Batch · AMP", `${cfg.cnn.batch_size} · ${cfg.cnn.amp ? "ya (fp16)" : "tidak"}`],
            ]}
          />
          <DataTable
            head={["Baseline linear probe", "Nilai"]}
            dense
            rows={[
              ["Backbone (beku)", cfg.linear_probe.backbone],
              ["Fitur", "token CLS, 384 dimensi"],
              ["Klasifikator", "Regresi logistik berbobot kelas"],
              ["C dipilih dari val", cfg.linear_probe.C_grid.join(", ")],
              ["Seed", cfg.seeds.join(", ")],
            ]}
          />
        </div>
        <div className="mt-5">
          <Callout title="Kenapa bobot pretrained umum, bukan khusus dermatologi?">
            Bobot ImageNet dan DINOv2 (LVD-142M) tidak dilatih pada citra lesi kulit, sehingga tidak ada risiko
            data uji ikut “terlihat” lewat pretraining. Sumber dan versi bobot dicatat di <code className="font-mono text-xs">weights.txt</code>.
          </Callout>
        </div>
      </Section>

      {/* -------------------------------------------------- augmentasi */}
      <Section id="augmentasi" title="Augmentasi yang meniru kamera ponsel" kicker="Rentang parameternya ditetapkan dari sifat foto ponsel sebelum eksperimen, tidak pernah dituning dengan skor PAD-UFES-20 (itu akan membocorkan data uji OOD).">
        <div className="grid gap-5 md:grid-cols-2 lg:grid-cols-4">
          {[
            ["Zoom-out", `p = ${cfg.augment.phone.zoom_out_p}, skala ${cfg.augment.phone.zoom_out_scale.join("–")}×`, "Ponsel memotret dari jarak lebih jauh: lesi tampak kecil di tengah kulit."],
            ["Warna & white balance", `jitter ${cfg.augment.phone.color_jitter.join("/")}, gain ${cfg.augment.phone.wb_gain.join("–")}`, "Pencahayaan ruangan dan white balance otomatis tiap ponsel berbeda."],
            ["Blur", `p = ${cfg.augment.phone.blur_p}, σ ${cfg.augment.phone.blur_sigma.join("–")}`, "Tangan goyang dan fokus otomatis yang meleset."],
            ["Kompresi JPEG", `p = ${cfg.augment.phone.jpeg_p}, kualitas ${cfg.augment.phone.jpeg_quality.join("–")}`, "Foto ponsel disimpan dan dikirim dalam JPEG terkompresi."],
          ].map(([t, p, d]) => (
            <Card key={t} className="p-5">
              <div className="font-semibold">{t}</div>
              <div className="mt-1 font-mono text-[11px] text-lime">{p}</div>
              <p className="mt-2 text-sm text-muted leading-relaxed">{d}</p>
            </Card>
          ))}
        </div>
        <div className="mt-5">
          <Link to="/demo" className="inline-flex items-center gap-2 rounded-full border border-lime/50 px-4 py-2 text-sm text-lime hover:bg-lime/10">
            Coba sendiri: simulasikan foto ponsel dan lihat prediksinya berubah →
          </Link>
        </div>
      </Section>

      {/* -------------------------------------------------- evaluasi */}
      <Section id="evaluasi" title="Evaluasi: skor + rentang, bukan satu angka" kicker="Test set dibuka satu kali setelah semua run selesai. Pembukaannya dikunci oleh berkas TEST_OPENED.json beserta SHA-256 setiap checkpoint.">
        <div className="grid gap-5 lg:grid-cols-3">
          <Card className="p-6">
            <div className="font-mono text-xs text-lime">METRIK UTAMA</div>
            <p className="mt-3 text-sm text-ink-2 leading-relaxed">
              <Term tip="Rerata recall per kelas. Tebakan acak di 4 kelas = 25%.">Balanced accuracy</Term> dan{" "}
              <Term tip="Rerata F1 per kelas tanpa bobot ukuran kelas.">macro-F1</Term>, ditambah recall per kelas
              (terutama melanoma), confusion matrix, dan{" "}
              <Term tip="Expected Calibration Error: seberapa jauh keyakinan model berbeda dari akurasinya. 0 = terkalibrasi sempurna.">ECE</Term>.
            </p>
          </Card>
          <Card className="p-6">
            <div className="font-mono text-xs text-lime">BOOTSTRAP KLASTER · B = {r.summary.B}</div>
            <p className="mt-3 text-sm text-ink-2 leading-relaxed">
              Satu lesi HAM bisa punya beberapa foto dan satu pasien PAD bisa punya beberapa lesi. Karena itu yang
              di-resample adalah <b className="text-ink">lesi/pasien</b>, bukan citra; bootstrap per citra memberi CI yang terlalu sempit.
            </p>
          </Card>
          <Card className="p-6">
            <div className="font-mono text-xs text-lime">PERBANDINGAN BERPASANGAN</div>
            <p className="mt-3 text-sm text-ink-2 leading-relaxed">
              Intervensi vs utama dihitung dengan resample dan seed yang sama, sehingga variasi data saling meniadakan
              dan selisihnya lebih presisi daripada membandingkan dua CI terpisah.
            </p>
          </Card>
        </div>
        <div className="mt-5 grid gap-5 lg:grid-cols-2">
          <Callout title="Uji korupsi: overlap vs non-overlap">
            Blur, JPEG, kecerahan, dan kontras dilatih langsung oleh augmentasi intervensi, jadi ketahanan terhadapnya
            bukan bukti generalisasi. Yang informatif adalah korupsi yang <b>tidak</b> dilatih (noise, pixelate).
          </Callout>
          <Callout title="Grad-CAM dengan uji pengacakan">
            Peta Grad-CAM hanya ditafsirkan bila berubah ketika bobot model diacak bertahap (Adebayo dkk. 2018).
            Peta yang tetap sama setelah bobot diacak tidak menjelaskan apa pun. <Link className="text-lime" to="/interpretasi">Lihat →</Link>
          </Callout>
        </div>
      </Section>
    </div>
  );
}

function DataSection() {
  const r = useResults();
  const classes = r.meta.classes;
  const names = r.meta.class_names;
  const ham = (c: string) => r.data_counts[c]?.ham ?? 0;
  const pad = (c: string) => r.data_counts[c]?.pad ?? 0;
  const maxN = Math.max(...classes.flatMap((c) => [ham(c), pad(c)]), 1);
  const map: Record<string, [string, string]> = { MEL: ["mel", "MEL"], NEV: ["nv", "NEV"], BCC: ["bcc", "BCC"], AK: ["akiec", "ACK"] };
  return (
    <Section id="data" title="Dua dataset, satu jenis lesi, dua jenis kamera" kicker="Kedua dataset memuat lesi yang sebagian besar sama tetapi difoto dengan alat sangat berbeda, sehingga selisih kinerja dapat dikaitkan dengan pergeseran kondisi akuisisi.">
      <div className="grid gap-5 lg:grid-cols-2">
        <Card className="p-6">
          <div className="flex items-baseline justify-between">
            <h3 className="text-xl font-bold">HAM10000</h3>
            <span className="font-mono text-xs text-muted">CC BY-NC 4.0</span>
          </div>
          <p className="mt-1 text-sm text-ink-2">Dermatoskop · pembesaran & cahaya terkontrol · 10.015 citra, 7 kelas</p>
          <div className="mt-4 font-mono text-xs text-lime">PERAN: LATIH · VALIDASI · UJI IN-DISTRIBUTION</div>
        </Card>
        <Card className="p-6" tone="coral">
          <div className="flex items-baseline justify-between">
            <h3 className="text-xl font-bold">PAD-UFES-20</h3>
            <span className="font-mono text-xs text-muted">CC BY 4.0</span>
          </div>
          <p className="mt-1 text-sm text-ink-2">Berbagai kamera ponsel · cahaya, jarak & latar bebas · 2.298 citra, 6 kelas</p>
          <div className="mt-4 font-mono text-xs text-coral">PERAN: UJI OUT-OF-DISTRIBUTION SAJA</div>
        </Card>
      </div>
      <div className="mt-5">
        <Card className="p-6">
          <h3 className="font-semibold">Empat kelas yang berpadanan · jumlah citra yang dipakai</h3>
          <p className="mt-1 text-sm text-muted">Kelas lain dibuang (HAM: bkl, df, vasc · PAD: SCC, SEK). Angka dihitung langsung dari berkas metadata.</p>
          <div className="mt-5 space-y-4">
            {classes.map((c) => (
              <div key={c} className="grid gap-2 sm:grid-cols-[11rem_1fr] sm:items-center">
                <div>
                  <div className="font-mono text-sm text-ink">{c}</div>
                  <div className="text-xs text-muted">{names[c]} · <span className="font-mono">{map[c]?.[0]} / {map[c]?.[1]}</span></div>
                </div>
                <div className="space-y-1.5">
                  {[["HAM", ham(c), "var(--color-ink-2)"], ["PAD", pad(c), "var(--color-coral)"]].map(([d, n, col]) => (
                    <div key={d as string} className="grid grid-cols-[2.5rem_1fr_3.5rem] items-center gap-2 text-xs">
                      <span className="font-mono text-muted">{d}</span>
                      <div className="h-2.5 rounded bg-white/5">
                        <div className="h-full rounded" style={{ width: `${((n as number) / maxN) * 100}%`, background: col as string }} />
                      </div>
                      <span className="text-right tnum text-ink-2">{(n as number).toLocaleString("id-ID")}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>
      <div className="mt-5">
        <Callout tone="warn" title="Dua ketidakcocokan label (dibahas di Keterbatasan)">
          (1) <code className="font-mono text-xs">akiec</code> di HAM mencakup keratosis aktinik <i>dan</i> penyakit Bowen,
          sedangkan <code className="font-mono text-xs">ACK</code> di PAD hanya keratosis aktinik. (2) Hanya ±24% citra ACK dan
          NEV di PAD yang terkonfirmasi biopsi, jadi kualitas label kedua domain berbeda.
        </Callout>
      </div>
    </Section>
  );
}

function LeakageSection() {
  const r = useResults();
  const L = r.summary.leakage;
  const [kind, setKind] = useState<string>("all");
  const pairs = useMemo(
    () => r.leakage_pairs.filter((p) => kind === "all" || p.kind === kind).slice(0, 40),
    [r.leakage_pairs, kind],
  );
  if (!L) return null;
  const kinds = Object.keys(L.pairs_by_kind ?? {});
  const kindLabel: Record<string, string> = {
    cross_dataset: "HAM ↔ PAD",
    diff_lesion_ham: "lesi berbeda (HAM)",
    diff_lesion_pad: "lesi berbeda (PAD)",
    same_lesion: "lesi sama",
  };
  return (
    <Section id="kebocoran" title="Cek kebocoran: mencari kembaran tersembunyi" kicker="Dua foto yang nyaris identik di train dan test membuat model cukup mengingat, tidak perlu mengenali. Kami mencarinya di dalam dan antar dataset dengan dua cara yang saling melengkapi.">
      <div className="grid gap-4 grid-cols-2 lg:grid-cols-4">
        {[
          ["Citra diperiksa", L.n_images],
          ["Lesi HAM → grup", `${L.ham_lesions} → ${L.ham_groups_after_merge}`],
          ["PAD dikeluarkan dari OOD", L.pad_excluded_from_ood],
          ["Pasangan konflik label", L.pairs_label_conflict],
        ].map(([t, v]) => (
          <Card key={t as string} className="p-5">
            <div className="font-mono text-[11px] uppercase tracking-widest text-muted">{t}</div>
            <div className="mt-2 text-3xl font-bold tnum">{String(v ?? "–")}</div>
          </Card>
        ))}
      </div>
      <div className="mt-5 grid gap-5 lg:grid-cols-2">
        <Card className="p-6">
          <div className="font-mono text-xs text-lime">pHASH · HAMMING ≤ {L.thresholds?.phash_max_hamming}</div>
          <p className="mt-2 text-sm text-ink-2 leading-relaxed">
            <Term tip="Perceptual hash: sidik jari 64-bit dari versi kecil citra. Dua citra yang hanya beda kompresi, ukuran, atau sedikit warna punya hash hampir sama.">Perceptual hash</Term>{" "}
            menangkap salinan yang hanya beda ukuran atau kompresi.
          </p>
          <div className="mt-4 font-mono text-xs text-lime">KOSINUS DINOv2 · ≥ {num(L.thresholds?.cosine_min)}</div>
          <p className="mt-2 text-sm text-ink-2 leading-relaxed">
            Embedding DINOv2 menangkap foto lesi yang sama dari sudut atau pencahayaan sedikit berbeda.
            Lesi berbeda yang punya citra kembar digabung (union-find) menjadi satu grup sebelum split.
          </p>
          {r.meta.is_demo && (
            <p className="mt-3 text-xs text-amber">Mode demo: bobot DINOv2 acak, sehingga hanya pHash yang aktif (ambang kosinus &gt; 1).</p>
          )}
        </Card>
        {r.has_contact_sheet ? (
          <Card className="p-3">
            <div className="max-h-[26rem] overflow-y-auto rounded-xl">
              <img src="/media/contact_sheet.png" alt="Contact sheet pasangan kandidat duplikat untuk diperiksa manual" className="w-full" loading="lazy" />
            </div>
            <p className="px-3 py-2 text-xs text-muted">Contact sheet untuk pemeriksaan manual (proposal §3.1).</p>
          </Card>
        ) : <div />}
      </div>
      {r.leakage_pairs.length > 0 && (
        <div className="mt-5">
          <div className="mb-3 flex flex-wrap items-center gap-3">
            <Segmented
              label="Jenis pasangan"
              value={kind}
              onChange={setKind}
              options={[{ value: "all", label: "semua" }, ...kinds.map((k) => ({ value: k, label: `${kindLabel[k] ?? k} · ${L.pairs_by_kind?.[k]}` }))]}
            />
          </div>
          <DataTable
            dense
            head={["Citra A", "Citra B", "Jenis", "Metode", "Hamming", "Kosinus", "Label"]}
            rows={pairs.map((p) => [
              <span className="font-mono text-xs">{p.image_id_a}</span>,
              <span className="font-mono text-xs">{p.image_id_b}</span>,
              kindLabel[p.kind] ?? p.kind,
              p.method,
              p.phash_dist ?? "–",
              num(p.cosine, 3),
              p.label_conflict ? <span className="text-coral">{p.label_a} ≠ {p.label_b}</span> : p.label_a,
            ])}
          />
        </div>
      )}
    </Section>
  );
}

function SplitSection() {
  const r = useResults();
  const seeds = Object.keys(r.splits).filter((k) => k.startsWith("lesion_")).sort();
  const [key, setKey] = useState(seeds[0] ?? "");
  const sp = r.splits[key] ?? r.summary.split_counts;
  const classes = r.meta.classes;
  const parts: [string, Record<string, number>, string][] = [
    ["Latih", sp.counts.train, "var(--color-ink-2)"],
    ["Validasi", sp.counts.val, "var(--color-ink-2)"],
    ["Uji ID", sp.counts.test, "var(--color-lime)"],
    ["Uji OOD", sp.ood_counts, "var(--color-coral)"],
  ];
  const tot = (o: Record<string, number>) => classes.reduce((a, c) => a + (o[c] ?? 0), 0);
  return (
    <Section id="split" title="Split per lesi, bukan per citra" kicker="HAM10000 dibagi 70/15/15 per grup lesi. Test set tetap untuk semua run (seed 2026); train/val dipartisi ulang per seed supaya varians data ikut terukur."
      aside={seeds.length > 1 ? <Segmented label="Seed" value={key} onChange={setKey} options={seeds.map((s) => ({ value: s, label: s.replace("lesion_seed", "") }))} /> : undefined}>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {parts.map(([name, o, col]) => {
          const t = tot(o);
          return (
            <Card key={name} className="p-5">
              <div className="flex items-baseline justify-between">
                <span className="font-mono text-xs uppercase tracking-widest" style={{ color: col }}>{name}</span>
                <span className="text-2xl font-bold tnum">{t.toLocaleString("id-ID")}</span>
              </div>
              <div className="mt-4 flex h-3 overflow-hidden rounded-full gap-[2px]">
                {classes.map((c, i) => (
                  <div key={c} title={`${c}: ${o[c] ?? 0}`} style={{ width: `${((o[c] ?? 0) / Math.max(1, t)) * 100}%`, background: col, opacity: 1 - i * 0.2 }} />
                ))}
              </div>
              <div className="mt-3 grid grid-cols-4 gap-1 text-center">
                {classes.map((c) => (
                  <div key={c}>
                    <div className="font-mono text-[10px] text-muted">{c}</div>
                    <div className="text-sm tnum text-ink-2">{o[c] ?? 0}</div>
                  </div>
                ))}
              </div>
            </Card>
          );
        })}
      </div>
      <div className="mt-5 grid gap-5 lg:grid-cols-2">
        <Callout tone="good" title={`Lesi yang ada di train dan test: ${sp.lesion_overlap_train_test}`}>
          Kode split memuat <code className="font-mono text-xs">assert</code> bahwa tidak ada grup lesi di dua sisi.
          {sp.ood_excluded > 0 && <> {sp.ood_excluded} citra PAD dikeluarkan dari uji OOD karena kembar dengan citra HAM.</>}
        </Callout>
        <Callout title="RQ3 memakai pembanding yang sengaja ‘salah’">
          Untuk mengukur dampak kebocoran, kami juga melatih model dengan split acak <b>per citra</b>, sehingga foto
          lain dari lesi yang sama bisa muncul di train dan test. <Link className="text-lime" to="/hasil#rq3">Hasilnya →</Link>
        </Callout>
      </div>
    </Section>
  );
}
