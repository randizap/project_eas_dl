import { Link } from "react-router-dom";
import { Card, Eyebrow, StatTile, Term, Verdict } from "../components/ui";
import { ciText, pct, useResults } from "../lib/data";
import { rq1, rq2, rq3 } from "../lib/verdicts";

const CHECKS = [
  { n: "01", t: "Cek kebocoran", d: "pHash + kosinus DINOv2 di dalam & antar dataset; duplikat lintas lesi digabung.", c: "#ff7a59", to: "/metode#kebocoran" },
  { n: "02", t: "Split per lesi", d: "70/15/15 per lesion_id. Satu lesi hanya ada di satu sisi.", c: "#c6f26b", to: "/metode#split" },
  { n: "03", t: "Baseline", d: "Linear probe di atas fitur DINOv2 yang dibekukan.", c: "#5cc8ff", to: "/metode#model" },
  { n: "04", t: "Skor + rentang", d: "3 seed, CI 95% bootstrap klaster, test dibuka sekali.", c: "#a78bfa", to: "/hasil" },
  { n: "05", t: "Bisa diulang", d: "Satu perintah, log semua run, SHA-256 bobot.", c: "#ff6fae", to: "/eksperimen" },
];

export default function Home() {
  const r = useResults();
  const main = r.summary.configs.main;
  const id = main?.test.bal_acc as { mean: number; ci: [number, number] } | undefined;
  const ood = main?.ood.bal_acc as { mean: number; ci: [number, number] } | undefined;
  const verdicts = [
    { q: "RQ1", title: "Seberapa besar penurunan dari dermatoskop ke ponsel?", v: rq1(r), to: "/hasil#rq1" },
    { q: "RQ2", title: "Apakah augmentasi ala ponsel memperkecil penurunan itu?", v: rq2(r), to: "/hasil#rq2" },
    { q: "RQ3", title: "Seberapa menyesatkan split acak per citra?", v: rq3(r), to: "/hasil#rq3" },
  ];

  return (
    <div>
      {/* ------------------------------------------------ hero */}
      <section className="relative grid gap-12 pt-14 pb-16 sm:pt-24 lg:grid-cols-[1.25fr_1fr] lg:items-center fade-in">
        <div>
          <Eyebrow>EAS · Proyek Akhir · Deep Learning</Eyebrow>
          <h1 className="mt-6 text-5xl sm:text-6xl lg:text-7xl font-extrabold tracking-tight leading-[1.02]">
            Akurat di klinik.
            <br />
            <span className="text-lime">Di ponsel?</span>
          </h1>
          <p className="mt-6 max-w-xl text-lg text-ink-2 leading-relaxed">
            Kami melatih model klasifikasi lesi kulit <b className="text-ink">hanya</b> pada citra dermatoskop (HAM10000),
            lalu mengujinya pada foto ponsel dari layanan dermatologi di Brasil (PAD-UFES-20). Pertanyaannya sederhana:
            apakah model belajar dari <b className="text-ink">lesinya</b>, atau dari <b className="text-ink">alat yang memotretnya</b>?
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <Link to="/hasil" className="rounded-full bg-lime px-5 py-2.5 font-semibold text-bg hover:brightness-110">Lihat hasil →</Link>
            <Link to="/demo" className="rounded-full border border-line-strong px-5 py-2.5 text-ink hover:border-lime">Coba model di browser</Link>
          </div>
          <div className="mt-8 flex flex-wrap gap-2 font-mono text-xs text-muted">
            <span className="rounded-full border border-line px-3 py-1">Track D · Pergeseran Data</span>
            <span className="rounded-full border border-line px-3 py-1">4 kelas: MEL · NEV · BCC · AK</span>
            <span className="rounded-full border border-line px-3 py-1">Colab T4 · 3 seed</span>
          </div>
        </div>
        <HeroVisual />
      </section>

      {/* ------------------------------------------------ angka utama */}
      <section className="py-8">
        <div className="mb-6">
          <Eyebrow n="01">Model yang sama, dua angka</Eyebrow>
          <h2 className="mt-3 text-3xl font-bold tracking-tight">Yang dinilai adalah angka kedua.</h2>
        </div>
        <div className="grid gap-5 md:grid-cols-2">
          <StatTile
            label="Diuji di HAM10000 test · dermatoskop"
            value={<>{pct(id?.mean)}<span className="text-3xl">%</span></>}
            sub={<>Balanced accuracy model utama · CI 95% {ciText(main?.test.bal_acc as never)}</>}
          >
            <Meter v={id?.mean ?? 0} color="var(--color-ink-2)" />
          </StatTile>
          <StatTile
            tone="coral"
            label="Diuji di PAD-UFES-20 · foto ponsel"
            value={<>{pct(ood?.mean)}<span className="text-3xl">%</span></>}
            sub={<>Data yang tidak pernah dilihat saat latih maupun memilih model · CI 95% {ciText(main?.ood.bal_acc as never)}</>}
          >
            <Meter v={ood?.mean ?? 0} color="var(--color-coral)" />
          </StatTile>
        </div>
        <p className="mt-4 text-sm text-muted">
          <Term tip="Rerata recall keempat kelas. Tidak tertipu kelas mayoritas: model yang selalu menebak 'nevus' hanya mendapat 25%.">
            Balanced accuracy
          </Term>{" "}
          dipakai karena kelas sangat tidak seimbang (nevus ≈ 77% data latih). Tebakan acak = 25%.
        </p>
      </section>

      {/* ------------------------------------------------ RQ */}
      <section className="py-12">
        <div className="mb-6">
          <Eyebrow n="02">Tiga pertanyaan penelitian</Eyebrow>
          <h2 className="mt-3 text-3xl font-bold tracking-tight">Jawaban singkat, lengkap dengan rentang ketidakpastian</h2>
          <p className="mt-2 text-ink-2 max-w-3xl">
            Kalimat di bawah dibuat otomatis dari angka. Sebuah efek hanya disebut “nyata” bila interval kepercayaan 95%-nya tidak memuat nol.
          </p>
        </div>
        <div className="grid gap-5 lg:grid-cols-3">
          {verdicts.map(({ q, title, v, to }) => (
            <Link key={q} to={to} className="group">
              <Card className="h-full p-6 transition-colors group-hover:border-lime/40">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-sm font-semibold text-lime">{q}</span>
                  <Verdict ok={v.significant}>{v.significant === null ? "n/a" : v.significant ? "CI ≠ 0" : "CI memuat 0"}</Verdict>
                </div>
                <h3 className="mt-3 text-ink-2 text-sm">{title}</h3>
                <p className="mt-4 text-xl font-semibold leading-snug">{v.headline}</p>
                <p className="mt-3 text-sm text-muted leading-relaxed">{v.detail}</p>
                <span className="mt-5 inline-block font-mono text-xs text-lime opacity-70 group-hover:opacity-100">detail →</span>
              </Card>
            </Link>
          ))}
        </div>
      </section>

      {/* ------------------------------------------------ pipeline */}
      <section className="py-12">
        <div className="mb-8">
          <Eyebrow n="03">Lima pemeriksaan di sepanjang pipeline</Eyebrow>
          <h2 className="mt-3 text-3xl font-bold tracking-tight">Skor tinggi belum tentu skor jujur</h2>
        </div>
        <ol className="grid gap-4 md:grid-cols-5">
          {CHECKS.map((c, i) => (
            <li key={c.n} className="relative">
              {i < CHECKS.length - 1 && (
                <span className="hidden md:block absolute top-8 left-[calc(50%+2.25rem)] right-[calc(-50%+2.25rem)] h-px" style={{ background: `linear-gradient(90deg, ${c.c}, ${CHECKS[i + 1].c})`, opacity: 0.5 }} />
              )}
              <Link to={c.to} className="group block text-center">
                <span className="mx-auto grid h-16 w-16 place-items-center rounded-full border-2 text-xl font-bold" style={{ borderColor: c.c, color: c.c, boxShadow: `0 0 24px -6px ${c.c}` }}>
                  {c.n}
                </span>
                <Card className="mt-4 p-4 text-left h-[calc(100%-5rem)] group-hover:border-line-strong">
                  <div className="font-semibold">{c.t}</div>
                  <p className="mt-1.5 text-sm text-muted leading-relaxed">{c.d}</p>
                </Card>
              </Link>
            </li>
          ))}
        </ol>
      </section>

      {/* ------------------------------------------------ alur kerja */}
      <section className="py-12">
        <Card className="p-6 sm:p-10 dot-grid">
          <Eyebrow n="04">Dari Colab ke situs ini</Eyebrow>
          <div className="mt-6 grid gap-6 md:grid-cols-4">
            {[
              ["Google Colab (T4)", "Notebook menjalankan seluruh pipeline: data → kebocoran → split → latih → evaluasi."],
              ["Ekspor", "Tahap export menulis results.json, 2 model ONNX, dan contoh Grad-CAM ke web/public."],
              ["GitHub", "Hasil di-commit ke repositori, termasuk log semua run dan hash bobot."],
              ["Vercel", "Setiap push memicu build ulang; model berjalan di browser Anda dengan ONNX Runtime Web."],
            ].map(([t, d], i) => (
              <div key={t}>
                <div className="font-mono text-xs text-lime">0{i + 1}</div>
                <div className="mt-2 font-semibold">{t}</div>
                <p className="mt-1 text-sm text-muted leading-relaxed">{d}</p>
              </div>
            ))}
          </div>
        </Card>
      </section>
    </div>
  );
}

function Meter({ v, color }: { v: number; color: string }) {
  return (
    <div className="mt-6 h-2.5 rounded-full bg-white/5 overflow-hidden" aria-hidden>
      <div className="h-full rounded-full" style={{ width: `${Math.max(0, Math.min(1, v)) * 100}%`, background: color }} />
    </div>
  );
}

/** Ilustrasi: grid patch yang "menyala" seperti peta atensi (dekoratif). */
function HeroVisual() {
  const cells = Array.from({ length: 100 }, (_, i) => {
    const x = i % 10, y = Math.floor(i / 10);
    const d = Math.hypot(x - 5.2, y - 4.3);
    const d2 = Math.hypot(x - 2.6, y - 6.8);
    return Math.max(0, 1 - d / 2.6, 0.75 - d2 / 2.2);
  });
  const color = (v: number) =>
    v > 0.75 ? "#ff7a59" : v > 0.5 ? "#e0a84e" : v > 0.25 ? "#7a9a3e" : "#2a2a44";
  return (
    <Card className="p-6 sm:p-8" >
      <div className="flex justify-between font-mono text-xs text-muted">
        <span>peta atensi · patch 7×7 → 10×10</span>
        <span className="text-lime">ilustrasi</span>
      </div>
      <div className="mt-6 grid grid-cols-10 gap-1.5" aria-hidden>
        {cells.map((v, i) => (
          <div key={i} className="aspect-square rounded-md transition-colors" style={{ background: color(v), opacity: 0.35 + v * 0.65 }} />
        ))}
      </div>
      <div className="mt-6 flex items-center gap-3 font-mono text-xs text-muted">
        rendah
        <span className="h-1.5 flex-1 rounded-full bg-gradient-to-r from-[#2a2a44] via-[#e0a84e] to-[#ff7a59]" />
        tinggi
      </div>
      <p className="mt-4 text-sm text-muted">
        Grad-CAM menunjukkan bagian citra yang paling memengaruhi prediksi. Kami memeriksa apakah fokusnya ada di lesi,
        bukan di sudut vignette dermatoskop. <Link to="/interpretasi" className="text-lime">Lihat peta aslinya →</Link>
      </p>
    </Card>
  );
}
