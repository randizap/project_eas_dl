import { Callout, Card, PageHeader, Section } from "../components/ui";
import { colabUrl, repoUrl, useResults } from "../lib/data";

const REFS = [
  "P. Tschandl, C. Rosendahl, H. Kittler. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Scientific Data 5, 180161, 2018.",
  "A. G. C. Pacheco dkk. PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones. Data in Brief 32, 106221, 2020.",
  "S. P. Mohanty, D. P. Hughes, M. Salathé. Using deep learning for image-based plant disease detection. Frontiers in Plant Science 7, 1419, 2016.",
  "D. Hendrycks, T. Dietterich. Benchmarking neural network robustness to common corruptions and perturbations. ICLR, 2019.",
  "J. Adebayo dkk. Sanity checks for saliency maps. NeurIPS, 2018.",
  "M. Tan, Q. Le. EfficientNet: Rethinking model scaling for convolutional neural networks. ICML, 2019.",
  "M. Oquab dkk. DINOv2: Learning robust visual features without supervision. TMLR, 2024.",
  "X. Bouthillier dkk. Accounting for variance in machine learning benchmarks. MLSys, 2021.",
];

const LIMITS = [
  ["Label tidak identik", "akiec (HAM) memuat penyakit Bowen; ACK (PAD) tidak. Sebagian label PAD tidak terkonfirmasi biopsi."],
  ["Satu sumber OOD", "PAD-UFES-20 berasal dari satu layanan di Brasil; populasi dan warna kulit berbeda dari Indonesia."],
  ["Pergeseran ganda", "Perangkat, populasi pasien, dan distribusi kelas berubah bersamaan, jadi celah tidak murni efek kamera."],
  ["Korupsi tanpa CI", "Hasil uji korupsi dirata-ratakan atas seed tanpa bootstrap, dan implementasinya tidak identik bit-per-bit dengan ImageNet-C."],
  ["Grad-CAM resolusi kasar", "Peta 7×7 dari blok terakhir EfficientNet; energi tepi hanya heuristik."],
  ["Demo browser", "Hanya model seed tuning (seed 0) yang diekspor; baseline DINOv2 tidak dijalankan di browser karena ukurannya."],
];

export default function About() {
  const r = useResults();
  const gh = repoUrl();
  const colab = colabUrl();
  return (
    <div>
      <PageHeader n="06" eyebrow="Tentang" title={r.meta.title} lead={`${r.meta.track} · Proyek Akhir (EAS) Mata Kuliah Deep Learning, 2026.`} />
      <div className="grid gap-5 md:grid-cols-2">
        {r.meta.authors.map((a) => (
          <Card key={a.nrp} className="p-6">
            <div className="font-mono text-xs text-muted">{a.nrp}</div>
            <div className="mt-1 text-xl font-bold">{a.name}</div>
          </Card>
        ))}
      </div>

      <Section title="Sumber daya proyek">
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {[
            ["Repositori GitHub", gh, "Kode, konfigurasi, log run, laporan"],
            ["Notebook Colab", colab, "Jalankan ulang seluruh eksperimen di GPU T4"],
            ["Proposal", gh ? repoUrl("/blob/main/proposal.pdf") : "", "Rumusan masalah & rencana"],
            ["Model card", gh ? repoUrl("/blob/main/model_card.md") : "", "Batas & risiko model"],
            ["Pernyataan kontribusi & AI", gh ? repoUrl("/blob/main/pernyataan.md") : "", "Siapa mengerjakan apa"],
            ["weights.txt", gh ? repoUrl("/blob/main/weights.txt") : "", "Tautan & SHA-256 bobot"],
          ].map(([t, href, d]) => (
            <a key={t} href={href || undefined} target="_blank" rel="noreferrer"
              className={href ? "group" : "pointer-events-none opacity-50"}>
              <Card className="h-full p-5 group-hover:border-lime/40">
                <div className="font-semibold">{t} {href && <span className="text-lime">↗</span>}</div>
                <p className="mt-1 text-sm text-muted">{d}</p>
                {!href && <p className="mt-2 font-mono text-[10px] text-amber">atur VITE_GITHUB_REPO di Vercel</p>}
              </Card>
            </a>
          ))}
        </div>
      </Section>

      <Section title="Keterbatasan" kicker="Dilaporkan apa adanya; hasil negatif juga hasil.">
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {LIMITS.map(([t, d]) => (
            <Card key={t} className="p-5">
              <div className="font-semibold">{t}</div>
              <p className="mt-1 text-sm text-muted leading-relaxed">{d}</p>
            </Card>
          ))}
        </div>
      </Section>

      <Section title="Etika & lisensi">
        <div className="grid gap-4 lg:grid-cols-2">
          <Callout tone="warn" title="Bukan alat diagnosis medis">
            Model ini hanya untuk penelitian akademik non-komersial tentang pergeseran domain. Jangan dipakai untuk
            keputusan klinis.
          </Callout>
          <Callout title="Data">
            Kedua dataset publik dan telah dianonimkan. PAD-UFES-20 dikumpulkan dengan persetujuan komite etik UFES
            (No. 500002/478). HAM10000 berlisensi CC BY-NC 4.0 (non-komersial); PAD-UFES-20 CC BY 4.0. Citra contoh di
            situs ini berasal dari kedua dataset tersebut.
          </Callout>
        </div>
      </Section>

      <Section title="Daftar pustaka">
        <ol className="space-y-2 text-sm text-ink-2 list-decimal pl-6 marker:font-mono marker:text-muted">
          {REFS.map((x) => <li key={x}>{x}</li>)}
        </ol>
      </Section>
      <p className="mt-6 font-mono text-xs text-muted">data dibuat {r.meta.generated_utc}</p>
    </div>
  );
}
