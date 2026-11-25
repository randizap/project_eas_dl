import { lazy, Suspense, useEffect, useState } from "react";
import { NavLink, Route, Routes, useLocation } from "react-router-dom";
import { cx } from "./components/ui";
import { colabUrl, repoUrl, useDataState } from "./lib/data";
import Home from "./pages/Home";
import Method from "./pages/Method";
import ResultsPage from "./pages/Results";
import Interpret from "./pages/Interpret";
import Experiments from "./pages/Experiments";
import About from "./pages/About";

// Demo memuat onnxruntime-web (besar): dipisah agar halaman lain tetap ringan.
const Demo = lazy(() => import("./pages/Demo"));

const NAV = [
  { to: "/", label: "Ringkasan", n: "00" },
  { to: "/metode", label: "Metode", n: "01" },
  { to: "/hasil", label: "Hasil", n: "02" },
  { to: "/interpretasi", label: "Grad-CAM", n: "03" },
  { to: "/demo", label: "Coba Model", n: "04" },
  { to: "/eksperimen", label: "Log Eksperimen", n: "05" },
  { to: "/tentang", label: "Tentang", n: "06" },
];

function Nav() {
  const [open, setOpen] = useState(false);
  const loc = useLocation();
  useEffect(() => { setOpen(false); window.scrollTo({ top: 0 }); }, [loc.pathname]);
  return (
    <nav className="sticky top-0 z-50 border-b border-line bg-bg/80 backdrop-blur-xl">
      <div className="mx-auto flex h-16 max-w-7xl items-center gap-6 px-5 sm:px-8">
        <NavLink to="/" className="flex items-center gap-3 shrink-0">
          <span className="relative grid h-8 w-8 place-items-center rounded-lg bg-lime">
            <span className="h-3 w-3 rounded-full bg-bg" />
          </span>
          <span className="leading-tight">
            <span className="block text-sm font-bold">Derma → Ponsel</span>
            <span className="block font-mono text-[10px] uppercase tracking-widest text-muted">EAS · Track D</span>
          </span>
        </NavLink>
        <div className="hidden lg:flex items-center gap-1 ml-auto">
          {NAV.map((n) => (
            <NavLink
              key={n.to}
              to={n.to}
              end={n.to === "/"}
              className={({ isActive }) =>
                cx(
                  "rounded-lg px-3 py-2 text-sm transition-colors",
                  isActive ? "bg-white/[0.06] text-ink" : "text-muted hover:text-ink",
                  n.to === "/demo" && !isActive && "text-lime hover:text-lime",
                )
              }
            >
              {n.label}
            </NavLink>
          ))}
        </div>
        <button
          className="lg:hidden ml-auto rounded-lg border border-line px-3 py-2 font-mono text-xs text-ink-2"
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
          aria-controls="mobile-nav"
        >
          {open ? "tutup ✕" : "menu ☰"}
        </button>
      </div>
      {open && (
        <div id="mobile-nav" className="lg:hidden border-t border-line bg-bg px-5 py-3 fade-in">
          {NAV.map((n) => (
            <NavLink key={n.to} to={n.to} end={n.to === "/"}
              className={({ isActive }) => cx("flex items-center gap-4 rounded-lg px-3 py-3", isActive ? "bg-white/[0.06] text-ink" : "text-ink-2")}>
              <span className="font-mono text-xs text-muted">{n.n}</span>
              {n.label}
            </NavLink>
          ))}
        </div>
      )}
    </nav>
  );
}

function DemoBanner() {
  const s = useDataState();
  if (s.status !== "ready" || !s.data.meta.is_demo) return null;
  return (
    <div className="border-b border-amber/30 bg-amber/10">
      <div className="mx-auto max-w-7xl px-5 sm:px-8 py-2.5 text-sm text-amber flex flex-wrap items-center gap-x-3 gap-y-1">
        <span className="font-mono text-xs font-bold uppercase tracking-widest">Data demo</span>
        <span className="text-ink-2">
          Angka di situs ini berasal dari <b className="text-ink">uji asap pada data sintetis</b>, bukan hasil eksperimen.
          Jalankan notebook Colab lalu push <code className="font-mono text-xs">web/public/</code> untuk menggantinya.
        </span>
      </div>
    </div>
  );
}

function Footer() {
  const gh = repoUrl();
  const colab = colabUrl();
  return (
    <footer className="mt-24 border-t border-line">
      <div className="mx-auto grid max-w-7xl gap-8 px-5 sm:px-8 py-12 md:grid-cols-3 text-sm">
        <div>
          <div className="font-bold">Proyek Akhir Deep Learning · EAS</div>
          <p className="mt-2 text-muted leading-relaxed">
            Raditya Andiza Putro (23083010058) · Rosyidatul Kamila (23083010076)
          </p>
        </div>
        <div className="text-muted leading-relaxed">
          <b className="text-amber">Bukan alat diagnosis medis.</b> Model ini dibuat untuk penelitian akademik tentang
          pergeseran domain. Jangan gunakan untuk menilai lesi kulit sungguhan; periksakan ke dokter.
        </div>
        <div className="flex flex-col gap-2 md:items-end font-mono text-xs">
          {gh && <a className="text-ink-2 hover:text-lime" href={gh} target="_blank" rel="noreferrer">↗ Repositori GitHub</a>}
          {colab && <a className="text-ink-2 hover:text-lime" href={colab} target="_blank" rel="noreferrer">↗ Buka notebook di Colab</a>}
          <span className="text-muted">HAM10000 CC BY-NC 4.0 · PAD-UFES-20 CC BY 4.0</span>
        </div>
      </div>
    </footer>
  );
}

function Loading() {
  return (
    <div className="grid min-h-[60vh] place-items-center">
      <div className="text-center">
        <div className="mx-auto h-10 w-10 animate-spin rounded-full border-2 border-line-strong border-t-lime" />
        <p className="mt-4 font-mono text-xs uppercase tracking-widest text-muted">memuat hasil eksperimen…</p>
      </div>
    </div>
  );
}

function Missing({ message }: { message: string }) {
  return (
    <div className="mx-auto max-w-2xl py-24">
      <div className="eyebrow">◆ results.json belum ada</div>
      <h1 className="mt-4 text-3xl font-bold">Website belum punya data untuk ditampilkan</h1>
      <p className="mt-4 text-ink-2 leading-relaxed">
        Situs ini membaca <code className="font-mono text-sm text-lime">/data/results.json</code> yang ditulis oleh
        tahap <code className="font-mono text-sm">export</code> di notebook Colab. Jalankan notebook sampai selesai,
        lalu commit folder <code className="font-mono text-sm">web/public/</code> ke GitHub. Vercel akan men-deploy ulang otomatis.
      </p>
      <p className="mt-4 font-mono text-xs text-muted">galat: {message}</p>
    </div>
  );
}

export default function App() {
  const s = useDataState();
  return (
    <div className="min-h-screen flex flex-col">
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:z-[60] focus:m-2 focus:rounded focus:bg-lime focus:px-3 focus:py-2 focus:text-bg">
        Lompat ke konten
      </a>
      <Nav />
      <DemoBanner />
      <main id="main" className="mx-auto w-full max-w-7xl flex-1 px-5 sm:px-8">
        {s.status === "loading" && <Loading />}
        {s.status === "error" && <Missing message={s.message} />}
        {s.status === "ready" && (
          <Suspense fallback={<Loading />}>
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/metode" element={<Method />} />
              <Route path="/hasil" element={<ResultsPage />} />
              <Route path="/interpretasi" element={<Interpret />} />
              <Route path="/demo" element={<Demo />} />
              <Route path="/eksperimen" element={<Experiments />} />
              <Route path="/tentang" element={<About />} />
              <Route path="*" element={<Missing message="halaman tidak ditemukan (404)" />} />
            </Routes>
          </Suspense>
        )}
      </main>
      <Footer />
    </div>
  );
}
