import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import type { ConfigId, Results, Stat } from "./types";

type State = { status: "loading" } | { status: "error"; message: string } | { status: "ready"; data: Results };

const Ctx = createContext<State>({ status: "loading" });

export function DataProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<State>({ status: "loading" });
  useEffect(() => {
    fetch("/data/results.json", { cache: "no-cache" })
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((data: Results) => setState({ status: "ready", data }))
      .catch((e) => setState({ status: "error", message: String(e) }));
  }, []);
  return <Ctx.Provider value={state}>{children}</Ctx.Provider>;
}

export function useDataState() {
  return useContext(Ctx);
}

/** Hanya dipakai di dalam halaman yang dirender setelah data siap. */
export function useResults(): Results {
  const s = useContext(Ctx);
  if (s.status !== "ready") throw new Error("results.json belum dimuat");
  return s.data;
}

// ---------------------------------------------------------------- konstanta tampilan
export const CONFIGS: ConfigId[] = ["baseline_lp", "main", "phone"];

export const CONFIG_META: Record<ConfigId, { name: string; short: string; color: string; desc: string }> = {
  baseline_lp: {
    name: "Baseline · DINOv2 + regresi logistik",
    short: "Baseline",
    color: "var(--s1)",
    desc: "Fitur DINOv2 ViT-S/14 dibekukan, hanya regresi logistik berbobot kelas yang dilatih.",
  },
  main: {
    name: "Model utama · EfficientNet-B0",
    short: "Utama",
    color: "var(--s2)",
    desc: "EfficientNet-B0 pretrained ImageNet, fine-tuning penuh 224 px, augmentasi standar.",
  },
  phone: {
    name: "Intervensi · EfficientNet-B0 + aug. ponsel",
    short: "Intervensi",
    color: "var(--s3)",
    desc: "Model utama + augmentasi karakteristik ponsel: warna & white balance, blur, JPEG, zoom-out.",
  },
};

export const PART_LABEL = { test: "HAM10000 test · ID", ood: "PAD-UFES-20 · OOD" } as const;
export const METRIC_LABEL: Record<string, string> = {
  bal_acc: "Balanced accuracy",
  macro_f1: "Macro-F1",
};

// ---------------------------------------------------------------- format angka (gaya Indonesia)
export function pct(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "–";
  return (v * 100).toFixed(digits).replace(".", ",");
}
export function signedPct(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "–";
  const s = (v * 100).toFixed(digits).replace(".", ",");
  return v > 0 ? `+${s}` : s.replace("-", "−");
}
export function num(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "–";
  return v.toFixed(digits).replace(".", ",");
}
export function ciText(s: Stat | undefined, signed = false): string {
  if (!s) return "–";
  const f = signed ? signedPct : pct;
  return `[${f(s.ci[0])}; ${f(s.ci[1])}]`;
}
export function ciExcludesZero(s: Stat | undefined): boolean {
  if (!s || s.ci[0] === null || s.ci[1] === null) return false;
  return s.ci[0] > 0 || s.ci[1] < 0;
}

export const REPO = (import.meta.env.VITE_GITHUB_REPO as string | undefined)?.trim() || "";
export const repoUrl = (path = "") => (REPO ? `https://github.com/${REPO}${path}` : "");
export const colabUrl = () =>
  REPO ? `https://colab.research.google.com/github/${REPO}/blob/main/notebooks/EAS_DL_Colab.ipynb` : "";
