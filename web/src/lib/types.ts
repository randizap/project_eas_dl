// Bentuk results.json yang ditulis src/export_web.py. Semua angka berasal dari pipeline.

export type ConfigId = "baseline_lp" | "main" | "phone";
export type Part = "test" | "ood";

export interface Stat {
  mean: number | null;
  sd_seeds?: number | null;
  ci: [number | null, number | null];
}

export interface PerSeedMetrics {
  bal_acc: number;
  macro_f1: number;
  acc: number;
  ece: number;
  n: number;
  confusion: number[][];
  [k: string]: number | number[][];
}

export interface PartEntry {
  per_seed: PerSeedMetrics[];
  n_images: number;
  n_clusters: number;
  [metric: string]: Stat | PerSeedMetrics[] | number;
}

export interface ConfigEntry {
  test: PartEntry;
  ood: PartEntry;
  gap: Record<string, Stat>;
}

export interface PairedStat extends Stat {
  ci_excludes_zero?: boolean;
}

export interface Rq3Mode {
  bal_acc: Stat;
  macro_f1: Stat;
  bal_acc_lesion_seen_in_train?: { n: number; value: number | null };
  bal_acc_lesion_unseen?: { n: number; value: number | null };
}

export interface Rq3Entry {
  image: Rq3Mode;
  lesion: Rq3Mode;
  inflation_bal_acc: Stat & { note?: string };
  inflation_macro_f1: Stat & { note?: string };
  image_ood_bal_acc: number | null;
}

export interface Summary {
  classes: string[];
  B: number;
  ci_level: number;
  configs: Record<ConfigId, ConfigEntry>;
  paired_phone_minus_main: {
    test: Record<string, PairedStat>;
    ood: Record<string, PairedStat>;
    gap_change_bal_acc: Stat;
    gap_change_macro_f1: Stat;
  };
  rq3: Partial<Record<ConfigId, Rq3Entry>>;
  corruption?: {
    clean_bal_acc: Partial<Record<ConfigId, number>>;
    mean_by_group: Partial<Record<ConfigId, { overlap?: number; non_overlap?: number }>>;
    note: string;
  };
  gradcam_randomization?: {
    run_id: string;
    domain: Part;
    step: number;
    randomized_upto: string;
    spearman: number | null;
    ssim: number | null;
  }[];
  gradcam_energy?: { run_id: string; domain: Part; border_energy: number | null; lesion_energy?: number | null }[];
  leakage?: Record<string, unknown> & {
    n_images?: number;
    ham_lesions?: number;
    ham_groups_after_merge?: number;
    largest_group_images?: number;
    groups_with_mixed_labels?: number;
    pad_excluded_from_ood?: number;
    pairs_by_kind?: Record<string, number>;
    pairs_label_conflict?: number;
    pairs_rejected_manually?: number;
    thresholds?: { cosine_min: number; phash_max_hamming: number };
  };
  split_counts: SplitInfo;
}

export interface SplitInfo {
  counts: Record<"train" | "val" | "test", Record<string, number>>;
  ood_counts: Record<string, number>;
  ood_excluded: number;
  lesion_overlap_train_test: number;
  mode: string;
  seed: number;
}

export interface RunLogRow {
  run_id: string;
  timestamp_utc: string;
  stage: string;
  config_name: string | null;
  seed: number | null;
  split_mode: string | null;
  status: string;
  gpu: string | null;
  train_minutes: number | null;
  best_epoch: number | null;
  val_bal_acc: number | null;
  val_macro_f1: number | null;
  hparams: string | null;
  notes: string | null;
}

export interface HistoryRow {
  epoch: number;
  train_loss: number;
  val_bal_acc: number;
  val_macro_f1: number;
  lr: number;
}

export interface CorruptionRow {
  config: ConfigId;
  corruption: string;
  severity: number;
  overlap_with_phone_aug: boolean;
  mean: number;
  std: number | null;
}

export interface ModelInfo {
  id: "main" | "phone";
  run_id: string;
  label: string;
  file: string;
  size_mb: number;
  sha256: string;
  checkpoint_sha256: string;
  max_abs_diff_vs_torch: number;
}

export interface GradcamExample {
  run_id: string;
  domain: Part;
  image_id: string;
  image: string;
  label: string;
  pred: string | null;
  probs: number[] | null;
  steps: string[];
  maps: string[];
}

export interface Results {
  meta: {
    is_demo: boolean;
    generated_utc: string;
    title: string;
    track: string;
    authors: { name: string; nrp: string }[];
    classes: string[];
    class_names: Record<string, string>;
    environment: Record<string, string>;
    test_opened: { opened_utc: string; forced_reopen: boolean; checkpoints: Record<string, string> } | null;
  };
  config: any;
  summary: Summary;
  corruption_detail: CorruptionRow[];
  runs_log: RunLogRow[];
  histories: Record<string, HistoryRow[]>;
  done: Record<string, any>;
  tuned: { lr: number; val_bal_acc_per_lr: Record<string, number>; seed: number; selected_on: string } | null;
  data_counts: Record<string, Record<string, number>>;
  orig_label_counts: { domain: string; orig_label: string; n: number }[];
  splits: Record<string, SplitInfo>;
  leakage_pairs: Record<string, any>[];
  has_contact_sheet: boolean;
  models: ModelInfo[];
  gradcam_examples: GradcamExample[];
}
