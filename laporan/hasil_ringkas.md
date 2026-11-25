# Ringkasan hasil (otomatis dari outputs/predictions)

Format: rerata 3 seed [CI 95% bootstrap klaster, B=2000] (sd antar seed). Satuan: %.

## RQ1/RQ2 - in-distribution (HAM test) vs out-of-distribution (PAD-UFES-20)

| Konfigurasi | BalAcc ID | BalAcc OOD | Celah BalAcc | Macro-F1 ID | Macro-F1 OOD | Celah Macro-F1 |
|---|---|---|---|---|---|---|
| Baseline (DINOv2 + LR) | 71.2 [66.2, 75.9] (sd 0.6) | 42.0 [38.8, 45.7] (sd 0.6) | 29.3 [23.2, 34.8] | 62.7 [57.6, 67.2] | 26.7 [24.8, 28.6] | 36.0 [30.4, 40.6] |
| EfficientNet-B0 | 75.7 [71.5, 79.7] (sd 0.7) | 45.0 [42.1, 48.3] (sd 1.2) | 30.7 [25.4, 35.8] | 68.1 [63.6, 72.0] | 35.5 [33.0, 38.0] | 32.6 [27.5, 37.3] |
| EfficientNet-B0 + aug. ponsel | 77.9 [73.7, 82.1] (sd 1.0) | 50.9 [47.4, 54.6] (sd 0.8) | 27.0 [21.4, 32.4] | 69.6 [64.9, 73.7] | 38.9 [36.5, 41.2] | 30.7 [25.5, 35.5] |

## Recall per kelas (rerata seed [CI])

| Konfigurasi | Domain | MEL | NEV | BCC | AK | ECE |
|---|---|---|---|---|---|---|
| Baseline (DINOv2 + LR) | ID | 70.9 [62.5, 78.9] | 78.5 [75.5, 81.5] | 85.2 [75.3, 93.2] | 50.4 [35.8, 64.0] | 0.056 |
| Baseline (DINOv2 + LR) | OOD | 27.6 [15.6, 40.7] | 90.3 [86.5, 93.6] | 47.8 [44.2, 51.5] | 2.3 [1.4, 3.4] | 0.393 |
| EfficientNet-B0 | ID | 80.7 [74.1, 86.5] | 80.2 [77.6, 83.0] | 81.1 [72.3, 88.8] | 60.7 [48.6, 73.6] | 0.073 |
| EfficientNet-B0 | OOD | 19.9 [10.3, 31.4] | 85.0 [80.7, 88.7] | 57.6 [54.7, 60.4] | 17.6 [15.5, 19.7] | 0.294 |
| EfficientNet-B0 + aug. ponsel | ID | 83.1 [76.8, 89.0] | 78.8 [75.7, 81.6] | 88.1 [80.6, 94.6] | 61.5 [47.5, 75.3] | 0.063 |
| EfficientNet-B0 + aug. ponsel | OOD | 51.3 [37.9, 65.1] | 67.8 [62.4, 72.6] | 67.5 [64.7, 70.3] | 16.9 [14.6, 19.2] | 0.292 |

## RQ2 - intervensi dikurangi model utama (berpasangan)

| Metrik | Domain | Selisih [CI] | CI tidak memuat 0? |
|---|---|---|---|
| bal_acc | ID | 2.2 [-1.8, 6.0] | tidak |
| macro_f1 | ID | 1.5 [-2.0, 4.9] | tidak |
| recall_MEL | ID | 2.4 [-2.5, 7.3] | tidak |
| recall_NEV | ID | -1.5 [-3.2, 0.2] | tidak |
| recall_BCC | ID | 7.0 [-0.4, 15.7] | tidak |
| recall_AK | ID | 0.7 [-11.4, 12.6] | tidak |
| bal_acc | OOD | 5.8 [2.9, 8.9] | ya |
| macro_f1 | OOD | 3.3 [1.4, 5.3] | ya |
| recall_MEL | OOD | 31.4 [20.8, 41.8] | ya |
| recall_NEV | OOD | -17.2 [-21.9, -12.5] | ya |
| recall_BCC | OOD | 9.9 [7.3, 12.5] | ya |
| recall_AK | OOD | -0.7 [-2.9, 1.5] | tidak |
| perubahan celah bal_acc | ID-OOD | -3.7 [-8.7, 1.1] | - |
| perubahan celah macro_f1 | ID-OOD | -1.9 [-5.9, 2.1] | - |

## RQ3 - split acak per citra vs per lesi (test in-distribution)

| Konfigurasi | BalAcc split citra | BalAcc split lesi | Inflasi [CI] | BalAcc OOD (model split citra) |
|---|---|---|---|---|
| Baseline (DINOv2 + LR) | 76.9 [72.6, 80.9] | 71.2 [66.4, 75.8] | 5.7 [-0.8, 12.0] | 41.4 |
|  - rincian seed pertama | lesi ada di train: 77.0 (n=417) | lesi tidak ada di train: 71.2 (n=882) | | |
| EfficientNet-B0 | 80.2 [76.3, 84.1] | 75.7 [71.4, 79.7] | 4.5 [-1.3, 10.6] | 41.6 |
|  - rincian seed pertama | lesi ada di train: 81.7 (n=417) | lesi tidak ada di train: 78.8 (n=882) | | |

## Uji korupsi (HAM test, balanced accuracy %, rerata korupsi x severity x seed)

| Konfigurasi | Bersih | Korupsi non-overlap | Korupsi overlap dgn aug. ponsel |
|---|---|---|---|
| Baseline (DINOv2 + LR) | 71.2 | 38.2 | 55.4 |
| EfficientNet-B0 | 75.7 | 27.2 | 41.2 |
| EfficientNet-B0 + aug. ponsel | 77.9 | 33.2 | 65.6 |

## Grad-CAM - uji pengacakan parameter (Spearman rerata vs peta asli)

| Run | Domain | Kepala diacak | Semua lapisan diacak |
|---|---|---|---|
| cnn_main_lesion_s0 | ood | 0.298 | 0.175 |
| cnn_main_lesion_s0 | test | 0.186 | 0.066 |
| cnn_phone_lesion_s0 | ood | -0.093 | 0.128 |
| cnn_phone_lesion_s0 | test | 0.192 | -0.019 |

### Energi Grad-CAM (heuristik)

```
             run_id domain  border_energy
 cnn_main_lesion_s0    ood          0.204
 cnn_main_lesion_s0   test          0.215
cnn_phone_lesion_s0    ood          0.211
cnn_phone_lesion_s0   test          0.192
```

## Pemeriksaan kebocoran

```json
{
  "candidate_pairs_by_kind": {
    "cross_dataset": 32,
    "diff_lesion_ham": 880,
    "diff_lesion_pad": 12,
    "same_lesion": 412
  },
  "groups_with_mixed_labels": 0,
  "ham_groups_after_merge": 6564,
  "ham_lesions": 6572,
  "largest_group_images": 6,
  "n_images": 10530,
  "pad_excluded_from_ood": 0,
  "pairs_by_kind": {
    "diff_lesion_ham": 8,
    "diff_lesion_pad": 7,
    "same_lesion": 108
  },
  "pairs_label_conflict": 6,
  "pairs_rejected_manually": 0,
  "thresholds": {
    "cosine_min": 0.95,
    "dup_rule": {
      "cosine_alone": 0.985,
      "phash_and_cosine": 0.9
    },
    "phash_max_hamming": 4
  }
}
```

## Jumlah citra per split (split per lesi, seed pertama)

```json
{
  "counts": {
    "test": {
      "AK": 45,
      "BCC": 81,
      "MEL": 166,
      "NEV": 1000
    },
    "train": {
      "AK": 234,
      "BCC": 356,
      "MEL": 770,
      "NEV": 4708
    },
    "val": {
      "AK": 48,
      "BCC": 77,
      "MEL": 177,
      "NEV": 997
    }
  },
  "lesion_overlap_train_test": 0,
  "mode": "lesion",
  "ood_counts": {
    "AK": 730,
    "BCC": 845,
    "MEL": 52,
    "NEV": 244
  },
  "ood_excluded": 0,
  "seed": 0
}
```
