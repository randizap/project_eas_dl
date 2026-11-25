# Generalisasi Klasifikasi Lesi Kulit: Dermatoskopi → Citra Ponsel

**Proyek Akhir (EAS) Deep Learning · Track D: Pergeseran Data (Distribution Shift)**
Raditya Andiza Putro (23083010058) · Rosyidatul Kamila (23083010076)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/randizap/project_eas_dl/blob/main/notebooks/EAS_DL_Colab.ipynb)
&nbsp;·&nbsp; 🌐 **Website:** <https://project-eas-dl.vercel.app>

Model dilatih **hanya** pada HAM10000 (dermatoskopi), lalu diuji pada test set HAM10000 yang belum pernah dilihat (*in-distribution*, ID) dan pada PAD-UFES-20 (foto ponsel, *out-of-distribution*, OOD).

| | Pertanyaan | Bukti yang dihasilkan pipeline | Di website |
|---|---|---|---|
| RQ1 | Seberapa besar penurunan balanced accuracy & macro-F1 dari ID ke OOD? | `tab_id_ood`, celah ID−OOD + CI 95% | `/hasil#rq1` |
| RQ2 | Apakah augmentasi karakteristik ponsel memperkecil celah, dan berapa biayanya di ID? | `tab_paired` (selisih berpasangan + CI), uji korupsi | `/hasil#rq2`, `/hasil#korupsi` |
| RQ3 | Seberapa besar skor menjadi terlalu tinggi bila HAM10000 dibagi acak per citra, bukan per lesi? | `tab_rq3` (inflasi + CI, rincian lesi terlihat/tidak di train) | `/hasil#rq3` |

> **Status:** pipeline, notebook Colab, dan website sudah diuji *end-to-end* pada data sintetis (`--smoke`). Website saat ini menampilkan banner **DATA DEMO**. Angka asli muncul otomatis setelah notebook Colab dijalankan dan hasilnya di-push. Jangan menulis angka dengan tangan.

---

## 1. Alur kerja: Colab → GitHub → Vercel

```
 ┌───────────────────────┐    push     ┌──────────┐   auto-deploy   ┌─────────────────────────┐
 │ Google Colab (T4 GPU) │ ──────────▶ │  GitHub  │ ──────────────▶ │ Vercel (situs statis)   │
 │ notebooks/            │  web/public │  repo    │                 │ React + ONNX Runtime Web│
 │ EAS_DL_Colab.ipynb    │  runs/log   │          │                 │ model jalan di browser  │
 └──────────┬────────────┘             └──────────┘                 └─────────────────────────┘
            │ checkpoint .pt, cache, prediksi
            ▼
     Google Drive (aman saat sesi Colab putus)
```

1. **Training di Colab.** Notebook menjalankan `prepare → leakage → split → tune → train → evaluate → report → export`. Checkpoint disimpan di Drive; sesi yang putus dilanjutkan dari epoch terakhir.
2. **Ekspor.** Tahap `export` (`src/export_web.py`) menulis `web/public/data/results.json` (semua angka), `web/public/models/{main,phone}.onnx` (EfficientNet-B0 + kepala CAM, diverifikasi sama dengan PyTorch), dan contoh Grad-CAM.
3. **GitHub.** Sel terakhir notebook meng-commit hasil (bukan checkpoint) lalu push.
4. **Vercel.** Setiap push memicu build ulang `web/`. Situs membaca `results.json`; demo menjalankan model ONNX **di browser pengunjung**.

## 2. Struktur repositori (sesuai slide 8)

```
project_eas_dl/
├── README.md               # dokumen ini
├── requirements.txt        # versi paket terkunci (dikunci ulang dari Colab, sel 5.3)
├── requirements-colab.txt  # paket tambahan untuk Colab
├── proposal.pdf            # proposal yang diajukan
├── laporan/                # sumber LaTeX IEEE; tables/ & figures/ diisi otomatis
├── model_card.md           # model card + datasheet
├── runs/log.csv            # SEMUA run, termasuk yang gagal (ditulis otomatis)
├── weights.txt             # tautan + SHA-256 bobot (ditulis otomatis)
├── pernyataan.md           # kontribusi anggota & pemakaian AI
├── configs/default.yaml    # SEMUA hiperparameter & ambang
├── notebooks/
│   └── EAS_DL_Colab.ipynb  # ▶ training di Google Colab, satu klik per langkah
├── src/                    # pipeline Python (lihat §6)
│   └── export_web.py       # ekspor hasil + model ONNX ke web/public
├── tests/test_core.py      # uji unit metrik, bootstrap, korupsi
├── web/                    # website (Vite + React + TypeScript + Tailwind)
│   ├── public/data/results.json   # ← ditulis tahap export
│   ├── public/models/*.onnx       # ← ditulis tahap export
│   └── src/pages/          # Ringkasan, Metode, Hasil, Grad-CAM, Demo, Log, Tentang
└── vercel.json             # konfigurasi build & routing Vercel
```

## 3. Langkah demi langkah

### 3.1 Sekali saja: GitHub + Vercel
1. Buat repositori baru di GitHub (mis. `project_eas_dl`), lalu dari folder ini:
   ```bash
   git remote add origin https://github.com/randizap/project_eas_dl.git
   git push -u origin main
   ```
2. Ganti `USERNAME` di badge Colab di atas.
3. Di <https://vercel.com/new>, **Import** repositori tersebut. Biarkan *Root Directory* = `./` (konfigurasi diambil dari `vercel.json`). Tambahkan *Environment Variable* `VITE_GITHUB_REPO` = `randizap/project_eas_dl` (untuk tautan GitHub/Colab di situs), lalu **Deploy**.

### 3.2 Training di Google Colab
1. Klik badge **Open in Colab**, pilih **Runtime → Change runtime type → T4 GPU**.
2. Di **Colab Secrets** (ikon 🔑): `KAGGLE_USERNAME`, `KAGGLE_KEY` (unduh data), dan `GITHUB_TOKEN` (push hasil; *fine-grained token*, izin *Contents: read & write* untuk repo ini).
3. Jalankan sel dari atas ke bawah. Sesi putus? Jalankan ulang 0.2 → 0.3 → sel terakhir yang belum selesai.
4. Sel 6.1 meng-commit `web/public/`, `runs/log.csv`, `weights.txt`, `requirements.txt`, dan tabel/gambar laporan, lalu push. Vercel deploy otomatis; banner DATA DEMO hilang dengan sendirinya.

### 3.3 Menjalankan di laptop (opsional, tanpa GPU)
```powershell
.venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q tests                     # uji unit
python -m src.run_all --smoke                 # seluruh pipeline + export pada data sintetis (CPU, ±5 menit)
cd web; npm install; npm run dev              # website di http://localhost:5173
```
Tahap `tune`/`train`/`evaluate` menolak jalan tanpa GPU CUDA (kecuali `--allow-cpu`), karena 12 run EfficientNet di CPU bisa makan berhari-hari.

### 3.4 Setelah evaluasi
```bash
python -m src.hash_weights --link-prefix "https://drive.google.com/drive/folders/<ID_FOLDER>"   # weights.txt (sel 5.1)
cd laporan && pdflatex laporan && bibtex laporan && pdflatex laporan && pdflatex laporan && cp laporan.pdf ..
```
Atau unggah folder `laporan/` ke Overleaf.

## 4. Website

| Halaman | Isi |
|---|---|
| `/` Ringkasan | Angka utama ID vs OOD, jawaban RQ1–3 (kalimat dibuat otomatis dari CI), lima pemeriksaan |
| `/metode` | Data & pemetaan kelas, cek kebocoran (pasangan + contact sheet), split per lesi, konfigurasi, augmentasi, evaluasi |
| `/hasil` | Dumbbell ID→OOD + CI, recall per kelas, confusion matrix per seed, forest plot RQ2/RQ3, kurva korupsi |
| `/interpretasi` | Uji pengacakan Grad-CAM + galeri dengan slider pengacakan lapisan |
| `/demo` | Unggah citra → simulasi kamera ponsel (zoom-out, warna, white balance, blur, JPEG) → prediksi + Grad-CAM kedua model, uji sapuan ketahanan. **Semua di browser.** |
| `/eksperimen` | Log semua run, kurva training, tuning lr, kunci test set, SHA-256 model |
| `/tentang` | Tim, keterbatasan, etika & lisensi, pustaka |

Setiap grafik punya tooltip, tampilan **tabel** (aksesibilitas), dan palet tervalidasi untuk buta warna. Kalimat kesimpulan hanya menyebut efek "nyata" bila CI 95% tidak memuat nol.

## 5. Data

| Dataset | Unduh dari | Lisensi | Letakkan di |
|---|---|---|---|
| HAM10000 (Tschandl dkk. 2018) | Harvard Dataverse `doi:10.7910/DVN/DBW86T` (setujui *terms of use*), atau mirror Kaggle `kmader/skin-cancer-mnist-ham10000` | CC BY-NC 4.0, non-komersial | `data/raw/ham10000/` → `HAM10000_metadata.csv` + semua `*.jpg` (subfolder bebas) |
| PAD-UFES-20 (Pacheco dkk. 2020) | Mendeley Data `doi:10.17632/zr7vgbcyr2.1` | CC BY 4.0 | `data/raw/pad_ufes_20/` → `metadata.csv` + semua `*.png` (subfolder bebas) |

Kode mencari citra secara rekursif dan **mengekstrak semua zip secara otomatis** (termasuk zip di dalam zip). Cukup taruh file unduhan apa adanya ke folder tersebut. Metadata boleh `.csv` (Kaggle/Mendeley) atau `.tab` (Dataverse). Zip test ISIC2018 dan masker segmentasi tidak diekstrak.

**Empat kelas yang berpadanan** (proposal §3.1):

| Kelas | HAM10000 (`dx`) | n HAM | PAD-UFES-20 (`diagnostic`) | n PAD |
|---|---|---|---|---|
| MEL | `mel` | 1.113 | `MEL` | 52 |
| NEV | `nv` | 6.705 | `NEV` | 244 |
| BCC | `bcc` | 514 | `BCC` | 845 |
| AK | `akiec` | 327 | `ACK` | 730 |
| dibuang | `bkl`, `df`, `vasc` | 1.356 | `SCC`, `SEK` | 427 |

Angka n berasal dari publikasi dataset (Tschandl dkk. 2018, Pacheco dkk. 2020, Tabel 2). Tahap `prepare` mencetak hitungan aktual dari file Anda; kalau berbeda, angka cetakan itulah yang benar.

⚠️ **Dua ketidakcocokan label yang wajib dibahas di Keterbatasan:**
1. `akiec` HAM mencakup keratosis aktinik **dan** penyakit Bowen (karsinoma intraepitelial), sedangkan `ACK` PAD hanya keratosis aktinik (penyakit Bowen di PAD digabung ke SCC, yang kita buang). Jadi kelas "AK" tidak identik di kedua domain.
2. Hanya 24,4% citra `ACK` dan 24,6% citra `NEV` di PAD yang terbukti biopsi (Pacheco dkk. 2020), sehingga kualitas label kedua domain berbeda.

## 6. Lima pemeriksaan wajib (slide 4) → di mana dalam kode

| # | Pemeriksaan | Implementasi |
|---|---|---|
| 01 | **Cek kebocoran** | `src/leakage.py`: pHash 64-bit (Hamming ≤ 4) + kosinus embedding DINOv2 (≥ 0,95) pada **semua** citra kedua dataset. Lesi HAM berbeda yang punya citra hampir-kembar digabung (union-find) menjadi satu grup. Citra PAD yang kembar dengan citra HAM dikeluarkan dari uji OOD. Hasil: `outputs/leakage/report.json`, `pairs.csv`, `contact_sheet.png`. |
| 02 | **Split per sumber** | `src/split.py`: 70/15/15 per **grup** (lesi + gabungan duplikat), terstratifikasi. Ada `assert` bahwa tidak ada grup/lesi di dua sisi split. PAD tidak pernah dipakai untuk melatih atau memilih model. |
| 03 | **Baseline sederhana** | `src/train_linear_probe.py`: fitur DINOv2 ViT-S/14 beku + regresi logistik berbobot kelas, C dipilih di val. |
| 04 | **Skor + rentang** | 3 seed untuk baseline & model final; CI 95% dari **bootstrap klaster** (klaster = lesi untuk HAM, pasien untuk PAD; B = 2000); perbandingan intervensi vs utama **berpasangan** (resample & seed yang sama). `src/metrics.py`, `src/make_report.py`. |
| 05 | **Bisa diulang** | Seed tetap, `configs/default.yaml`, `runs/log.csv` (termasuk run gagal, ditulis otomatis), `outputs/environment.txt`, SHA-256 bobot di `weights.txt`, satu perintah `python -m src.run_all`. |

**Pemeriksaan manual duplikat** (proposal §3.1): setelah tahap `leakage`, buka `outputs/leakage/contact_sheet.png`, salin `pairs.csv` ke `pairs_reviewed.csv`, isi kolom `manual_verdict` dengan `bukan` untuk pasangan yang ternyata bukan duplikat, lalu jalankan ulang `--stages leakage split`. Jika ambang terlalu longgar sehingga terbentuk grup raksasa, tahap `split` **menolak berjalan**. Kalau ambang diubah setelah ada run terlatih, tahap `split` juga menolak menimpa split lama.

**Test set dibuka sekali** (slide 7): `src/evaluate.py` menolak jalan jika masih ada run yang belum selesai. Ia menulis `outputs/TEST_OPENED.json` (beserta SHA-256 setiap checkpoint) **sebelum** prediksi pertama, dan menolak dijalankan ulang. `--force-reopen-test` hanya untuk galat teknis, dan pemakaiannya tercatat di `runs/log.csv` sebagai `FORCED REOPEN`.

## 7. Keputusan metodologis & alasannya

| Keputusan | Alasan | Status vs proposal |
|---|---|---|
| Latih pada **4 kelas** saja (subset HAM), bukan 7 lalu dipetakan | Evaluasi hanya 4 kelas; melatih 7 kelas membuat prediksi ke `bkl/df/vasc` otomatis salah di OOD dan mengaburkan RQ1. | Tafsiran proposal §3.1 (dituliskan di laporan) |
| **Test HAM tetap** (`test_seed=2026`); train/val dipartisi ulang per seed | Kalau split juga tetap, regresi logistik (lbfgs) deterministik dan "3 seed" baseline akan identik (sd = 0, tidak bermakna). Mengacak partisi train/val menangkap varians pengambilan sampel data, yang menurut Bouthillier dkk. (2021) termasuk sumber varians terbesar. Test tetap supaya semua konfigurasi diuji di data yang sama. | **Deviasi kecil**, wajib disebut di laporan |
| Bootstrap per **klaster**, bukan per citra | Satu lesi HAM bisa punya beberapa foto, dan satu pasien PAD bisa punya beberapa lesi. Bootstrap per citra akan memberi CI yang terlalu sempit. | Penajaman proposal §3.3 |
| Seleksi checkpoint & C & lr dari **balanced accuracy val HAM** | Metrik utama; tahan terhadap ketidakseimbangan kelas (NEV ≈ 77% data latih). | Sesuai |
| lr dituning hanya pada model utama, seed 0, grid {1e-4, 3e-4, 1e-3}; intervensi memakai lr yang sama | Hemat GPU (slide 7). Satu-satunya perbedaan model utama vs intervensi adalah augmentasinya. | Sesuai |
| Augmentasi standar: RandomResizedCrop(0,6–1), flip H/V, rotasi 90°, color jitter ringan | Lesi tidak punya orientasi kanonik; rotasi 90° tidak menambah sudut hitam baru. | Sesuai |
| Augmentasi ponsel: zoom-out 0,4–0,9× (p=0,5), color jitter kuat + gain white balance 0,85–1,15, blur σ 0,1–2 (p=0,3), JPEG q 30–90 (p=0,5) | Ditetapkan **a priori** dari sifat foto ponsel. **Tidak** dituning dengan PAD, karena itu akan membocorkan data uji OOD ke pemilihan model. | Sesuai |
| PAD disimpan PNG di cache (tidak dikonversi ke JPEG) | Supaya tidak menambahkan artefak JPEG ke data OOD, mengingat JPEG juga salah satu augmentasi intervensi. | Detail implementasi |
| Korupsi dibagi **overlap / non-overlap** dengan augmentasi ponsel | Blur, JPEG, dan kecerahan/kontras dilatih langsung oleh intervensi, jadi ketahanan terhadapnya bukan bukti generalisasi. Yang informatif adalah korupsi non-overlap (noise, pixelate). | Penajaman proposal §3.3 |
| Grad-CAM + uji pengacakan bertingkat (Adebayo dkk. 2018), plus energi peta di pita tepi 15% | Slide 9: Grad-CAM tanpa uji pengacakan tidak dihitung. Energi tepi adalah heuristik untuk artefak vignette, penggaris, dan latar. | Sesuai |
| ECE dilaporkan sebagai metrik sekunder | Slide 4 meminta kalibrasi dipilih sesuai track. Pada pergeseran domain, kalibrasi biasanya memburuk, jadi ini informatif dan murah dihitung. | Tambahan |

## 8. Anggaran GPU (perkiraan, ukur sendiri)

Jumlah run CNN: 3 (tuning lr) + 3 (utama) + 3 (intervensi) + 3 (RQ3 split per citra) = **12**, masing-masing ≤ 15 epoch dengan early stopping. Linear probe dan uji korupsi hanya butuh beberapa menit. Di Colab T4 dengan 2 vCPU, pemuatan citra dan augmentasi ponsel (JPEG, blur) kemungkinan besar menjadi bottleneck di CPU, bukan GPU. Ukur waktu epoch pertama dari log, lalu kalikan. Kalau kuota tidak cukup, kurangi `tune_lr_grid` atau `rq3_cnn_seeds` di config, **dan** laporkan pengurangan itu di Keterbatasan. Seed baseline dan model final tidak boleh dikurangi (slide 7).

## 9. Verifikasi yang sudah dilakukan (tanpa GPU & tanpa dataset asli)

```bash
python src/check_env.py            # cek lingkungan & data
python -m pytest -q tests          # 4 uji: metrik = sklearn, ECE, bootstrap klaster utuh, korupsi deterministik & monoton
python -m src.run_all --smoke      # seluruh pipeline + export pada data sintetis kecil (CPU, ±5 menit)
```
Yang terverifikasi pada data sintetis:
- Semua tahap berjalan, termasuk tabel LaTeX dan gambar.
- Duplikat lintas-lesi dan lintas-dataset yang sengaja ditanam **terdeteksi**: digabung ke satu grup / dikeluarkan dari OOD.
- Ambang yang terlalu longgar membentuk grup raksasa, dan split **dihentikan**.
- Membuka test set kedua kali **ditolak**.
- Run yang diinterupsi tercatat `failed` di log, lalu **dilanjutkan** dari epoch berikutnya.
- Skor "bersih" pada jalur uji korupsi sama dengan skor test biasa (konsistensi praproses).
- Kompilasi lolos di Python 3.11, 3.12, dan 3.13. Uji penuh di **Python 3.11 dengan versi persis `requirements.txt`**.
- Bisa dijalankan sebagai file (`python src\run_all.py`) dari folder kerja mana pun.
- DataLoader dengan worker *spawn* (perilaku Windows) berjalan.
- Format unduhan asli tertangani: metadata `.tab` dari Dataverse, zip bagian 1/2, zip di dalam zip milik PAD; zip test ISIC2018 dilewati.

- **Website**: `npm run build` lolos (TypeScript strict); semua halaman dicek di browser (desktop & 375 px) tanpa galat konsol; demo memuat kedua model ONNX dan menjalankan inferensi + Grad-CAM + uji sapuan di browser.
- **Ekspor ONNX**: keluaran ONNX identik dengan PyTorch (selisih maks < 1e-4, diperiksa otomatis saat ekspor).
- Checkpoint yang korup akibat sesi terputus saat menyimpan kini terdeteksi; training diulang dari awal alih-alih gagal terus (dan penyimpanan dibuat atomik).

Yang **tidak** terverifikasi: performa nyata, waktu GPU, dan pengunduhan bobot pretrained (host bobot tidak bisa dijangkau dari lingkungan pengujian, jadi uji asap memakai arsitektur yang sama dengan bobot acak). Catatan: dengan bobot acak dan hanya ±50 langkah, CNN tampak menebak satu kelas di mode eval. Sudah dibuktikan bahwa ini efek statistik BatchNorm yang belum konvergen (setelah kalibrasi ulang BN, akurasi train/val = 1,0 pada data sintetis), bukan bug label atau praproses. Dengan bobot ImageNet, masalah ini tidak terjadi.

## 10. Pemetaan ke rubrik (slide 9)

| Komponen | Bobot | Bukti di repo |
|---|---|---|
| Split & cek bocor | 20 | §4 no. 01–02, `outputs/leakage/`, `outputs/splits/*.json`, RQ3 |
| Kedalaman track | 20 | ID vs OOD, intervensi berpasangan, korupsi overlap/non-overlap, Grad-CAM tersanitasi |
| Tanya jawab individu | 20 | Setiap anggota harus bisa menjelaskan §6–7 dan setiap file di `src/` (lihat `pernyataan.md`) |
| Evaluasi & analisis | 15 | CI bootstrap klaster, sd antar seed, recall per kelas (khususnya MEL), confusion matrix, ECE |
| Laporan | 15 | `laporan/laporan.tex` (IEEE, ada bagian Keterbatasan) |
| Bisa diulang | 10 | §3, §9, notebook Colab, halaman `/eksperimen` |

## 11. Etika
Kedua dataset publik dan sudah dianonimkan. PAD-UFES-20 dikumpulkan dengan persetujuan komite etik UFES (No. 500002/478) dan Plataforma Brasil (No. 4.007.097). HAM10000 hanya untuk penggunaan non-komersial. **Model ini bukan alat diagnosis medis.**

