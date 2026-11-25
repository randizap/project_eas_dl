# Model Card — Klasifikasi Lesi Kulit 4 Kelas (HAM10000 → PAD-UFES-20)

Format mengikuti *Model Cards for Model Reporting* (Mitchell dkk. 2019), ditambah *Datasheet* untuk data turunan yang kami pakai (Gebru dkk. 2021). Bagian bertanda **[OTOMATIS]** diisi dari `outputs/results/results.md` setelah evaluasi. Jangan menulis angka dengan tangan.

---

## 1. Detail model
- **Pengembang:** Raditya Andiza Putro, Rosyidatul Kamila — Proyek Akhir Deep Learning, 2026.
- **Tiga konfigurasi:**
  1. *Baseline*: fitur DINOv2 ViT-S/14 beku (timm `vit_small_patch14_dinov2.lvd142m`) + regresi logistik berbobot kelas.
  2. *Model utama*: EfficientNet-B0 (torchvision `IMAGENET1K_V1`), *fine-tuning* penuh, 224 px, cross-entropy berbobot kelas, AdamW, jadwal kosinus dengan warmup.
  3. *Intervensi*: model utama + augmentasi karakteristik ponsel (zoom-out, warna/white balance, blur, JPEG).
- **Keluaran:** probabilitas 4 kelas: MEL (melanoma), NEV (nevus), BCC (karsinoma sel basal), AK (keratosis aktinik; lihat §6 tentang ketidakcocokan definisi).
- **Bobot:** lihat `weights.txt` (tautan + SHA-256). Lisensi bobot turunan mengikuti lisensi data latih: **non-komersial** (HAM10000, CC BY-NC 4.0).

## 2. Penggunaan yang dimaksud
- **Dimaksudkan untuk:** penelitian akademik tentang generalisasi lintas perangkat akuisisi, yaitu mengukur seberapa besar kinerja turun dari citra dermatoskopi ke foto ponsel.
- **TIDAK dimaksudkan untuk:** diagnosis, triase, skrining, atau keputusan klinis apa pun; aplikasi untuk pasien; penggunaan komersial.
- **Pengguna yang dimaksud:** dosen penilai dan peneliti ML.

## 3. Faktor yang relevan
- **Perangkat akuisisi** (faktor yang diteliti): dermatoskop (HAM10000) vs kamera ponsel (PAD-UFES-20).
- **Populasi & lokasi** (faktor perancu): HAM10000 berasal dari Wina (Austria) dan Queensland (Australia); PAD-UFES-20 dari Espírito Santo (Brasil), dari program layanan gratis untuk pasien berpenghasilan rendah. Selisih kinerja **tidak bisa** sepenuhnya diatribusikan ke perangkat, karena populasi, tipe kulit, distribusi lokasi lesi, dan proses pelabelan ikut berubah.
- **Prevalensi kelas:** NEV mendominasi HAM (≈ 77% dari 4 kelas), sedangkan BCC dan AK mendominasi PAD. Pergeseran prior ini memengaruhi macro-F1 lebih kuat daripada balanced accuracy.
- **Artefak dermatoskopi:** sudut vignette hitam, penggaris, gelembung gel, rambut, dan tinta penanda bedah. Artefak semacam ini diketahui bisa menjadi jalan pintas (Winkler dkk. 2019).

## 4. Metrik
- **Utama:** balanced accuracy dan macro-F1, rerata 3 seed, dengan CI 95% dari bootstrap klaster (klaster = lesi di HAM, pasien di PAD; B = 2000).
- **Sekunder:** recall per kelas (terutama MEL), confusion matrix, ECE (15 bin), balanced accuracy di bawah 9 korupsi × 5 severity, uji pengacakan Grad-CAM.
- **Alasan:** data sangat tidak seimbang, sehingga akurasi biasa akan didominasi NEV. Recall MEL penting secara klinis karena kesalahan *false negative* paling mahal.

## 5. Data evaluasi & data latih
Lihat **Datasheet** di bawah. Ringkasnya, latih/val/test hanya dari HAM10000 (split 70/15/15 per grup lesi), dan PAD-UFES-20 dipakai **hanya** sebagai uji OOD.

## 6. Analisis kuantitatif **[OTOMATIS]**
Salin tabel dari `outputs/results/results.md` ke sini setelah `python -m src.run_all --stages evaluate report`:

| Konfigurasi | BalAcc ID [CI] | BalAcc OOD [CI] | Celah [CI] | Recall MEL OOD [CI] | ECE OOD |
|---|---|---|---|---|---|
| Baseline | — | — | — | — | — |
| Model utama | — | — | — | — | — |
| Intervensi | — | — | — | — | — |

## 7. Pertimbangan etis
- Kedua dataset publik dan sudah dianonimkan. PAD-UFES-20 disetujui komite etik UFES (No. 500002/478) dan Plataforma Brasil (No. 4.007.097).
- **Risiko utama bila disalahgunakan sebagai alat klinis:** melanoma yang terlewat (false negative) bisa berakibat fatal. PAD hanya memuat **52** citra MEL, sehingga estimasi recall MEL di OOD sangat tidak pasti (CI lebar). Angka itu tidak boleh dibaca sebagai jaminan keamanan.
- **Keadilan:** kami tidak punya data tipe kulit yang seimbang dan tidak mengevaluasi kinerja per tipe Fitzpatrick. Model kemungkinan besar berkinerja lebih buruk pada kulit gelap, yang kurang terwakili di kedua dataset. Ini **asumsi** berdasarkan komposisi data, belum diuji.

## 8. Batasan & peringatan
- Kelas "AK" tidak identik antar domain: `akiec` HAM mencakup penyakit Bowen, sedangkan `ACK` PAD tidak.
- Label PAD untuk ACK/NEV sebagian besar berupa diagnosis klinis (≈ 24% biopsi), sehingga ada derau label di data uji OOD.
- Parameter augmentasi ponsel ditetapkan a priori, bukan dioptimalkan. Hasil negatif untuk intervensi **tidak** berarti augmentasi apa pun pasti gagal.
- Korupsi sintetis adalah implementasi ulang parameter ImageNet-C, jadi angkanya tidak setara dengan tabel ImageNet-C.
- Bobot DINOv2 dilatih pada LVD-142M (data web terkurasi). Kami **berasumsi**, tetapi tidak bisa menjamin, bahwa citra ISIC/HAM tidak ada di data pretraining tersebut.

---

# Datasheet — Data turunan yang dipakai proyek ini

**Motivasi.** Dibuat untuk mengukur celah generalisasi dari citra dermatoskopi ke foto ponsel dengan split yang bebas kebocoran. Pembuat: tim proyek. Tidak ada pendanaan.

**Komposisi.**
- Sumber: HAM10000 (10.015 citra dermatoskopi, 7 kelas) dan PAD-UFES-20 (2.298 foto ponsel, 6 kelas, 1.373 pasien).
- Yang dipakai: 4 kelas berpadanan (MEL, NEV, BCC, AK). Menurut publikasi asli, jumlahnya ±8.659 citra HAM dan ±1.871 citra PAD. Hitungan pasti tercetak oleh tahap `prepare` dan tersimpan di `outputs/splits/*.json`.
- Unit independen: lesi (HAM tidak punya ID pasien), pasien (PAD).
- Duplikat: dideteksi dengan pHash dan kosinus DINOv2, lalu diperiksa manual. Laporan ada di `outputs/leakage/report.json`.
- Data sensitif: citra kulit dan metadata klinis (usia, jenis kelamin, lokasi lesi). Tidak ada nama atau wajah. Kami tidak memakai metadata klinis sebagai fitur.

**Proses pengumpulan.** Tidak ada pengumpulan baru; kami hanya memakai rilis publik. Proses asli: HAM10000 dikumpulkan selama 20 tahun dari dua situs (Tschandl dkk. 2018); PAD-UFES-20 dari program PAD-UFES, Brasil (Pacheco dkk. 2020).

**Praproses.** Resize sisi pendek ke 256 px (bikubik). Format sumber dipertahankan (HAM JPEG q95, PAD PNG). Kelas di luar 4 kelas dibuang. Lesi yang punya citra hampir-kembar digabung menjadi satu grup split. Citra PAD yang kembar dengan HAM dibuang dari uji OOD.

**Penggunaan.** Hanya untuk eksperimen proyek ini. **Tidak cocok** untuk melatih alat klinis, dan tidak cocok untuk menilai keadilan lintas tipe kulit.

**Distribusi.** Kami **tidak** mendistribusikan ulang citra. Pengguna mengunduh sendiri dari sumber resmi dan mematuhi lisensinya (HAM10000: CC BY-NC 4.0; PAD-UFES-20: CC BY 4.0). Yang kami bagikan hanya kode, daftar split (ID citra), dan bobot.

**Pemeliharaan.** Tidak dipelihara setelah EAS. Kesalahan bisa dilaporkan ke anggota tim.
