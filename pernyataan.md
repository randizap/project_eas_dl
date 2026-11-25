# Pernyataan Kontribusi & Pemakaian AI

> Slide 3: bantuan AI **boleh**, asal **dilaporkan** dan kode **bisa dijelaskan sendiri**. Nilai per orang berasal dari tanya jawab dan pernyataan ini (slide 9). Isi bagian bertanda ✏️ dengan jujur. Jangan klaim pekerjaan yang tidak Anda lakukan.

## 1. Kontribusi anggota ✏️

| Pekerjaan | Raditya Andiza Putro | Rosyidatul Kamila |
|---|---|---|
| Proposal | ✏️ % | ✏️ % |
| Unduh & periksa data, verifikasi manual duplikat | ✏️ | ✏️ |
| Menjalankan & memantau training (Colab/Kaggle) | ✏️ | ✏️ |
| Analisis hasil & Grad-CAM | ✏️ | ✏️ |
| Penulisan laporan | ✏️ | ✏️ |
| Presentasi | ✏️ | ✏️ |

Bagian kode yang **masing-masing** siap jelaskan baris per baris saat tanya jawab: ✏️

## 2. Pemakaian AI

**Alat:** Claude (Anthropic), model Claude Opus 5.5, melalui aplikasi Claude, 25 November 2026.

**Yang dihasilkan AI (draf awal, lalu kami periksa):**
- Kerangka seluruh kode di `src/` (pipeline, deteksi duplikat, split, training, evaluasi, korupsi, Grad-CAM, pembuat tabel), `tests/`, dan `configs/default.yaml`.
- Draf `README.md`, `model_card.md`, dan kerangka `laporan/laporan.tex` (bagian pendahuluan, metode, dan keterbatasan; bagian hasil dan pembahasan sengaja dibiarkan kosong).
- Kritik terhadap proposal (ketidakcocokan kelas AK, pergeseran prior, n melanoma PAD yang kecil, tumpang-tindih korupsi dengan augmentasi, determinisme linear probe).
- Notebook Colab `notebooks/EAS_DL_Colab.ipynb`, modul ekspor `src/export_web.py` (results.json + model ONNX dengan kepala CAM), perbaikan penyimpanan checkpoint atomik di `src/train_cnn.py`, seluruh website di `web/` (React + ONNX Runtime Web), dan `vercel.json`.

**Yang dijalankan AI atas permintaan kami:**
- AI (Claude, lewat browser) mengoperasikan notebook Colab di akun Google kami: mengunduh dataset (token Kaggle dimasukkan sendiri oleh kami di Colab Secrets), menjalankan kebocoran/split, tuning, training, evaluasi, laporan, dan ekspor, lalu mengambil hasilnya dari Drive dan melakukan push ke GitHub. Angka hasil berasal dari run tersebut (`runs/log.csv`, `weights.txt`, `laporan/hasil_ringkas.md`), bukan dikarang.
- Selama proses: (1) aturan duplikat diperketat ke "pHash DAN kosinus" setelah split menolak grup duplikat berukuran 525 citra (sebelum training, sebelum test dibuka); (2) Google Drive penuh sehingga `best.pt` run `cnn_main_lesion_s2` tidak tersimpan; run itu dilatih ulang dengan seed yang sama (kurva validasi identik, epoch terbaik 7, val bal. acc 0,7749) sebelum test dibuka; (3) report & export dijalankan di runtime CPU karena kuota GPU habis.

**Yang TIDAK dilakukan AI:**
- AI tidak memilih model atau hiperparameter berdasarkan hasil test. Test set dibuka satu kali (`TEST_OPENED.json`, berisi SHA-256 tiap checkpoint).
- AI tidak menulis bagian hasil & pembahasan laporan; itu tugas kami berdasarkan `laporan/hasil_ringkas.md`.

**Verifikasi yang kami lakukan sendiri:** ✏️ (contoh: menjalankan `pytest` & `--smoke`; membaca setiap modul; memeriksa contact sheet duplikat secara manual; mencocokkan jumlah citra dengan publikasi dataset; memeriksa referensi)

**Perubahan yang kami buat terhadap kode AI:** ✏️

## 3. Pernyataan
Kami menyatakan bahwa isi di atas benar, bahwa kami memahami dan bisa menjelaskan seluruh kode yang dikumpulkan, dan bahwa angka hasil yang dilaporkan berasal dari eksperimen kami sendiri.

✏️ Tanda tangan / nama & tanggal: ______________________
