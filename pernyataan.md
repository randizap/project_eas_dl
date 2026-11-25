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

**Yang TIDAK dilakukan AI:**
- AI tidak menjalankan eksperimen asli dan tidak menghasilkan angka hasil apa pun. Semua angka di laporan berasal dari `outputs/` hasil run kami sendiri.
- AI tidak mengakses dataset asli. Kode hanya diuji pada data sintetis.

**Verifikasi yang kami lakukan sendiri:** ✏️ (contoh: menjalankan `pytest` & `--smoke`; membaca setiap modul; memeriksa contact sheet duplikat secara manual; mencocokkan jumlah citra dengan publikasi dataset; memeriksa referensi)

**Perubahan yang kami buat terhadap kode AI:** ✏️

## 3. Pernyataan
Kami menyatakan bahwa isi di atas benar, bahwa kami memahami dan bisa menjelaskan seluruh kode yang dikumpulkan, dan bahwa angka hasil yang dilaporkan berasal dari eksperimen kami sendiri.

✏️ Tanda tangan / nama & tanggal: ______________________
