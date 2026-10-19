# AI Scratch 🧠 - Belajar AI Lokal Tanpa API

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CPU%20Ready-EE4C2C.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E.svg)](https://huggingface.co/)
[![Offline](https://img.shields.io/badge/Offline-100%25%20Local-success.svg)](#)
[![No API Key](https://img.shields.io/badge/API%20Key-Not%20Required-brightgreen.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Repositori ini dibuat sebagai materi pembelajaran praktis untuk memahami, membangun, dan menjalankan AI percakapan (*Large Language Model / LLM*) **dari prinsip pertama (*first principles*)**. 

Semua kode berjalan **100% lokal di komputermu (CPU)** tanpa membungkus (*wrapping*) API berbayar seperti OpenAI GPT atau Google Gemini, tanpa kuota token, dan tanpa mengirimkan data ke internet.

---

## 📑 Daftar Isi

- [Latar Belakang & Filosofi](#-latar-belakang--filosofi)
- [Struktur Proyek](#-struktur-proyek)
- [Kebutuhan Sistem](#-kebutuhan-sistem)
- [Panduan Instalasi & Persiapan](#-panduan-instalasi--persiapan)
- [Eksperimen 1: Mini-GPT Dari Scratch](#-eksperimen-1-mini-gpt-dari-scratch-01_mini_gpt_dari_scratchpy)
  - [Komponen Arsitektur](#komponen-arsitektur-yang-dibedah)
  - [Cara Menjalankan](#cara-menjalankan-eksperimen-1)
  - [Contoh Log Training & Output](#contoh-output-training)
- [Eksperimen 2: Chatbot Q&A Interaktif Lokal](#-eksperimen-2-chatbot-qa-interaktif-lokal-02_local_chatbot_interaktifpy)
  - [Tentang Model SmolLM-135M](#mengapa-smollm-135m-instruct)
  - [Fitur Utama](#fitur-utama)
  - [Cara Menjalankan](#cara-menjalankan-eksperimen-2)
- [Tabel Komparasi Eksperimen](#-tabel-komparasi-eksperimen)
- [Tips & Troubleshooting](#-tips--troubleshooting)
- [Roadmap Pengembangan](#-roadmap-pengembangan)
- [Kontribusi & Lisensi](#-kontribusi--lisensi)

---

## 💡 Latar Belakang & Filosofi

Kebanyakan tutorial AI saat ini hanya mengajarkan cara menginstal pustaka API pihak ketiga (`pip install openai`), memasukkan API key, dan memanggil endpoint. Pendekatan ini membuat AI terasa seperti sebuah **"kotak hitam" (black box)** yang misterius.

Melalui proyek ini, kita membongkar kotak hitam tersebut menjadi dua tahapan belajar:
1. **Memahami Fondasi Matematika**: Membangun arsitektur Transformer murni menggunakan PyTorch dari baris nol (tokenisasi, self-attention, layer normalization, residual stream, training loop, hingga autoregressive generation).
2. **Pemanfaatan Model Open-Weights**: Menjalankan model bahasa kecil (*Small Language Model*) siap pakai langsung di RAM komputer sendiri secara *air-gapped* (offline penuh).

---

## 📁 Struktur Proyek

```
Aiscratch/
├── .venv/                          # Virtual environment Python (lokal)
├── 01_mini_gpt_dari_scratch.py     # Eksperimen 1: Arsitektur Transformer murni dari nol
├── 02_local_chatbot_interaktif.py  # Eksperimen 2: Chatbot Q&A interaktif lokal (SmolLM-135M)
├── requirements.txt                # Daftar dependensi utama (PyTorch CPU, Transformers)
├── .gitignore                      # Berkas pengecualian Git (cache, weights, venv)
└── README.md                       # Dokumentasi lengkap & panduan belajar
```

---

## 💻 Kebutuhan Sistem

- **Sistem Operasi**: Windows 10/11, macOS, atau Linux.
- **Python**: Versi `3.10`, `3.11`, atau `3.12`.
- **Prosesor (CPU)**: Prosesor laptop/PC standar (Intel Core i3/i5/i7/i9 atau AMD Ryzen).
- **RAM**: Minimum 4 GB (disarankan 8 GB atau lebih).
- **GPU**: **Tidak diperlukan sama sekali**. Seluruh eksperimen dioptimalkan untuk CPU.
- **Ruang Disk**: ~500 MB (untuk cache bobot model SmolLM-135M dan pustaka Python).

---

## 🚀 Panduan Instalasi & Persiapan

### 1. Kloning Repositori
```bash
git clone https://github.com/Faathirazukhruf/aiscratch.git
cd aiscratch
```

### 2. Buat & Aktifkan Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(Jika muncul peringatan script policy di PowerShell, jalankan terlebih dahulu: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalasi Dependensi

Untuk menghemat kuota dan ruang disk, instal **PyTorch versi CPU** bersama pustaka `transformers`:

**Instalasi Cepat via requirements.txt:**
```bash
pip install -r requirements.txt
```

**Atau instalasi manual PyTorch CPU khusus:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers
```

---

## 🔬 Eksperimen 1: Mini-GPT Dari Scratch (`01_mini_gpt_dari_scratch.py`)

File ini membongkar arsitektur dasar Transformer model (arsitektur yang sama yang menjadi fondasi GPT-4, LLaMA, Mistral, dan Claude) hanya dengan menggunakan operasi tensor PyTorch.

### Diagram Alur Transformer Sederhana
```
Input Teks ("Halo")
       │
       ▼
 [ Tokenizer ] ───────────► Konversi Karakter ke ID Angka
       │
       ▼
 [ Token + Position Emb ] ─► Vektor Makna + Koordinat Posisi
       │
  ┌────┴─────────────────────────────┐
  │ Transformer Block (x3 Layer)     │
  │  ├── LayerNorm                   │
  │  ├── Multi-Head Self-Attention   │  <-- Query, Key, Value + Triangular Mask
  │  ├── Residual Add (x + Attn(x))  │
  │  ├── LayerNorm                   │
  │  ├── Feed-Forward Network (ReLU) │
  │  └── Residual Add (x + FFWD(x))  │
  └────┬─────────────────────────────┘
       │
       ▼
 [ Linear Head + Softmax ] ─► Probabilitas Karakter Berikutnya
       │
       ▼
 Karakter Baru ("!")
```

### Komponen Arsitektur yang Dibedah:
1. **Character-Level Tokenizer**:
   Membangun fungsi `encode` dan `decode` berbasis kamus karakter unik (tanpa modul pihak ketiga).
2. **Positional Encoding & Embedding**:
   Memberikan koordinat posisi spasial pada setiap token agar model memahami urutan kata dalam kalimat.
3. **Causal Multi-Head Self-Attention (`Head` & `MultiHeadAttention`)**:
   - Menghitung interaksi antar-kata melalui perkalian matriks Query ($Q$), Key ($K$), dan Value ($V$).
   - Dilengkapi *triangular lower mask* (`torch.tril`) agar model tidak bisa "menyontek" kata di masa depan (*autoregressive causal constraint*).
4. **FeedForward Layer & Layer Normalization**:
   Jaringan saraf linier dengan fungsi aktivasi ReLU serta LayerNorm untuk menstabilkan aliran gradien selama proses backpropagation.
5. **Residual Connections**:
   Jalur pintas penjumlahan ($x + f(x)$) untuk mencegah hilangnya gradien (*vanishing gradient*) pada lapisan dalam.
6. **Training Loop & Loss Monitoring**:
   Menggunakan *AdamW optimizer* dan *CrossEntropyLoss*. Nilai error (*loss*) dilatih untuk turun dari ~3.7 (tebakan acak) ke ~0.4 (memahami pola bahasa).

### Cara Menjalankan Eksperimen 1:
```powershell
python 01_mini_gpt_dari_scratch.py
```
*(Atau gunakan path langsung interpreter: `.\.venv\Scripts\python.exe 01_mini_gpt_dari_scratch.py`)*

### Contoh Output Training:
```text
--- 1. TOKENIZER INFO ---
Total karakter unik (Vocab Size): 46
Contoh encode 'Halo': [11, 23, 34, 37]
Contoh decode balik: 'Halo'

--- 2. MEMULAI PROSES TRAINING ---
Jumlah parameter model: 161,006 bobot matematika
Melatih model agar memahami pola percakapan...
Epoch   1/300 | Loss (Tingkat Error): 3.8214
Epoch  50/300 | Loss (Tingkat Error): 2.1402
Epoch 100/300 | Loss (Tingkat Error): 1.1290
Epoch 150/300 | Loss (Tingkat Error): 0.6721
Epoch 200/300 | Loss (Tingkat Error): 0.4312
Epoch 250/300 | Loss (Tingkat Error): 0.3891
Epoch 300/300 | Loss (Tingkat Error): 0.3540

--- 3. TRAINING SELESAI! ---
Loss berhasil turun drastis (model sudah belajar pola!).

--- 4. MENGUJI AI MEMBALAS TEKS ---
Prompt yang kita berikan:
User: Siapa kamu?
AI:

Hasil respons AI:
------------------------------------------
Aku adalah model AI lokal buatanmu dari nol!
------------------------------------------
```

---

## 💬 Eksperimen 2: Chatbot Q&A Interaktif Lokal (`02_local_chatbot_interaktif.py`)

Setelah memahami arsitekturnya di Eksperimen 1, eksperimen ini beralih ke penerapan praktis: menjalankan model bahasa terbuka (*open-weights*) yang sudah matang di komputer lokal.

### Mengapa SmolLM-135M-Instruct?
- **Ukuran Sangat Ringkas**: Hanya **~270 MB** (mudah diunduh dan tidak membebani memori).
- **Performa di CPU Cepat**: Dapat berjalan mulus di CPU laptop standar tanpa lag yang mengganggu.
- **Instruction-Tuned**: Sudah dilatih secara khusus untuk memahami format dialog tanya-jawab (*role-play chat*).

### Fitur Utama:
- 🔒 **100% Offline & Privat**: Tidak ada data percakapan yang dikirim ke server luar; inferensi berjalan langsung di memori RAM komputermu.
- 🔄 **Memori Konteks Percakapan (Multi-turn)**: Riwayat obrolan disimpan secara dinamis, sehingga AI dapat mengingat pertanyaan atau konteks sebelumnya.
- 🎛️ **Sampling Parameters yang Disesuaikan**:
  - `temperature = 0.7`: Memberikan keseimbangan antara variasi kreatif dan konsistensi jawaban.
  - `top_p = 0.9`: Nucleus sampling untuk memfilter kata-kata dengan probabilitas rendah agar jawaban tetap koheren.
  - `max_new_tokens = 150`: Membatasi panjang respons agar inferensi tetap gesit.

### Cara Menjalankan Eksperimen 2:
```powershell
python 02_local_chatbot_interaktif.py
```
*(Atau: `.\.venv\Scripts\python.exe 02_local_chatbot_interaktif.py`)*

> **Catatan Unduhan Pertama:** Pada eksekusi pertama, bobot model (~270 MB) akan otomatis diunduh satu kali dari Hugging Face Hub dan disimpan di cache lokal komputermu. Setelah itu, program dapat dijalankan tanpa koneksi internet sama sekali.

Untuk mengakhiri sesi percakapan, ketik `keluar`, `exit`, atau `quit`.

---

## 📊 Tabel Komparasi Eksperimen

| Aspek | Eksperimen 1: Mini-GPT Dari Scratch | Eksperimen 2: Local Chatbot (SmolLM) |
| :--- | :--- | :--- |
| **Tujuan Utama** | Edukasi arsitektur & matematika Transformer | Aplikasi chatbot interaktif siap pakai |
| **Arsitektur** | Custom PyTorch (3 Layer, 4 Head) | LLaMA-based architecture (SmolLM) |
| **Jumlah Parameter** | ~160 Ribu Parameter | 135 Juta Parameter (~270 MB) |
| **Tokenizer** | Character-level (Kamus Karakter Lokal) | Byte-level BPE Tokenizer (Hugging Face) |
| **Data Latih** | Mini dataset percakapan internal | Triliunan token data pre-training |
| **Kebutuhan RAM** | < 100 MB | ~500 MB - 1 GB RAM |
| **Kecepatan di CPU**| Instan (< 1 detik) | ~1-3 detik per respons |
| **Internet Saat Jalan**| 0% (Sepenuhnya offline) | 0% (Hanya perlu download bobot di awal) |

---

## 🛠️ Tips & Troubleshooting

### 1. Peringatan PowerShell: "running scripts is disabled on this system"
Jika kamu mendapati error saat mengaktifkan `.venv\Scripts\Activate.ps1`, jalankan perintah berikut di PowerShell:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 2. Menjalankan Langsung Tanpa Aktivasi Virtual Environment
Kamu selalu bisa langsung mengeksekusi Python di dalam virtual environment tanpa harus mengaktifkannya terlebih dahulu:
```powershell
.\.venv\Scripts\python.exe 01_mini_gpt_dari_scratch.py
.\.venv\Scripts\python.exe 02_local_chatbot_interaktif.py
```

### 3. Di Mana Bobot Model SmolLM Disimpan?
Bobot model Hugging Face disimpan otomatis di folder cache profil penggunamu:
- **Windows**: `C:\Users\<NamaUser>\.cache\huggingface\hub\`
- **Linux/macOS**: `~/.cache/huggingface/hub/`

Kamu dapat menghapus folder cache ini kapan saja jika ingin mengosongkan ruang disk.

---

## 🗺️ Roadmap Pengembangan

Beberapa rencana materi dan eksperimen yang dapat ditambahkan berikutnya:
- [ ] **Eksperimen 3: Tokenizer BPE (Byte-Pair Encoding)** menggunakan pustaka `tiktoken` / `tokenizers`.
- [ ] **Eksperimen 4: Kuantisasi Model (GGUF 4-bit)** untuk menjalankan model yang lebih besar (seperti Qwen2.5 1.5B / LLaMA-3.2 1B) di RAM 4 GB dengan `llama.cpp`.
- [ ] **Eksperimen 5: Fine-Tuning Ringan (LoRA / PEFT)** untuk mengajarkan persona atau dataset kustom berbahasa Indonesia.
- [ ] **Eksperimen 6: Antarmuka Web Chat Sederhana** menggunakan Streamlit atau Vanilla HTML/CSS/JS.

---

## 📄 Kontribusi & Lisensi

Kontribusi, perbaikan kode (*pull requests*), atau ide eksperimen baru sangat dipersilakan! Silakan buat *issue* atau kirimkan *pull request*.

Proyek ini didistribusikan di bawah lisensi **MIT License**. Bebas dipelajari, dimodifikasi, dan dibagikan untuk kepentingan edukasi dan riset personal.
