# AI Scratch - Belajar AI Lokal Tanpa API

Proyek ini dibuat untuk belajar membuat dan menjalankan AI percakapan (Chatbot) sendiri dari nol, **100% lokal di komputermu (CPU) tanpa wrapping API eksternal (seperti OpenAI/Gemini)**.

---

## 📁 Struktur Proyek

```
Aiscratch/
├── .venv/                          # Virtual environment Python 3.12 (PyTorch CPU + Transformers)
├── 01_mini_gpt_dari_scratch.py     # Eksperimen 1: Bikin arsitektur Transformer murni dari nol
├── 02_local_chatbot_interaktif.py  # Eksperimen 2: Chatbot Q&A lokal interaktif (SmolLM-135M)
└── README.md                       # Panduan & rangkuman materi
```

---

## 🔬 Eksperimen 1: Mini-GPT Dari Nol (`01_mini_gpt_dari_scratch.py`)

File ini membongkar matematika di balik LLM modern:
1. **Tokenizer**: Mengubah karakter teks menjadi angka index tensor.
2. **Positional Encoding & Embedding**: Memberi koordinat posisi kata dalam kalimat.
3. **Multi-Head Self-Attention**: Algoritma yang menghitung bobot keterkaitan antar kata (Query, Key, Value).
4. **FeedForward Layer & LayerNorm**: Jaringan syaraf penstabil nilai gradien.
5. **Training Loop & Backpropagation**: Menggunakan *AdamW* dan *CrossEntropyLoss* untuk menurunkan nilai eror (*loss*) dari ~3.7 ke ~0.4.

### Cara Menjalankan:
```powershell
.\.venv\Scripts\python.exe 01_mini_gpt_dari_scratch.py
```

---

## 💬 Eksperimen 2: Chatbot Q&A Interaktif Lokal (`02_local_chatbot_interaktif.py`)

File ini menjalankan model *open-weights* berukuran mini (**SmolLM-135M-Instruct**):
- **Ukuran:** Hanya ~270 MB (sangat ringan di RAM dan hemat ruang).
- **Komputasi:** Berjalan 100% pada CPU laptop/PC tanpa kartu grafis khusus.
- **Kemandirian:** 0% koneksi internet/API saat inference berlangsung.
- **Fitur:** Mendukung memori percakapan berulang (*context window*).

### Cara Menjalankan Chatbot Interaktif:
Buka PowerShell / Terminal di folder ini, lalu jalankan:
```powershell
.\.venv\Scripts\python.exe 02_local_chatbot_interaktif.py
```
Ketik pertanyaanmu, dan ketik `keluar` untuk selesai.
