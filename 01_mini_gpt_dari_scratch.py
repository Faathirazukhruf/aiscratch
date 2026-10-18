import torch
import torch.nn as nn
from torch.nn import functional as F

# ==============================================================================
# LANGKAH 1: DATASET PERCAKAPAN MINI
# ==============================================================================
# AI belajar dari pola teks. Kita siapkan contoh percakapan sederhana.
data_percakapan = """
User: Halo!
AI: Halo juga! Ada yang bisa kubantu?
User: Siapa kamu?
AI: Aku adalah model AI lokal buatanmu dari nol!
User: Kamu jalan di mana?
AI: Aku berjalan 100% di komputer lokalmu tanpa internet.
User: Apa rahasiamu?
AI: Rahasiaku adalah rumus matematika Attention dan prediksi token berikutnya.
User: Keren banget!
AI: Terima kasih! Selamat belajar AI dari scratch!
"""

# ==============================================================================
# LANGKAH 2: TOKENIZER (MENGUBAH TEKS MENJADI ANGKA)
# ==============================================================================
# Komputer tidak paham huruf, hanya paham angka.
chars = sorted(list(set(data_percakapan)))
vocab_size = len(chars)

# Kamus karakter -> angka dan sebaliknya
char_ke_id = {ch: i for i, ch in enumerate(chars)}
id_ke_char = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [char_ke_id[c] for c in s]
decode = lambda l: "".join([id_ke_char[i] for i in l])

print(f"--- 1. TOKENIZER INFO ---")
print(f"Total karakter unik (Vocab Size): {vocab_size}")
print(f"Contoh encode 'Halo': {encode('Halo')}")
print(f"Contoh decode balik: '{decode(encode('Halo'))}'\n")

# Siapkan data dalam bentuk PyTorch Tensor
data_tensor = torch.tensor(encode(data_percakapan), dtype=torch.long)


# ==============================================================================
# LANGKAH 3: ARSITEKTUR MINI TRANSFORMER (OTAK AI)
# ==============================================================================
# Ini adalah komponen inti yang dipakai oleh GPT-4, LLaMA, dan Claude!
n_embd = 64        # Ukuran representasi vektor untuk tiap karakter
block_size = 32    # Panjang konteks teks yang diingat (context length)
n_head = 4         # Jumlah perhatian (Multi-Head Attention)
n_layer = 3        # Kedalaman layer transformer
learning_rate = 1e-3
epochs = 300       # Jumlah putaran latihan

class Head(nn.Module):
    """Satu kepala Self-Attention: menghitung seberapa terkait satu kata dengan kata lainnya"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Masking segitiga bawah: AI tidak boleh 'menyontek' kata di masa depan!
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        
        # Hitung attention scores (Dot product query x key)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        # Terapkan masking segitiga
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        
        # Kalikan dengan value
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """Menggabungkan beberapa kepala attention agar bisa melihat banyak konteks sekaligus"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

class FeedForward(nn.Module):
    """Jaringan saraf sederhana setelah attention untuk memproses informasi"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Satu Blok Transformer = LayerNorm + MultiHeadAttention + FeedForward"""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    """Model Lengkap Mini-GPT Dari Scratch"""
    def __init__(self):
        super().__init__()
        # Token Embedding (Karakter -> Vektor)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # Position Embedding (Posisi urutan kata -> Vektor)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # Tumpukan blok transformer
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        # Linear layer terakhir untuk memprediksi karakter berikutnya
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T, C)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """Membuat teks baru dengan menebak karakter berikutnya satu per satu"""
        for _ in range(max_new_tokens):
            # Batasi panjang input sesuai block_size
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            # Ambil prediksi untuk langkah terakhir
            logits = logits[:, -1, :]
            # Ubah skor logits jadi probabilitas (0% - 100%)
            probs = F.softmax(logits, dim=-1)
            # Ambil sampel karakter berikutnya
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ==============================================================================
# LANGKAH 4: PROSES TRAINING (MELATIH OTAK AI)
# ==============================================================================
def get_batch(batch_size=8):
    # Mengambil potongan data secara acak untuk dipelajari
    ix = torch.randint(len(data_tensor) - block_size, (batch_size,))
    x = torch.stack([data_tensor[i:i+block_size] for i in ix])
    # Target adalah karakter berikutnya (geser 1 posisi ke depan)
    y = torch.stack([data_tensor[i+1:i+block_size+1] for i in ix])
    return x, y

model = MiniGPT()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"--- 2. MEMULAI PROSES TRAINING ---")
print(f"Jumlah parameter model: {sum(p.numel() for p in model.parameters()):,} bobot matematika")
print("Melatih model agar memahami pola percakapan...")

for epoch in range(1, epochs + 1):
    xb, yb = get_batch()

    # 1. Forward pass (tebak)
    logits, loss = model(xb, yb)

    # 2. Reset gradien sebelumnya
    optimizer.zero_grad(set_to_none=True)

    # 3. Backpropagation (hitung kesalahan dan arah koreksi bobot)
    loss.backward()

    # 4. Update bobot (belajar)
    optimizer.step()

    if epoch % 50 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{epochs} | Loss (Tingkat Error): {loss.item():.4f}")

print("\n--- 3. TRAINING SELESAI! ---")
print("Loss berhasil turun drastis (model sudah belajar pola!).\n")

# ==============================================================================
# LANGKAH 5: MENCOBA HASIL PREDIKSI (GENERATE TEKS)
# ==============================================================================
print("--- 4. MENGUJI AI MEMBALAS TEKS ---")
prompt = "User: Siapa kamu?\nAI:"
print(f"Prompt yang kita berikan:\n{prompt}")

context = torch.tensor([encode(prompt)], dtype=torch.long)
output_ids = model.generate(context, max_new_tokens=60)[0].tolist()

print("\nHasil respons AI:")
print("------------------------------------------")
print(decode(output_ids))
print("------------------------------------------")
