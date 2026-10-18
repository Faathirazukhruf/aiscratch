import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==============================================================================
# MODEL AI LOKAL (OPEN-WEIGHTS)
# ==============================================================================
# Kita gunakan "SmolLM-135M-Instruct":
# 1. Ukuran file kecil (~270 MB), sangat ramah untuk laptop biasa/CPU.
# 2. Sudah melalui tahap pre-training dan instruction-tuning (paham format tanya jawab).
# 3. Berjalan 100% OFFLINE di komputermu tanpa perlu API key atau internet.
MODEL_NAME = "HuggingFaceTB/SmolLM-135M-Instruct"

print("==================================================================")
print("🤖 MEMUAT MODEL AI LOKAL KE MEMORI (CPU)...")
print("Catatan: Download bobot model hanya terjadi sekali di awal (~270MB).")
print("==================================================================")

# 1. Load Tokenizer (penerjemah kata <-> token angka)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 2. Load Arsitektur & Bobot Matematika Model ke RAM CPU
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32
)

print("\n✅ Model AI lokal siap digunakan!")
print("Ketik pertanyaanmu di bawah (Ketik 'keluar' atau 'exit' untuk berhenti).\n")

# Riwayat percakapan sederhana
messages = [
    {"role": "system", "content": "You are a helpful, friendly local AI assistant."}
]

while True:
    try:
        user_input = input("Kamu: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["keluar", "exit", "quit"]:
            print("Sampai jumpa! Semangat terus eksplorasi AI lokalnya! 👋")
            break

        # Tambahkan pertanyaan pengguna ke riwayat
        messages.append({"role": "user", "content": user_input})

        # Format prompt percakapan sesuai standar chat template
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Ubah teks menjadi Tensor PyTorch (Tokenization)
        inputs = tokenizer(prompt, return_tensors="pt")

        print("AI sedang berpikir...", end="\r", flush=True)

        # Proses Inference (Prediksi kata demi kata menggunakan rumus Self-Attention di CPU)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,        # Mengatur kreativitas (makin kecil makin fokus/deterministik)
                top_p=0.9,              # Nucleus sampling
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        # Potong prompt awal agar hanya menampilkan jawaban baru
        input_length = inputs["input_ids"].shape[1]
        response_tokens = outputs[0][input_length:]
        jawaban = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()

        print(f"AI: {jawaban}\n")

        # Simpan jawaban AI ke dalam riwayat percakapan agar ada memori konteks
        messages.append({"role": "assistant", "content": jawaban})

    except KeyboardInterrupt:
        print("\nSesi dihentikan. Sampai jumpa!")
        break
