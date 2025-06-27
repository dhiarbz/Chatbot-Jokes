import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import T5Tokenizer, T5ForConditionalGeneration
from peft import PeftModel
import os
import pandas as pd

# --- Konfigurasi RAG ---
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    RAG_ENABLED = True
except ImportError:
    print("PERINGATAN: Pustaka untuk RAG (sentence-transformers, faiss-cpu, pandas) tidak ditemukan. RAG akan dinonaktifkan.")
    RAG_ENABLED = False

# ====================================================================
# Konfigurasi Path
# ====================================================================
MODEL_PATH = 'hasil_jokes_hf'
BASE_MODEL_NAME = 'google/flan-t5-small'
KNOWLEDGE_BASE_DIR = 'knowledge_base'
RETRIEVER_MODEL_NAME = 'all-MiniLM-L6-v2'

# ====================================================================
# Fungsi Terpadu untuk Memuat Semua Dependensi
# ====================================================================
def load_all_dependencies():
    """
    Memuat semua model (Generator & Retriever) dan membangun Vector DB 
    dari semua file di direktori knowledge_base.
    Ini adalah satu-satunya fungsi yang perlu dipanggil saat inisialisasi.
    """
    global RAG_ENABLED

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Memuat model ke perangkat: {device}")

    # 1. Muat Model Generator (T5 yang sudah di-fine-tune)
    try:
        tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH, legacy=False)
        print("Tokenizer berhasil dimuat dari path lokal.")
    except Exception:
        print("Gagal memuat tokenizer lokal, memuat dari model dasar.")
        tokenizer = T5Tokenizer.from_pretrained(BASE_MODEL_NAME, legacy=False)

    base_model = T5ForConditionalGeneration.from_pretrained(
        BASE_MODEL_NAME, torch_dtype=torch.float16 if str(device) == "cuda" else torch.float32,
    )
    base_model.resize_token_embeddings(len(tokenizer))
    generator_model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    generator_model.to(device)
    generator_model.eval()
    print("✅ Model Generator (T5 + PEFT) berhasil dimuat.")

    # 2. Muat Model Retriever dan Bangun Vector DB (jika RAG aktif)
    retriever_model, faiss_index, structured_documents = None, None, []
    if RAG_ENABLED:
        try:
            print(f"\nMemuat model Retriever: {RETRIEVER_MODEL_NAME}...")
            retriever_model = SentenceTransformer(RETRIEVER_MODEL_NAME, device=device)
            print(f"Memindai direktori knowledge base: '{KNOWLEDGE_BASE_DIR}'")
            
            if not os.path.exists(KNOWLEDGE_BASE_DIR):
                print(f"PERINGATAN: Direktori '{KNOWLEDGE_BASE_DIR}' tidak ditemukan.")
            else:
                all_texts_for_embedding = []
                for filename in sorted(os.listdir(KNOWLEDGE_BASE_DIR)):
                    filepath = os.path.join(KNOWLEDGE_BASE_DIR, filename)
                    file_category = os.path.splitext(filename)[0].replace("_", " ").title()

                    if filename.endswith('.csv'):
                        df = pd.read_csv(filepath)
                        print(f"Memproses '{filename}' ({len(df)} baris)...")
                        for _, row in df.iterrows():
                            row_text = ' '.join(str(item) for item in row.values if pd.notna(item))
                            full_text = f"Konteks dari kategori '{file_category}': {row_text}"
                            all_texts_for_embedding.append(full_text)
                            structured_documents.append({'source': filename, 'content': full_text})
                    
                    elif filename.endswith('.txt'):
                        print(f"Memproses '{filename}'...")
                        with open(filepath, 'r', encoding='utf-8') as f:
                            for line in f:
                                line = line.strip()
                                if line:
                                    full_text = f"Konteks dari kategori '{file_category}': {line}"
                                    all_texts_for_embedding.append(full_text)
                                    structured_documents.append({'source': filename, 'content': full_text})
                
                if all_texts_for_embedding:
                    print(f"\nTotal {len(all_texts_for_embedding)} dokumen ditemukan. Membangun Vector DB...")
                    doc_embeddings = retriever_model.encode(all_texts_for_embedding, convert_to_tensor=True, show_progress_bar=True)
                    embedding_dim = doc_embeddings.shape[1]
                    faiss_index = faiss.IndexFlatL2(embedding_dim)
                    faiss_index.add(doc_embeddings.cpu().numpy())
                    print("✅ Vector Database berhasil dibuat dan siap digunakan.")
                else:
                    print("PERINGATAN: Tidak ada dokumen valid (.csv atau .txt) yang ditemukan di knowledge base.")

        except Exception as e:
            print(f"ERROR saat inisialisasi RAG: {e}. RAG akan dinonaktifkan.")
            RAG_ENABLED = False
            
    # Kembalikan semua komponen yang dibutuhkan oleh aplikasi
    return generator_model, tokenizer, device, retriever_model, faiss_index, structured_documents

# ====================================================================
# Inisialisasi Aplikasi Flask & Muat Semua Dependensi
# ====================================================================
app = Flask(__name__)
CORS(app)

# Panggil satu fungsi terpadu untuk memuat semuanya
model, tokenizer, device, retriever, index, docs = load_all_dependencies()

# ====================================================================
# Definisikan Endpoint API untuk Chatbot
# ====================================================================
@app.route("/chat", methods=["POST"])
def chat_handler():
    """
    Endpoint untuk menerima request dari frontend dan memberikan respons dari model.
    """
    try:
        data = request.get_json()
        if not data or "prompt" not in data or not data["prompt"]:
            return jsonify({"error": "Input tidak valid. Harap sediakan JSON dengan key 'prompt'."}), 400

        user_prompt = data["prompt"]
        final_prompt = user_prompt

        # Logika RAG
        if RAG_ENABLED and index is not None and docs:
            print("\nMode RAG aktif. Mencari konteks yang relevan...")
            
            # 1. RETRIEVE
            query_embedding = retriever.encode([user_prompt], convert_to_tensor=True)
            distances, indices = index.search(query_embedding.cpu().numpy(), k=3)
            
            retrieved_docs_content = [docs[i]['content'] for i in indices[0]]
            context = "\n\n---\n\n".join(retrieved_docs_content)

            # 2. AUGMENT
            final_prompt = f"""Anda adalah asisten AI yang cerdas dan lucu.
Gunakan contoh-contoh konteks berikut untuk membantu Anda memberikan jawaban yang relevan dan kreatif.

Konteks yang Ditemukan:
\"\"\"
{context}
\"\"\"

Berdasarkan konteks di atas, jawab atau buatlah lelucon yang sesuai dengan permintaan pengguna berikut: "{user_prompt}"
"""
            print(f"Konteks ditemukan dari sumber: {[docs[i]['source'] for i in indices[0]]}")
        else:
            print("\nMode RAG dinonaktifkan atau knowledge base kosong. Menggunakan prompt asli.")

        # 3. GENERATE
        inputs = tokenizer(final_prompt, return_tensors="pt", max_length=1024, truncation=True).to(device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                max_length=150,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.9,
                repetition_penalty=1.2
            )
        
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return jsonify({"response": response_text})

    except Exception as e:
        print(f"Terjadi error: {e}")
        return jsonify({"error": "Terjadi kesalahan di server."}), 500

# Endpoint untuk mengecek server
@app.route("/", methods=["GET"])
def health_check():
    return "Backend chatbot aktif dan berjalan!"

# ====================================================================
# Jalankan Aplikasi
# ====================================================================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
