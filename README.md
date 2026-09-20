# Traffic Law RAG (Hệ thống RAG Pháp luật Giao thông Đường bộ)

Hệ thống Retrieval-Augmented Generation (RAG) phục vụ tra cứu, hỏi đáp văn bản quy phạm pháp luật giao thông đường bộ Việt Nam.

## 1. Dữ liệu văn bản pháp luật
- **Luật Trật tự, an toàn giao thông đường bộ 2024** (`36/2024/QH15`)
- **Nghị định 168/2024/NĐ-CP** (Xử phạt vi phạm hành chính về TTATGT đường bộ)
- **Nghị định 238/2026/NĐ-CP** (Sửa đổi, bổ sung một số điều của Nghị định 168/2024/NĐ-CP)

Tổng số chunk pháp lý: **2,305 chunks** (được phân đoạn chuẩn hóa cấp Điều / Khoản / Điểm kèm ngữ cảnh đầy đủ).

## 2. Retrieval Engines Hiện có

### 2.1. Dense Retrieval (Semantic Search)
- **Embedding Model:** `BAAI/bge-m3` (đa ngôn ngữ, hỗ trợ tiếng Việt sâu).
  - Dense vector dimension: **1024**
  - Normalization: **L2 normalize** (norm ≈ 1.0)
  - Device: Tự động sử dụng GPU CUDA (nếu có) hoặc CPU.
- **Vector Database:** **FAISS** (`IndexFlatIP`).
  - Sử dụng Inner Product trên vector đã L2-normalize, tương đương exact Cosine Similarity.
  - Ánh xạ trực tiếp chỉ mục FAISS position `0..N-1` sang danh sách metadata đầy đủ của từng chunk.
  - Xác thực tính toàn vẹn dữ liệu thông qua mã băm SHA-256 của corpus gốc.

### 2.2. Lexical Retrieval (Keyword Search)
- **Thuật toán:** **BM25Okapi** (`rank-bm25`).
- **Vietnamese Lexical Tokenizer:**
  - Chuẩn hóa Unicode NFC + lowercase.
  - Nhận diện và bảo toàn định dạng số thập phân (`0,25`), số tiền (`100.000`), ngày tháng (`26/12/2024`), và mã số văn bản (`168/2024/NĐ-CP`, `36/2024/QH15`).
  - Trích xuất unigram kết hợp consecutive word bigrams (ví dụ: `"mũ bảo hiểm"` → `["mũ", "bảo", "hiểm", "mũ_bảo", "bảo_hiểm"]`) phục vụ khớp cụm từ pháp lý.
  - Hoạt động hoàn toàn deterministic, không phụ thuộc mô hình NLP/LLM nặng.
- **Tính nhất quán:** 2,305 documents mapping 1:1 đồng nhất với chỉ mục FAISS.

### 2.3. Hybrid Retrieval (Reciprocal Rank Fusion - RRF)
- **Fusion:** **Reciprocal Rank Fusion (RRF)** thuần rank:
  $$RRF(d) = \sum_{r \in \{\text{dense}, \text{bm25}\}} \frac{1}{k + \text{rank}_r(d)} \quad (k = 60, \text{rank bắt đầu từ 1})$$
- **Candidate Pool:** `dense_top_k = 20`, `bm25_top_k = 20`, kết quả cuối `top_k = 5`.
- **Deduplication:** Khớp nối và loại bỏ trùng lặp dựa trên `chunk_id` chuẩn hóa.
- **Tie-Breaking:** 100% deterministic dựa trên ưu tiên overlap và phạt thứ hạng thiếu (`dense_top_k + 1`, `bm25_top_k + 1`).
- **Provenance:** Lưu vết đầy đủ thứ hạng và điểm số thành phần (`dense_rank`, `dense_score`, `bm25_rank`, `bm25_score`, `rrf_score`).

## 3. Cấu trúc lưu trữ
```
data/
├── chunks/
│   └── legal_chunks.json     # 2,305 chunks pháp luật chuẩn hóa
└── vector_store/
    ├── faiss.index           # Chỉ mục FAISS binary
    ├── metadata.json         # Metadata pháp lý đầy đủ tương ứng từng vector
    └── index_manifest.json   # Thông số kỹ thuật của index và mã hash SHA-256
```

## 4. Hướng dẫn sử dụng

### 4.1. Cài đặt dependencies
```bash
python -m pip install -r requirements.txt
```

### 4.2. Xây dựng FAISS Vector Store
Tạo chỉ mục vector từ `data/chunks/legal_chunks.json`:
```bash
python -m src.retrieval.faiss_store
```

### 4.3. Chạy Smoke Test
- **Dense Search (BGE-M3 + FAISS):**
  ```bash
  python scripts/smoke_test.py
  ```
- **Lexical Search (BM25Okapi):**
  ```bash
  python scripts/smoke_test_bm25.py
  ```
- **Hybrid Search (RRF):**
  ```bash
  python scripts/smoke_test_hybrid.py
  ```

### 4.4. Chạy kiểm thử toàn bộ (Pytest)
```bash
pytest -v
```

