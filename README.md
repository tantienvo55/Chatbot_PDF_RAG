# Traffic Law RAG (Hệ thống RAG Pháp luật Giao thông Đường bộ)

Hệ thống Retrieval-Augmented Generation (RAG) phục vụ tra cứu, hỏi đáp văn bản quy phạm pháp luật giao thông đường bộ Việt Nam.

## 1. Dữ liệu văn bản pháp luật
- **Luật Trật tự, an toàn giao thông đường bộ 2024** (`36/2024/QH15`)
- **Nghị định 168/2024/NĐ-CP** (Xử phạt vi phạm hành chính về TTATGT đường bộ)
- **Nghị định 238/2026/NĐ-CP** (Sửa đổi, bổ sung một số điều của Nghị định 168/2024/NĐ-CP)

Tổng số chunk pháp lý: **2,305 chunks** (được phân đoạn chuẩn hóa cấp Điều / Khoản / Điểm kèm ngữ cảnh đầy đủ).

## 2. Dense Retrieval & Vector Database
- **Embedding Model:** `BAAI/bge-m3` (đa ngôn ngữ, hỗ trợ tiếng Việt sâu).
  - Dense vector dimension: **1024**
  - Normalization: **L2 normalize** (norm ≈ 1.0)
  - Device: Tự động sử dụng GPU CUDA (nếu có) hoặc CPU.
- **Vector Database:** **FAISS** (`IndexFlatIP`).
  - Sử dụng Inner Product trên vector đã L2-normalize, tương đương exact Cosine Similarity.
  - Ánh xạ trực tiếp chỉ mục FAISS position `0..N-1` sang danh sách metadata đầy đủ của từng chunk.
  - Xác thực tính toàn vẹn dữ liệu thông qua mã băm SHA-256 của corpus gốc.

## 3. Cấu trúc lưu trữ Vector Store
```
data/vector_store/
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

### 4.3. Chạy Smoke Test Dense Search
Truy vấn thử nghiệm 5 câu hỏi pháp lý mẫu:
```bash
python scripts/smoke_test.py
```

### 4.4. Chạy kiểm thử toàn bộ (Pytest)
```bash
pytest -v
```
