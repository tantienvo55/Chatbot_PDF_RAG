"""
Tests for BGE-M3 dense embedder module (src/retrieval/bge_embedder.py).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.retrieval.bge_embedder import BGEEmbedder, DEFAULT_MODEL_NAME


@pytest.fixture(scope="session")
def shared_embedder():
    """Session-scoped embedder fixture so model is loaded only once across all tests."""
    return BGEEmbedder()


class TestBGEModel:
    def test_model_name_default(self, shared_embedder):
        """Model identifier must be BAAI/bge-m3."""
        assert shared_embedder.model_name == DEFAULT_MODEL_NAME
        assert shared_embedder.model_name == "BAAI/bge-m3"

    def test_model_loaded_successfully(self, shared_embedder):
        """SentenceTransformer instance is initialized and not None."""
        assert shared_embedder.model is not None

    def test_dimension_is_dynamic_and_positive(self, shared_embedder):
        """Dimension is resolved dynamically from model and is positive (e.g. 1024)."""
        dim = shared_embedder.dimension
        assert isinstance(dim, int)
        assert dim > 0
        expected_dim = (
            shared_embedder.model.get_embedding_dimension()
            if hasattr(shared_embedder.model, "get_embedding_dimension")
            else shared_embedder.model.get_sentence_embedding_dimension()
        )
        assert dim == expected_dim

    def test_encode_documents_shape_and_dtype(self, shared_embedder):
        """Encoding documents produces correct (N, dimension) float32 array."""
        docs = [
            "Người tham gia giao thông phải đi bên phải theo chiều đi của mình.",
            "Xử phạt vi phạm hành chính về trật tự an toàn giao thông.",
        ]
        embeddings = shared_embedder.encode_documents(docs)
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, shared_embedder.dimension)
        assert embeddings.dtype == np.float32

    def test_encode_query_shape_and_dtype(self, shared_embedder):
        """Encoding a query produces (1, dimension) float32 array."""
        query = "Vượt đèn đỏ bị phạt bao nhiêu tiền?"
        q_vec = shared_embedder.encode_query(query)
        assert isinstance(q_vec, np.ndarray)
        assert q_vec.shape == (1, shared_embedder.dimension)
        assert q_vec.dtype == np.float32

    def test_no_nan_no_inf_in_embeddings(self, shared_embedder):
        """Embeddings must not contain NaN or infinite values."""
        texts = ["Biển báo hiệu đường bộ", "Đèn tín hiệu giao thông", "Vạch kẻ đường"]
        vecs = shared_embedder.encode_documents(texts)
        assert not np.isnan(vecs).any(), "Found NaN in document embeddings"
        assert not np.isinf(vecs).any(), "Found Inf in document embeddings"

        q_vec = shared_embedder.encode_query("Tốc độ tối đa trên cao tốc?")
        assert not np.isnan(q_vec).any(), "Found NaN in query embedding"
        assert not np.isinf(q_vec).any(), "Found Inf in query embedding"

    def test_normalized_vector_norm_is_one(self, shared_embedder):
        """Normalized embeddings must have L2 norm approximately equal to 1.0."""
        texts = [
            "Điều khiển phương tiện không có giấy phép lái xe.",
            "Quy định về nồng độ cồn đối với người lái xe.",
        ]
        doc_vecs = shared_embedder.encode_documents(texts)
        doc_norms = np.linalg.norm(doc_vecs, axis=1)
        np.testing.assert_allclose(doc_norms, 1.0, atol=1e-4)

        q_vec = shared_embedder.encode_query("Mức phạt uống rượu bia khi lái xe")
        q_norm = np.linalg.norm(q_vec, axis=1)
        np.testing.assert_allclose(q_norm, 1.0, atol=1e-4)

    def test_empty_documents_handling(self, shared_embedder):
        """Empty documents list returns empty 2D array of shape (0, dimension)."""
        empty_vecs = shared_embedder.encode_documents([])
        assert empty_vecs.shape == (0, shared_embedder.dimension)
        assert empty_vecs.dtype == np.float32

    def test_empty_query_raises_value_error(self, shared_embedder):
        """Empty or whitespace-only query raises ValueError."""
        with pytest.raises(ValueError):
            shared_embedder.encode_query("")

        with pytest.raises(ValueError):
            shared_embedder.encode_query("   ")

        with pytest.raises(ValueError):
            shared_embedder.encode_query(None)

    def test_model_caching_across_instances(self, shared_embedder):
        """Instantiating another BGEEmbedder reuses the class-level cached model."""
        new_embedder = BGEEmbedder()
        assert new_embedder.model is shared_embedder.model
