"""Abstracción de almacenamiento vectorial con soporte para ChromaDB y respaldo persistente local.

Utiliza ChromaDB (`PersistentClient` + `sentence-transformers`) cuando el entorno dispone de
las dependencias compiladas (ej. Python 3.11/3.12 o contenedor Docker). En entornos como
Python 3.15 en Windows —donde `onnxruntime`/`chroma-hnswlib` aún no publican ruedas precompiladas—
proporciona automáticamente una colección vectorial persistente en disco con similitud coseno
sobre vectores TF/n-gramas normalizados, manteniendo exactamente la misma interfaz de ChromaDB.
"""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from app.config import Settings


def _normalize_text(text: str) -> str:
    """Normaliza texto en español eliminando tildes y convirtiendo a minúsculas."""
    nfkd = unicodedata.normalize("NFKD", text.lower())
    without_accents = "".join(c for c in nfkd if not unicodedata.combining(c))
    return without_accents


_STOPWORDS_ES = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "al",
    "a", "en", "con", "por", "para", "sin", "sobre", "entre", "hacia", "hasta",
    "que", "cual", "cuales", "como", "cuando", "donde", "quien", "es", "son",
    "fue", "ser", "esta", "estan", "este", "esta", "estos", "estas", "ese",
    "esa", "y", "o", "u", "e", "ni", "pero", "mas", "si", "no", "se", "su",
    "sus", "mi", "mis", "tu", "tus", "lo", "le", "les", "me", "te", "nos",
}


def _vectorize(text: str) -> Dict[str, float]:
    """Convierte un texto en un vector disperso L2-normalizado de palabras y n-gramas."""
    norm = _normalize_text(text)
    words = [w for w in re.findall(r"[a-z0-9]+", norm) if len(w) >= 2 and w not in _STOPWORDS_ES]
    counts: Counter[str] = Counter()
    for w in words:
        counts[f"w:{w}"] += 2.0
        if len(w) >= 4:
            # Prefijos/raíces para capturar variaciones morfológicas
            counts[f"p:{w[:4]}"] += 1.0
        if len(w) >= 5:
            counts[f"p:{w[:5]}"] += 1.0

    if not counts:
        # Fallback para textos muy cortos
        for token in re.findall(r"[a-z0-9]+", norm):
            counts[f"w:{token}"] += 1.0

    norm_sq = sum(v * v for v in counts.values())
    if norm_sq <= 0:
        return {}
    inv_norm = 1.0 / math.sqrt(norm_sq)
    return {k: v * inv_norm for k, v in counts.items()}


def _cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    """Calcula la similitud coseno entre dos vectores dispersos L2-normalizados."""
    if not vec_a or not vec_b:
        return 0.0
    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a
    dot = sum(val * vec_b.get(key, 0.0) for key, val in vec_a.items())
    return max(0.0, min(1.0, dot))


class LocalPersistentCollection:
    """Colección vectorial persistente en JSON compatible con la interfaz de ChromaDB."""

    def __init__(self, persist_dir: str, collection_name: str) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", collection_name)
        self.storage_path = self.persist_dir / f"collection_{safe_name}.json"
        if not self.storage_path.exists():
            self._save_items([])

    def _load_items(self) -> List[Dict[str, Any]]:
        try:
            if self.storage_path.exists():
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
        except Exception:
            pass
        return []

    def _save_items(self, items: List[Dict[str, Any]]) -> None:
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)

    def count(self) -> int:
        """Retorna el número total de fragmentos almacenados."""
        return len(self._load_items())

    def add(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Inserta o actualiza fragmentos con sus metadatos en el almacén persistente."""
        items = self._load_items()
        by_id: Dict[str, Dict[str, Any]] = {item["id"]: item for item in items}
        metadatas = metadatas or [{} for _ in ids]

        for doc_id, text, meta in zip(ids, documents, metadatas):
            by_id[doc_id] = {
                "id": doc_id,
                "document": text,
                "metadata": meta,
            }

        self._save_items(list(by_id.values()))

    def query(
        self,
        query_texts: List[str],
        n_results: int = 3,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, List[List[Any]]]:
        """Ejecuta una búsqueda por similitud vectorial y retorna resultados en formato ChromaDB."""
        items = self._load_items()
        if where:
            filtered = [
                it for it in items
                if all(it.get("metadata", {}).get(k) == v for k, v in where.items())
            ]
        else:
            filtered = items

        all_ids: List[List[str]] = []
        all_docs: List[List[str]] = []
        all_metas: List[List[Dict[str, Any]]] = []
        all_dists: List[List[float]] = []

        for q_text in query_texts:
            q_vec = _vectorize(q_text)
            scored: List[Tuple[float, Dict[str, Any]]] = []
            for it in filtered:
                doc_vec = _vectorize(it.get("document", ""))
                sim = _cosine_similarity(q_vec, doc_vec)
                dist = round(max(0.0, 1.0 - sim), 4)
                scored.append((dist, it))

            scored.sort(key=lambda pair: pair[0])
            top = scored[: max(0, n_results)]

            all_ids.append([it["id"] for _, it in top])
            all_docs.append([it["document"] for _, it in top])
            all_metas.append([it.get("metadata", {}) for _, it in top])
            all_dists.append([dist for dist, _ in top])

        return {
            "ids": all_ids,
            "documents": all_docs,
            "metadatas": all_metas,
            "distances": all_dists,
        }

    def delete(
        self,
        where: Optional[Dict[str, Any]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """Elimina fragmentos que coincidan con el filtro `where` o la lista `ids`."""
        items = self._load_items()
        remaining: List[Dict[str, Any]] = []
        id_set = set(ids or [])

        for it in items:
            if id_set and it.get("id") in id_set:
                continue
            if where and all(it.get("metadata", {}).get(k) == v for k, v in where.items()):
                continue
            remaining.append(it)

        self._save_items(remaining)


def create_vector_collection(settings: Settings) -> Any:
    """Inicializa y retorna una colección vectorial ChromaDB o el respaldo persistente local."""
    Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
    try:
        import chromadb  # type: ignore
        from chromadb.config import Settings as ChromaSettings  # type: ignore
        from chromadb.utils import embedding_functions  # type: ignore

        if not hasattr(chromadb, "PersistentClient"):
            raise RuntimeError("La versión instalada de chromadb no soporta PersistentClient.")

        client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        try:
            embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=settings.embedding_model
            )
        except Exception:
            embedding_fn = embedding_functions.DefaultEmbeddingFunction()

        collection = client.get_or_create_collection(
            name=settings.collection_name,
            embedding_function=embedding_fn,
            metadata={"description": "Índice de fragmentos académicos"},
        )
        # Verificar que la colección sea operativa
        _ = collection.count()
        return collection
    except Exception:
        return LocalPersistentCollection(
            persist_dir=settings.chroma_persist_dir,
            collection_name=settings.collection_name,
        )
