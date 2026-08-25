import hashlib

from fato_unb.ingestion.models import RawDocument
from fato_unb.rag.models import DocumentChunk


class SemanticChunker:
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Protege o split para não quebrar em abreviações comuns
        self.separators = ["\n\n", "\n", ". ", "; ", " "]

    def _split_into_base_units(self, text: str) -> list[str]:
        """Divide o texto preservando quebras de linha e sentenças."""
        if not text.strip():
            return []

        # Tenta separar por parágrafos primeiro
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [text.strip()]

        units: list[str] = []
        for p in paragraphs:
            # Se o parágrafo for maior que chunk_size palavras, quebra por sentenças
            if len(p.split()) > self.chunk_size:
                sentences = [
                    s.strip() for s in p.replace("\n", " ").split(". ") if s.strip()
                ]
                for s in sentences:
                    # Se ainda for grande, divide por palavras
                    words = s.split()
                    if len(words) > self.chunk_size:
                        for i in range(
                            0, len(words), self.chunk_size - self.chunk_overlap
                        ):
                            units.append(" ".join(words[i : i + self.chunk_size]))
                    else:
                        units.append(s + ".")
            else:
                units.append(p)
        return units

    def chunk_document(self, document: RawDocument) -> list[DocumentChunk]:
        if not document.content or not document.content.strip():
            return []

        units = self._split_into_base_units(document.content)
        if not units:
            return []

        raw_chunks: list[str] = []
        current_chunk_words: list[str] = []

        for unit in units:
            unit_words = unit.split()
            if len(current_chunk_words) + len(unit_words) <= self.chunk_size:
                current_chunk_words.extend(unit_words)
            else:
                if current_chunk_words:
                    raw_chunks.append(" ".join(current_chunk_words))
                    # Aplica o overlap pegando as últimas N palavras
                    overlap_words = (
                        current_chunk_words[-self.chunk_overlap :]
                        if self.chunk_overlap > 0
                        else []
                    )
                    current_chunk_words = list(overlap_words) + unit_words
                else:
                    raw_chunks.append(" ".join(unit_words))

        if current_chunk_words:
            raw_chunks.append(" ".join(current_chunk_words))

        total_chunks = len(raw_chunks)
        chunks: list[DocumentChunk] = []

        for idx, raw_text in enumerate(raw_chunks):
            enriched_content = (
                f"[Documento: {document.title}]\n"
                f"[Fonte: {document.source} | Ref: {document.semester_ref or 'Geral'}]\n\n"
                f"{raw_text}"
            )

            chunk_id_raw = f"{document.doc_id}_{idx}"
            chunk_id = hashlib.sha256(chunk_id_raw.encode("utf-8")).hexdigest()[:16]

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=document.doc_id,
                    content=enriched_content,
                    raw_text=raw_text,
                    chunk_index=idx,
                    total_chunks=total_chunks,
                    title=document.title,
                    url=document.url,
                    source=document.source,
                    semester_ref=document.semester_ref,
                )
            )

        return chunks
