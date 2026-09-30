import hashlib
import re

from fato_unb.ingestion.models import RawDocument
from fato_unb.rag.models import DocumentChunk

# Abreviações após as quais um ponto NÃO encerra a frase.
_ABBREVIATIONS = {
    "prof", "profa", "profs", "profas", "dr", "dra", "sr", "sra", "exmo", "exma",
    "art", "arts", "inc", "nº", "cap", "fl", "obs", "ltda", "cia", "av", "pág", "vol", "cf",
}  # fmt: skip

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…])\s+(?=[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ“\"'(\[])")
_TITLE_PREFIX = re.compile(r"^UnB Notícias\s*[-–]\s*")


def _split_sentences(line: str) -> list[str]:
    """Divide uma linha em frases sem quebrar em abreviações ("Profa.", "Art.", "J. Silva")."""
    pieces = _SENTENCE_BOUNDARY.split(line)
    sentences: list[str] = []
    for piece in pieces:
        if sentences:
            last_token = sentences[-1].split()[-1].rstrip(".").lower()
            is_initial = len(last_token) == 1 and last_token.isalpha()
            if last_token in _ABBREVIATIONS or is_initial:
                sentences[-1] = f"{sentences[-1]} {piece}"
                continue
        sentences.append(piece)
    return [s.strip() for s in sentences if s.strip()]


class _Unit:
    """Uma frase e a informação de que ela abre uma linha (parágrafo) do texto original."""

    __slots__ = ("text", "words", "starts_line")

    def __init__(self, text: str, starts_line: bool):
        self.text = text
        self.words = len(text.split())
        self.starts_line = starts_line


def _join(units: list[_Unit]) -> str:
    """Reúne frases preservando as quebras de linha originais."""
    out = []
    for i, unit in enumerate(units):
        if i:
            out.append("\n" if unit.starts_line else " ")
        out.append(unit.text)
    return "".join(out)


class SemanticChunker:
    """Chunker por frases com janela de contexto (sentence-window).

    - Unidade básica: frase; o texto é dividido primeiro por linha (o trafilatura separa
      parágrafos com "\\n", nunca "\\n\\n") e depois por frases.
    - Um chunk agrupa frases até `chunk_size` palavras. O tamanho é pequeno de propósito:
      o modelo denso é treinado com poucos tokens e a alegação a checar costuma ser uma frase.
    - `overlap_sentences` frases do fim de um chunk são repetidas no início do seguinte.
    - Cada chunk guarda `parent_text`: o trecho ao redor (~`parent_size` palavras) que deve ser
      entregue ao LLM. Assim a busca é precisa e o contexto da resposta é amplo.
    """

    def __init__(
        self,
        chunk_size: int = 120,
        overlap_sentences: int = 1,
        min_chunk_words: int = 30,
        parent_size: int = 300,
    ):
        if chunk_size < 1:
            raise ValueError("chunk_size deve ser positivo")
        self.chunk_size = chunk_size
        self.overlap_sentences = max(0, overlap_sentences)
        self.min_chunk_words = min_chunk_words
        self.parent_size = max(parent_size, chunk_size)

    # ------------------------------------------------------------------ unidades
    def _split_into_units(self, text: str) -> list[_Unit]:
        units: list[_Unit] = []
        for line in (ln.strip() for ln in text.split("\n")):
            if not line:
                continue
            for i, sentence in enumerate(_split_sentences(line)):
                for j, piece in enumerate(self._cap_words(sentence)):
                    units.append(_Unit(piece, starts_line=(i == 0 and j == 0)))
        return units

    def _cap_words(self, sentence: str) -> list[str]:
        """Frases maiores que o chunk são cortadas por palavras (sem overlap)."""
        words = sentence.split()
        if len(words) <= self.chunk_size:
            return [sentence]
        return [
            " ".join(words[i : i + self.chunk_size])
            for i in range(0, len(words), self.chunk_size)
        ]

    # ------------------------------------------------------------------ chunks
    def _pack(self, units: list[_Unit]) -> list[tuple[int, int]]:
        """Agrupa frases em intervalos [início, fim) de até `chunk_size` palavras."""
        spans: list[tuple[int, int]] = []
        start = 0
        while start < len(units):
            end, words = start, 0
            while end < len(units) and (end == start or words + units[end].words <= self.chunk_size):
                words += units[end].words
                end += 1
            spans.append((start, end))
            if end >= len(units):
                break
            # overlap: recua até `overlap_sentences` frases, garantindo avanço
            next_start = max(end - self.overlap_sentences, start + 1)
            while next_start < end and sum(u.words for u in units[next_start:end]) + units[end].words > self.chunk_size:
                next_start += 1
            start = next_start
        return self._merge_short_tail(units, spans)

    def _merge_short_tail(
        self, units: list[_Unit], spans: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """Evita um último chunk cujo conteúdo novo é minúsculo: funde com o anterior."""
        if len(spans) < 2:
            return spans
        (prev_start, prev_end), (last_start, last_end) = spans[-2], spans[-1]
        new_words = sum(u.words for u in units[max(last_start, prev_end) : last_end])
        merged_words = sum(u.words for u in units[prev_start:last_end])
        if new_words < self.min_chunk_words and merged_words <= int(self.chunk_size * 1.5):
            return spans[:-2] + [(prev_start, last_end)]
        return spans

    def _parent(self, units: list[_Unit], start: int, end: int) -> str:
        """Expande [start, end) para os lados até ~`parent_size` palavras."""
        words = sum(u.words for u in units[start:end])
        lo, hi = start, end
        while words < self.parent_size and (lo > 0 or hi < len(units)):
            if lo > 0 and (hi >= len(units) or (start - lo) <= (hi - end)):
                lo -= 1
                words += units[lo].words
            else:
                words += units[hi].words
                hi += 1
        return _join(units[lo:hi])

    @staticmethod
    def _clean_title(title: str, source: str) -> str:
        """Remove ruído do título no cabeçalho: prefixo "UnB Notícias - " e sufixo do site (" – dpg")."""
        cleaned = _TITLE_PREFIX.sub("", title)
        site = source.split(".")[0]
        for dash in ("–", "-"):
            suffix = f" {dash} {site}"
            if cleaned.endswith(suffix):
                cleaned = cleaned[: -len(suffix)]
        return cleaned.strip() or title

    def chunk_document(self, document: RawDocument) -> list[DocumentChunk]:
        if not document.content or not document.content.strip():
            return []

        units = self._split_into_units(document.content)
        if not units:
            return []

        spans = self._pack(units)
        total_chunks = len(spans)
        header_title = self._clean_title(document.title, document.source)
        chunks: list[DocumentChunk] = []

        for idx, (start, end) in enumerate(spans):
            raw_text = _join(units[start:end])
            enriched_content = (
                f"[Documento: {header_title}]\n"
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
                    parent_text=self._parent(units, start, end),
                    chunk_index=idx,
                    total_chunks=total_chunks,
                    title=document.title,
                    url=document.url,
                    source=document.source,
                    semester_ref=document.semester_ref,
                    published_at=document.published_at,
                )
            )

        return chunks
