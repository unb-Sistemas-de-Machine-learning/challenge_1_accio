import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

_ERRO_ONNX = "External data path"


def _cache_dir() -> Path:
    # mesmo padrão do fastembed: FASTEMBED_CACHE_PATH ou <tmp>/fastembed_cache
    bruto = os.getenv("FASTEMBED_CACHE_PATH")
    return Path(bruto).expanduser() if bruto else Path(tempfile.gettempdir()) / "fastembed_cache"


def materializar_symlinks(cache: Path | None = None) -> int:
    """Troca por cópias reais os symlinks dos snapshots do cache do Hugging Face.

    O onnxruntime recente recusa modelos com dados externos (model.onnx_data, caso do e5-large)
    quando o symlink do snapshot resolve para fora da pasta do model.onnx: "External data path
    escapes model directory". O cache do HF só tem symlinks para blobs/, então o e5-large nunca
    carrega a partir dele. Custa o tamanho do modelo em disco (os blobs ficam).
    """
    cache = cache or _cache_dir()
    trocados = 0
    for snapshot in cache.glob("models--*/snapshots/*"):
        for caminho in snapshot.rglob("*"):
            if not caminho.is_symlink():
                continue
            alvo = caminho.resolve()
            if not alvo.is_file():
                continue
            tmp = caminho.with_name(caminho.name + ".tmp")
            shutil.copyfile(alvo, tmp)
            os.replace(tmp, caminho)  # substitui o symlink em si, não o arquivo para onde ele aponta
            trocados += 1
    return trocados


def carregar_modelo(fabrica: Callable[[], T]) -> T:
    """Chama `fabrica()` (ex.: `lambda: TextEmbedding(...)`); se o onnxruntime recusar o caminho
    dos dados externos, materializa os symlinks do cache e tenta uma segunda vez."""
    try:
        return fabrica()
    except Exception as exc:
        if _ERRO_ONNX not in str(exc):
            raise
        trocados = materializar_symlinks()
        if not trocados:
            raise
        logger.warning(
            "onnxruntime recusou os symlinks do cache do modelo; %d arquivo(s) copiado(s) para "
            "arquivos reais. Tentando carregar de novo.",
            trocados,
        )
        return fabrica()
