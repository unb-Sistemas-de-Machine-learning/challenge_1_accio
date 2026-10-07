"""Observabilidade das chamadas ao LLM com Langfuse (custo, latência, tokens, erros).

Sem LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY (ou com LANGFUSE_TRACING_ENABLED=false) tudo aqui vira no-op, e
uma falha do Langfuse nunca derruba quem chamou: o bot continua respondendo.

Privacidade: o texto das afirmações é dado de usuário. Por padrão NÃO é enviado; vai só um resumo
("<N caracteres, sha256 ...>"). Com LANGFUSE_CAPTURE_CONTENT=true o texto vai, mas depois do `scrub`
(CPF, telefone, e-mail, matrícula); use isso em desenvolvimento e na avaliação, não em produção.
"""

import hashlib
import logging
import os
import threading
from contextlib import contextmanager
from typing import Any

from dotenv import load_dotenv

from fato_unb.bots.privacy import scrub

load_dotenv()

logger = logging.getLogger(__name__)

_lf = None
_tentou = False
_lock = threading.Lock()


def _ligado(nome: str, padrao: bool) -> bool:
    valor = os.getenv(nome)
    if valor is None:
        return padrao
    return valor.strip().lower() in ("1", "true", "yes", "sim")


def ativo() -> bool:
    """Há chaves e o tracing não foi desligado?"""
    return (
        _ligado("LANGFUSE_TRACING_ENABLED", True)
        and bool(os.getenv("LANGFUSE_PUBLIC_KEY"))
        and bool(os.getenv("LANGFUSE_SECRET_KEY"))
    )


def capturar_conteudo() -> bool:
    return _ligado("LANGFUSE_CAPTURE_CONTENT", False)


def conteudo(texto: str | None) -> str | None:
    """O que pode ser enviado do texto: ele mesmo (já sem dados pessoais) ou só um resumo."""
    if texto is None:
        return None
    limpo = scrub(texto)
    if capturar_conteudo():
        return limpo
    return f"<{len(limpo)} caracteres, sha256 {hashlib.sha256(limpo.encode()).hexdigest()[:10]}>"


def _mascarar(*, data: Any, **_: Any) -> Any:
    """Rede de segurança do SDK: passa o `scrub` por qualquer texto que escape para o Langfuse."""
    if isinstance(data, str):
        return scrub(data)
    if isinstance(data, list):
        return [_mascarar(data=item) for item in data]
    if isinstance(data, dict):
        return {k: _mascarar(data=v) for k, v in data.items()}
    return data


def cliente():
    """O cliente do Langfuse, criado uma vez; None se desativado ou se a criação falhar."""
    global _lf, _tentou
    if _lf is not None or _tentou:
        return _lf
    with _lock:
        if _lf is None and not _tentou:
            _tentou = True
            if ativo():
                try:
                    from langfuse import Langfuse

                    _lf = Langfuse(mask=_mascarar)
                    logger.info("Langfuse ativo: enviando traces de custo e latência.")
                except Exception:
                    logger.exception("Não consegui iniciar o Langfuse; seguindo sem observabilidade.")
            else:
                logger.info("Langfuse desativado (sem chaves ou LANGFUSE_TRACING_ENABLED=false).")
    return _lf


class _Observacao:
    """Envolve a observação do Langfuse para que um erro ao atualizá-la nunca chegue ao chamador."""

    def __init__(self, obs=None):
        self._obs = obs

    def update(self, **campos: Any) -> None:
        if self._obs is None:
            return
        try:
            self._obs.update(**campos)
        except Exception:
            logger.exception("Falha ao atualizar a observação do Langfuse.")


@contextmanager
def observacao(nome: str, *, tipo: str = "span", **campos: Any):
    """Abre uma observação (span, generation, retriever...) aninhada na atual. No-op se desativado."""
    lf = cliente()
    gerenciador = obs = None
    if lf is not None:
        try:
            gerenciador = lf.start_as_current_observation(name=nome, as_type=tipo, **campos)
            obs = gerenciador.__enter__()
        except Exception:
            logger.exception("Falha ao abrir a observação '%s' no Langfuse.", nome)
            gerenciador = None
    if gerenciador is None:
        yield _Observacao()
        return
    try:
        yield _Observacao(obs)
    except BaseException as exc:
        try:
            gerenciador.__exit__(type(exc), exc, exc.__traceback__)
        except Exception:
            logger.exception("Falha ao fechar a observação '%s' no Langfuse.", nome)
        raise
    try:
        gerenciador.__exit__(None, None, None)
    except Exception:
        logger.exception("Falha ao fechar a observação '%s' no Langfuse.", nome)


def flush() -> None:
    """Envia o que ainda está na fila (chamar antes de encerrar um processo de vida curta)."""
    if _lf is not None:
        try:
            _lf.flush()
        except Exception:
            logger.exception("Falha no flush do Langfuse.")


def encerrar() -> None:
    if _lf is not None:
        try:
            _lf.shutdown()
        except Exception:
            logger.exception("Falha ao encerrar o Langfuse.")
