"""Guardrails do fluxo de checagem. PRIMEIRA VERSÃO, para revisão: os limites abaixo são escolhas
iniciais, não calibradas. Cada guardrail acionado vira um código em `Checagem.guardrails`.

Entrada (antes do LLM)
  alegacao_invalida    alegação vazia, curta ou longa demais -> não chama o LLM
  injecao_suspeita     frases típicas de prompt injection na alegação -> só sinaliza (o texto já vai como dado)
  sem_evidencias       a busca não devolveu nada -> INCONCLUSIVO sem chamar o LLM
  evidencia_fraca      (desligado até calibrar) melhor nota do reranker abaixo do limiar -> BOATO_SEM_REGISTRO
Saída (depois do LLM)
  json_reparado        a 1ª resposta não era JSON válido e a 2ª foi
  json_invalido        nenhuma resposta válida -> INCONCLUSIVO
  llm_repetido         o provedor falhou de forma transitória (503/429/timeout) e a nova tentativa funcionou
  llm_indisponivel     erro do provedor (ou transitório que persistiu) -> INCONCLUSIVO
  fonte_inexistente    o LLM citou um número de evidência que não existe -> removido
  citacao_nao_verificada  citação que não aparece literalmente nas evidências citadas -> removida
  sem_fonte_ou_citacao_valida  CONFIRMADO/DESATUALIZADO sem fonte e citação verificáveis -> INCONCLUSIVO
  fonte_nao_oficial    CONFIRMADO só com fontes não oficiais (ex.: ADUnB) -> INCONCLUSIVO
  justificativa_vazia / justificativa_truncada
  confianca_limitada   confiança acima do teto, ou rebaixada junto com o veredito
"""

import json
import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator

from fato_unb.rag.models import VereditoType
from fato_unb.rag.retriever import Evidencia

VEREDITOS_QUE_EXIGEM_PROVA = {
    VereditoType.CONFIRMADO_OFICIALMENTE,
    VereditoType.DESATUALIZADO_OU_FORA_DE_CONTEXTO,
}

_PADROES_INJECAO = re.compile(
    r"ignor[ea]\w*\s+(todas?\s+)?(as\s+)?(instru|regras|ordens)"
    r"|ignore\s+(all\s+|the\s+)?(previous|above|prior)"
    r"|(system|developer)\s+prompt|prompt\s+do\s+sistema"
    r"|voc[êe]\s+agora\s+[ée]|you\s+are\s+now|act\s+as|finja\s+(ser|que)"
    r"|responda\s+(apenas\s+)?com\s+(confirmado|boato)|diga\s+que\s+[ée]\s+(verdade|confirmado)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GuardrailConfig:
    min_chars_alegacao: int = 8
    max_chars_alegacao: int = 500
    max_evidencias: int = 4
    max_chars_contexto: int = 1800  # por evidência enviada ao LLM (limita custo)
    max_tentativas_json: int = 2
    max_tentativas_llm: int = 3  # só para erros transitórios do provedor
    espera_inicial_s: float = 1.0  # cresce ×3 a cada tentativa: 1 s, 3 s
    max_chars_justificativa: int = 600
    min_chars_citacao: int = 12  # citações triviais ("de 2026") não provam nada
    max_confianca: float = 0.95
    confianca_rebaixada: float = 0.5
    dominios_oficiais: tuple[str, ...] = ("unb.br",)  # noticias.unb.br, dpg.unb.br... (não adunb.org)
    min_score_reranker: float | None = None  # desligado: falta calibrar com o reranker medido


class RespostaLLM(BaseModel):
    veredito: VereditoType
    justificativa: str = ""
    fontes: list[int] = Field(default_factory=list)
    citacoes: list[str] = Field(default_factory=list)
    confianca: float = 0.0

    @field_validator("fontes", mode="before")
    @classmethod
    def _aceita_colchetes(cls, v):
        if isinstance(v, list):
            return [int(re.sub(r"[^\d]", "", x)) if isinstance(x, str) else x for x in v]
        return v

    @field_validator("confianca", mode="before")
    @classmethod
    def _limita_confianca(cls, v):
        try:
            return min(max(float(v), 0.0), 1.0)
        except (TypeError, ValueError):
            return 0.0


@dataclass
class ResultadoSaida:
    veredito: VereditoType
    justificativa: str
    fontes: list[int]
    confianca: float
    acionados: list[str] = field(default_factory=list)


def validar_alegacao(alegacao: str, cfg: GuardrailConfig) -> str | None:
    """Devolve o motivo da rejeição, ou None se a alegação é aceitável."""
    texto = (alegacao or "").strip()
    if len(texto) < cfg.min_chars_alegacao:
        return f"A alegação é curta demais (mínimo {cfg.min_chars_alegacao} caracteres)."
    if len(texto) > cfg.max_chars_alegacao:
        return f"A alegação é longa demais (máximo {cfg.max_chars_alegacao} caracteres)."
    return None


def detectar_injecao(alegacao: str) -> bool:
    return bool(_PADROES_INJECAO.search(alegacao or ""))


def extrair_json(texto: str) -> dict:
    """Extrai o objeto JSON da resposta, tolerando cercas de markdown e texto ao redor."""
    limpo = re.sub(r"^```(?:json)?\s*|\s*```$", "", texto.strip(), flags=re.IGNORECASE)
    ini, fim = limpo.find("{"), limpo.rfind("}")
    if ini == -1 or fim <= ini:
        raise ValueError("resposta sem objeto JSON")
    return json.loads(limpo[ini : fim + 1])


def _norm(texto: str) -> str:
    return re.sub(r"\s+", " ", texto.replace("\xa0", " ")).strip().casefold()


def _eh_oficial(url: str, cfg: GuardrailConfig) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in cfg.dominios_oficiais)


def aplicar_guardrails_saida(
    resp: RespostaLLM,
    evidencias: list[Evidencia],
    textos_enviados: list[str],
    cfg: GuardrailConfig,
) -> ResultadoSaida:
    """Valida a resposta do LLM contra as evidências e rebaixa o veredito quando não se sustenta.

    `textos_enviados[i]` é o texto da evidência i+1 exatamente como foi ao LLM (já truncado)."""
    acionados: list[str] = []
    veredito = resp.veredito

    validas = [i for i in dict.fromkeys(resp.fontes) if 1 <= i <= len(evidencias)]
    if len(validas) != len(set(resp.fontes)):
        acionados.append("fonte_inexistente")

    base = _norm(" ".join(textos_enviados[i - 1] + " " + evidencias[i - 1].title for i in validas))
    verificadas = [
        c for c in resp.citacoes if len(c.strip()) >= cfg.min_chars_citacao and _norm(c) in base
    ]
    if len(verificadas) != len(resp.citacoes):
        acionados.append("citacao_nao_verificada")

    motivo_rebaixe: str | None = None
    if veredito in VEREDITOS_QUE_EXIGEM_PROVA and not (validas and verificadas):
        motivo_rebaixe = "sem_fonte_ou_citacao_valida"
    elif veredito == VereditoType.CONFIRMADO_OFICIALMENTE and not any(
        _eh_oficial(evidencias[i - 1].url, cfg) for i in validas
    ):
        motivo_rebaixe = "fonte_nao_oficial"

    justificativa = resp.justificativa.strip()
    if not justificativa:
        justificativa = "O modelo não apresentou justificativa."
        motivo_rebaixe = motivo_rebaixe or "justificativa_vazia"
    elif len(justificativa) > cfg.max_chars_justificativa:
        justificativa = justificativa[: cfg.max_chars_justificativa].rstrip() + "…"
        acionados.append("justificativa_truncada")

    confianca = resp.confianca
    if motivo_rebaixe:
        acionados.append(motivo_rebaixe)
        veredito = VereditoType.INCONCLUSIVO
        justificativa += f" (Veredito rebaixado automaticamente: {motivo_rebaixe}.)"
        confianca = min(confianca, cfg.confianca_rebaixada)
    if confianca > cfg.max_confianca:
        confianca = cfg.max_confianca
        acionados.append("confianca_limitada")

    return ResultadoSaida(veredito, justificativa, validas, confianca, acionados)
