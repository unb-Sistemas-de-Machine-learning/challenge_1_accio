import logging
import time
from dataclasses import dataclass, field
from datetime import date

from pydantic import ValidationError

from fato_unb.llm.clients import LLMClient, LLMError, cronometrar
from fato_unb.llm.guardrails import (
    GuardrailConfig,
    RespostaLLM,
    aplicar_guardrails_saida,
    detectar_injecao,
    extrair_json,
    validar_alegacao,
)
from fato_unb.llm.prompts import SYSTEM_PROMPT, montar_prompt_usuario, neutralizar
from fato_unb.rag.models import FonteCitada, VereditoJSON, VereditoType
from fato_unb.rag.retriever import Evidencia, Retriever

logger = logging.getLogger(__name__)

MAX_ESPERA_S = 65.0  # teto da espera entre repetições (cota por minuto zera em até 60 s)


@dataclass
class Checagem:
    """Resultado de uma verificação: o veredito e o que aconteceu no caminho."""

    veredito: VereditoJSON
    guardrails: list[str] = field(default_factory=list)  # códigos acionados (ver guardrails.py)
    evidencias: list[Evidencia] = field(default_factory=list)
    usou_llm: bool = False
    modelo: str | None = None
    tokens_entrada: int | None = None
    tokens_saida: int | None = None
    latencia_llm_ms: float | None = None
    prompt_usuario: str | None = None  # para auditoria e revisão
    respostas_brutas: list[str] = field(default_factory=list)  # texto exato do modelo, antes dos guardrails


class FactChecker:
    """alegação -> busca de evidências -> LLM -> guardrails -> VereditoJSON."""

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMClient | None,
        config: GuardrailConfig | None = None,
        hoje: date | None = None,
        dormir=time.sleep,  # injetável para os testes não esperarem de verdade
    ):
        self.retriever = retriever
        self.llm = llm
        self.cfg = config or GuardrailConfig()
        self.hoje = hoje
        self._dormir = dormir

    # --------------------------------------------------------------- resultados curtos
    def _sem_llm(
        self,
        alegacao: str,
        veredito: VereditoType,
        justificativa: str,
        codigo: str,
        confianca: float = 0.0,
        evidencias: list[Evidencia] | None = None,
        extras: list[str] | None = None,
    ) -> Checagem:
        return Checagem(
            veredito=VereditoJSON(
                veredito=veredito,
                justificativa=justificativa,
                fontes=[],
                confianca=confianca,
                afirmacao_analisada=alegacao.strip()[: self.cfg.max_chars_alegacao],
            ),
            guardrails=[*(extras or []), codigo],
            evidencias=evidencias or [],
        )

    # --------------------------------------------------------------- fluxo principal
    def verificar(self, alegacao: str) -> Checagem:
        motivo = validar_alegacao(alegacao, self.cfg)
        if motivo:
            return self._sem_llm(alegacao, VereditoType.INCONCLUSIVO, motivo, "alegacao_invalida")

        extras = ["injecao_suspeita"] if detectar_injecao(alegacao) else []

        evidencias = self.retriever.buscar(alegacao, limit=self.cfg.max_evidencias)
        if not evidencias:
            return self._sem_llm(
                alegacao,
                VereditoType.INCONCLUSIVO,
                "Não encontrei nenhum documento na base para verificar esta alegação.",
                "sem_evidencias",
                extras=extras,
            )

        melhor = evidencias[0]
        if (
            self.cfg.min_score_reranker is not None
            and melhor.reranked
            and melhor.score < self.cfg.min_score_reranker
        ):
            return self._sem_llm(
                alegacao,
                VereditoType.BOATO_SEM_REGISTRO,
                "Nenhum documento da base trata claramente deste assunto.",
                "evidencia_fraca",
                confianca=0.4,
                evidencias=evidencias,
                extras=extras,
            )

        hoje = self.hoje or date.today()
        prompt = montar_prompt_usuario(alegacao, evidencias, hoje, self.cfg.max_chars_contexto)
        textos = [self._texto_enviado(ev) for ev in evidencias]

        if self.llm is None:  # modo de inspeção (--dry-run): devolve o prompt sem chamar o modelo
            checagem = self._sem_llm(
                alegacao, VereditoType.INCONCLUSIVO, "Modo de inspeção: o LLM não foi chamado.",
                "dry_run", evidencias=evidencias, extras=extras,
            )
            checagem.prompt_usuario = prompt
            return checagem

        resposta, tokens_in, tokens_out, latencia, erro, reparos, brutas = self._chamar_llm(prompt)
        base = dict(
            evidencias=evidencias, usou_llm=True, modelo=self.llm.nome,
            tokens_entrada=tokens_in, tokens_saida=tokens_out,
            latencia_llm_ms=latencia, prompt_usuario=prompt,
            respostas_brutas=brutas,
        )
        if resposta is None:
            checagem = self._sem_llm(
                alegacao, VereditoType.INCONCLUSIVO,
                "Não consegui obter uma análise confiável do modelo. Tente novamente.",
                erro, evidencias=evidencias, extras=extras,
            )
            for k, v in base.items():
                setattr(checagem, k, v)
            return checagem

        saida = aplicar_guardrails_saida(resposta, evidencias, textos, self.cfg)
        fontes = [
            FonteCitada(title=evidencias[i - 1].title, url=evidencias[i - 1].url, source=evidencias[i - 1].source)
            for i in saida.fontes
        ]
        return Checagem(
            veredito=VereditoJSON(
                veredito=saida.veredito,
                justificativa=saida.justificativa,
                fontes=fontes,
                confianca=saida.confianca,
                afirmacao_analisada=alegacao.strip(),
            ),
            guardrails=[*extras, *reparos, *saida.acionados],
            **base,
        )

    # --------------------------------------------------------------- auxiliares
    def _texto_enviado(self, ev: Evidencia) -> str:
        """O texto da evidência exatamente como vai ao LLM (neutralizado e truncado)."""
        texto = neutralizar(ev.contexto.strip())
        return texto[: self.cfg.max_chars_contexto]

    def _gerar_com_repeticao(self, prompt: str, reparos: list[str]):
        """Uma chamada ao LLM, repetida (com espera crescente) só se o erro for transitório."""
        espera = self.cfg.espera_inicial_s
        for tentativa in range(1, self.cfg.max_tentativas_llm + 1):
            try:
                resultado = cronometrar(self.llm.gerar, SYSTEM_PROMPT, prompt)
            except LLMError as exc:
                if not exc.transitorio or tentativa == self.cfg.max_tentativas_llm:
                    raise
                # se o provedor disse quanto esperar (429), obedece, com teto para não travar o chamador
                pausa = min(max(espera, (exc.retry_apos or 0) + 0.5), MAX_ESPERA_S)
                logger.warning(f"LLM falhou (tentativa {tentativa}), repetindo em {pausa:.0f}s: {exc}")
                self._dormir(pausa)
                espera *= 3
                if "llm_repetido" not in reparos:
                    reparos.append("llm_repetido")
                continue
            return resultado

    def _chamar_llm(self, prompt: str):
        """Devolve (resposta|None, tokens_in, tokens_out, ms, erro, reparos, brutas).

        Repete a chamada em erro transitório do provedor e, separadamente, em JSON inválido.
        Todo o estado é local a esta chamada: o FactChecker é usado por várias threads/usuários ao mesmo tempo."""
        reparos: list[str] = []
        brutas: list[str] = []
        tokens_in = tokens_out = 0
        latencia = 0.0
        for tentativa in range(1, self.cfg.max_tentativas_json + 1):
            try:
                resp, ms = self._gerar_com_repeticao(prompt, reparos)
            except LLMError as exc:
                logger.warning(f"LLM indisponível: {exc}")
                return (None, tokens_in or None, tokens_out or None, latencia or None,
                        "llm_indisponivel", reparos, brutas)
            latencia += ms
            brutas.append(resp.texto)
            tokens_in += resp.tokens_entrada or 0
            tokens_out += resp.tokens_saida or 0
            try:
                parsed = RespostaLLM.model_validate(extrair_json(resp.texto))
            except (ValueError, ValidationError) as exc:
                logger.warning(f"Resposta inválida na tentativa {tentativa}: {exc}")
                continue
            if tentativa > 1:
                reparos.append("json_reparado")
            return parsed, tokens_in or None, tokens_out or None, latencia, "json_invalido", reparos, brutas
        return (None, tokens_in or None, tokens_out or None, latencia, "json_invalido", reparos, brutas)
