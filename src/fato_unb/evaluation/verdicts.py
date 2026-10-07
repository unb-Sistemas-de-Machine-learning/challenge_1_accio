"""Avaliação dos vereditos: roda o dataset pelo fluxo real (busca -> LLM -> guardrails) e mede o acerto."""

import json
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

from fato_unb.evaluation.dataset import (
    CasoTeste,
    contains_span,
    content_key,
    expected_content_keys,
    normalize_url,
    url_to_content_keys,
)
from fato_unb.ingestion.models import RawDocument
from fato_unb.llm.checker import FactChecker
from fato_unb.llm.prompts import neutralizar
from fato_unb.rag.models import VereditoType
from fato_unb.rag.retriever import Evidencia

META_ACERTO = 0.5  # meta do projeto (evaluation/README.md): acerto > 50%
ORDEM = [v.value for v in VereditoType]
CONFIRMADO = VereditoType.CONFIRMADO_OFICIALMENTE.value
INCONCLUSIVO = VereditoType.INCONCLUSIVO.value
FALHA_INFRA = "llm_indisponivel"  # o provedor falhou: o resultado não diz nada sobre o modelo

# US$ por 1M de tokens (entrada, saída); conferido em 2026-09 nas páginas de preço dos provedores.
PRECOS = {
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.6-flash": (0.75, 3.75),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    # DeepSeek-V4.1-Flash, conferido em 2026-10 na página oficial de preços. Valor de HORÁRIO DE PICO (teto):
    # fora do pico (seg-sex fora de 01-04h e 06-10h UTC, e fins de semana) cai pela metade (0,15 / 0,60). O token
    # de entrada em cache custa 0,006 (pico), mas esta tabela cobra tudo como "cache miss", então é um teto.
    "deepseek-flash": (0.30, 1.20),
    "deepseek-chat": (0.30, 1.20),  # apelido que a API resolve para o deepseek-flash
}


@dataclass
class ResultadoVeredito:
    id: str
    tipo: str
    desafio: str
    alegacao: str
    esperado: str | None
    obtido: str
    guardrails: list[str] = field(default_factory=list)
    evidencia_certa_no_contexto: bool | None = None  # o DOCUMENTO esperado veio entre as evidências
    evidencia_no_texto_enviado: bool | None = None  # a FRASE esperada (evidence_spans) estava no texto enviado ao LLM
    tokens_entrada: int | None = None
    tokens_saida: int | None = None
    latencia_ms: float | None = None
    justificativa: str = ""
    fontes: list[str] = field(default_factory=list)
    modelo: str | None = None


class RetrieverCacheado:
    """Devolve evidências já buscadas. Permite buscar em sequência (modelo de embedding local) e
    paralelizar só as chamadas ao LLM, sem alterar o fluxo real do FactChecker."""

    def __init__(self, evidencias_por_alegacao: dict[str, list[Evidencia]]):
        self._cache = evidencias_por_alegacao

    def buscar(self, alegacao: str, limit: int = 5, **_) -> list[Evidencia]:
        return self._cache[alegacao][:limit]


def urls_esperadas(caso: CasoTeste, docs: list[RawDocument]) -> set[str]:
    """URLs do documento esperado e de suas cópias (mesmo conteúdo, URL diferente)."""
    if not caso.expected_urls:
        return set()
    chaves = expected_content_keys(caso, url_to_content_keys(docs))
    return {normalize_url(str(d.url)) for d in docs if content_key(d) in chaves}


def frase_no_texto_enviado(caso: CasoTeste, evidencias: list[Evidencia], max_chars: int) -> bool | None:
    """A frase-evidência estava no texto que o LLM recebeu (contexto neutralizado e truncado)?

    É mais exigente que "o documento veio": a busca pode trazer a notícia certa e um trecho errado."""
    if not caso.evidence_spans:
        return None
    return any(contains_span(neutralizar(e.contexto.strip())[:max_chars], caso.evidence_spans) for e in evidencias)


def avaliar_caso(
    caso: CasoTeste, checker: FactChecker, esperadas: set[str], modelo: str | None
) -> ResultadoVeredito:
    c = checker.verificar(caso.alegacao)
    v = c.veredito
    chegou = None
    if esperadas:
        chegou = any(normalize_url(ev.url) in esperadas for ev in c.evidencias)
    return ResultadoVeredito(
        id=caso.id,
        tipo=caso.tipo.value,
        desafio=caso.desafio,
        alegacao=caso.alegacao,
        esperado=caso.veredito_esperado.value if caso.veredito_esperado else None,
        obtido=v.veredito.value,
        guardrails=list(c.guardrails),
        evidencia_certa_no_contexto=chegou,
        evidencia_no_texto_enviado=frase_no_texto_enviado(caso, c.evidencias, checker.cfg.max_chars_contexto),
        tokens_entrada=c.tokens_entrada,
        tokens_saida=c.tokens_saida,
        latencia_ms=c.latencia_llm_ms,
        justificativa=v.justificativa,
        fontes=[str(f.url) for f in v.fontes],
        modelo=c.modelo or modelo,
    )


def executar(
    casos: list[CasoTeste],
    docs: list[RawDocument],
    retriever,
    llm,
    *,
    hoje: date,
    paralelismo: int = 3,
    saida: Path | None = None,
    incluir_perguntas: bool = False,
    config=None,
    progresso=print,
) -> list[ResultadoVeredito]:
    """Roda os casos e devolve os resultados. Com `saida`, grava cada resultado em JSONL e, se o
    arquivo já existir, retoma de onde parou (pula os ids já feitos)."""
    selecionados = [c for c in casos if incluir_perguntas or c.veredito_esperado is not None]
    feitos: dict[str, ResultadoVeredito] = {}
    if saida and saida.exists():
        for linha in saida.read_text(encoding="utf-8").splitlines():
            if linha.strip():
                r = ResultadoVeredito(**json.loads(linha))
                feitos[r.id] = r  # a última linha de cada id vale
        # falha de infraestrutura (provedor fora do ar, cota) não é resultado: esses casos são refeitos
        falhos = [i for i, r in feitos.items() if FALHA_INFRA in r.guardrails]
        for i in falhos:
            del feitos[i]
        if feitos or falhos:
            progresso(f"Retomando: {len(feitos)} casos prontos em {saida}; refazendo {len(falhos)} que falharam por infraestrutura")
    pendentes = [c for c in selecionados if c.id not in feitos]

    # 1) busca em sequência (embedding local); 2) LLM em paralelo
    limite = (config.max_evidencias if config else 4)
    cache = {c.alegacao: retriever.buscar(c.alegacao, limit=limite) for c in selecionados}
    checker = FactChecker(RetrieverCacheado(cache), llm, config, hoje=hoje)
    max_chars = checker.cfg.max_chars_contexto
    for c in selecionados:  # resultados de execuções antigas podem não ter a métrica por frase: recalcula (sem LLM)
        r = feitos.get(c.id)
        if r is not None and r.evidencia_no_texto_enviado is None:
            r.evidencia_no_texto_enviado = frase_no_texto_enviado(c, cache[c.alegacao], max_chars)
    esperadas = {c.id: urls_esperadas(c, docs) for c in pendentes}

    resultados = list(feitos.values())
    with ThreadPoolExecutor(max_workers=max(1, paralelismo)) as pool:
        futuros = [pool.submit(avaliar_caso, c, checker, esperadas[c.id], getattr(llm, "nome", None)) for c in pendentes]
        for i, fut in enumerate(futuros, 1):
            r = fut.result()
            resultados.append(r)
            if saida:
                with saida.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
            progresso(f"[{i}/{len(pendentes)}] {r.id} esperado={r.esperado} obtido={r.obtido}")
    ordem = {c.id: i for i, c in enumerate(casos)}
    return sorted(resultados, key=lambda r: ordem.get(r.id, 0))


# ------------------------------------------------------------------ métricas
def _acerto(rs: list[ResultadoVeredito]) -> float | None:
    return sum(r.obtido == r.esperado for r in rs) / len(rs) if rs else None


def calcular_metricas(resultados: list[ResultadoVeredito]) -> dict:
    rotulados = [r for r in resultados if r.esperado is not None]
    falhas_infra = [r for r in rotulados if FALHA_INFRA in r.guardrails]
    validos = [r for r in rotulados if FALHA_INFRA not in r.guardrails]
    respondidos = [r for r in rotulados if r.obtido != INCONCLUSIVO]
    matriz = {e: {o: 0 for o in ORDEM} for e in ORDEM}
    for r in rotulados:
        matriz[r.esperado][r.obtido] += 1

    indevidas = [r for r in rotulados if r.obtido == CONFIRMADO and r.esperado != CONFIRMADO]
    perdidas = [r for r in rotulados if r.esperado == CONFIRMADO and r.obtido != CONFIRMADO]
    com_ev = [r for r in rotulados if r.evidencia_certa_no_contexto is True]
    sem_ev = [r for r in rotulados if r.evidencia_certa_no_contexto is False]
    com_frase = [r for r in rotulados if r.evidencia_no_texto_enviado is True]
    sem_frase = [r for r in rotulados if r.evidencia_no_texto_enviado is False]

    def agrupar(chave):
        grupos = defaultdict(list)
        for r in rotulados:
            grupos[getattr(r, chave)].append(r)
        return {k: {"n": len(v), "acerto": _acerto(v)} for k, v in sorted(grupos.items())}

    tokens_in = sum(r.tokens_entrada or 0 for r in resultados)
    tokens_out = sum(r.tokens_saida or 0 for r in resultados)
    modelo = next((r.modelo for r in resultados if r.modelo), None)
    preco = PRECOS.get((modelo or "").split(":")[-1])
    custo = (tokens_in * preco[0] + tokens_out * preco[1]) / 1e6 if preco else None
    lat = sorted(r.latencia_ms for r in resultados if r.latencia_ms)

    return {
        "modelo": modelo,
        "n_avaliados": len(rotulados),
        "acerto": _acerto(rotulados),
        "meta": META_ACERTO,
        "atinge_meta": (_acerto(rotulados) or 0) > META_ACERTO,
        "falhas_infra": len(falhas_infra),
        "acerto_sem_falhas_infra": _acerto(validos),
        "n_sem_falhas_infra": len(validos),
        "inconclusivos": len(rotulados) - len(respondidos),
        "taxa_inconclusivo": 1 - len(respondidos) / len(rotulados) if rotulados else None,
        "acerto_entre_respondidos": _acerto(respondidos),
        "confirmacoes_indevidas": [r.id for r in indevidas],
        "confirmacoes_perdidas": [r.id for r in perdidas],
        "matriz": matriz,
        "por_tipo": agrupar("tipo"),
        "por_desafio": agrupar("desafio"),
        "acerto_com_evidencia_certa": {"n": len(com_ev), "acerto": _acerto(com_ev)},
        "acerto_sem_evidencia_certa": {"n": len(sem_ev), "acerto": _acerto(sem_ev)},
        "acerto_com_frase_enviada": {"n": len(com_frase), "acerto": _acerto(com_frase)},
        "acerto_sem_frase_enviada": {"n": len(sem_frase), "acerto": _acerto(sem_frase), "ids": [r.id for r in sem_frase if r.obtido != r.esperado]},
        "guardrails": dict(Counter(g for r in resultados for g in r.guardrails).most_common()),
        "tokens_entrada": tokens_in,
        "tokens_saida": tokens_out,
        "custo_usd": custo,
        "latencia_media_ms": statistics.mean(lat) if lat else None,
        "latencia_p95_ms": lat[int(0.95 * (len(lat) - 1))] if lat else None,
    }


def _pct(x: float | None) -> str:
    return "n/d" if x is None else f"{100 * x:.1f}%"


def formatar_relatorio(m: dict, resultados: list[ResultadoVeredito], falhas: bool = False) -> str:
    n = m["n_avaliados"]
    acertos = round((m["acerto"] or 0) * n)
    linhas = [f"AVALIAÇÃO DE VEREDITOS — {m['modelo']}"]
    if m["falhas_infra"]:
        linhas += [
            f"  *** ATENÇÃO: {m['falhas_infra']} de {n} casos ({_pct(m['falhas_infra'] / n)}) falharam por INDISPONIBILIDADE DO PROVEDOR",
            "  *** (cota/503/timeout) e contam como INCONCLUSIVO. O acerto geral abaixo mede a infraestrutura, não o modelo.",
            f"  *** Sem essas falhas: acerto {_pct(m['acerto_sem_falhas_infra'])} em {m['n_sem_falhas_infra']} casos. Rode de novo para refazê-las.",
            "",
        ]
    linhas += [
        f"  casos avaliados: {n}",
        f"  ACERTO GERAL   : {_pct(m['acerto'])} ({acertos}/{n})  | meta do projeto > {_pct(m['meta'])}: "
        + ("ATINGIDA" if m["atinge_meta"] else "NÃO atingida"),
        f"  INCONCLUSIVO   : {m['inconclusivos']} ({_pct(m['taxa_inconclusivo'])}); acerto entre os respondidos: {_pct(m['acerto_entre_respondidos'])}",
        f"  confirmações INDEVIDAS (erro mais grave): {len(m['confirmacoes_indevidas'])} {m['confirmacoes_indevidas']}",
        f"  confirmações perdidas: {len(m['confirmacoes_perdidas'])} {m['confirmacoes_perdidas']}",
        "",
        "  O DOCUMENTO certo veio entre as evidências?",
        f"    sim: n={m['acerto_com_evidencia_certa']['n']}, acerto {_pct(m['acerto_com_evidencia_certa']['acerto'])}",
        f"    não: n={m['acerto_sem_evidencia_certa']['n']}, acerto {_pct(m['acerto_sem_evidencia_certa']['acerto'])}",
        "  A FRASE com a evidência estava no texto enviado ao modelo? (mais exigente; explica erros de busca)",
        f"    sim: n={m['acerto_com_frase_enviada']['n']}, acerto {_pct(m['acerto_com_frase_enviada']['acerto'])}",
        f"    não: n={m['acerto_sem_frase_enviada']['n']}, acerto {_pct(m['acerto_sem_frase_enviada']['acerto'])}  erros nesse grupo: {m['acerto_sem_frase_enviada']['ids']}",
        "",
        "  Matriz de confusão (linha = esperado, coluna = obtido):",
    ]
    curtos = {v: v.split("_")[0][:6] for v in ORDEM}
    linhas.append("    " + " " * 14 + "".join(f"{curtos[o]:>8}" for o in ORDEM))
    for e in ORDEM:
        linhas.append(f"    {curtos[e]:<14}" + "".join(f"{m['matriz'][e][o]:>8}" for o in ORDEM))
    linhas += ["", "  Por tipo:"]
    linhas += [f"    {k:<14} n={v['n']:<3} acerto {_pct(v['acerto'])}" for k, v in m["por_tipo"].items()]
    linhas += ["  Por desafio:"]
    linhas += [f"    {k:<14} n={v['n']:<3} acerto {_pct(v['acerto'])}" for k, v in m["por_desafio"].items()]
    linhas += ["", f"  Guardrails acionados: {m['guardrails'] or 'nenhum'}"]
    custo = f"US$ {m['custo_usd']:.4f}" if m["custo_usd"] is not None else "n/d"
    linhas.append(
        f"  Tokens: {m['tokens_entrada']} entrada / {m['tokens_saida']} saída | custo {custo}"
        + (f" | latência média {m['latencia_media_ms']:.0f} ms, p95 {m['latencia_p95_ms']:.0f} ms" if m["latencia_media_ms"] else "")
    )
    if falhas:
        linhas += ["", "  ERROS:"]
        for r in resultados:
            if r.esperado is not None and r.obtido != r.esperado:
                ev = {True: "evidência ok", False: "SEM evidência", None: "-"}[r.evidencia_certa_no_contexto]
                linhas.append(f"    {r.id} [{r.tipo}/{r.desafio}] esperado={r.esperado} obtido={r.obtido} ({ev}) {r.guardrails or ''}")
                linhas.append(f"       {r.alegacao}")
                linhas.append(f"       -> {r.justificativa[:200]}")
    return "\n".join(linhas)
