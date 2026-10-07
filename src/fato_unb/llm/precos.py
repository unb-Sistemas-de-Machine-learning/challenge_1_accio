"""Preços dos provedores de LLM e cálculo de custo (usado pela avaliação offline e pela observabilidade)."""

# US$ por 1M de tokens (entrada, saída); conferido em 2026-09 nas páginas de preço dos provedores.
PRECOS = {
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.6-flash": (0.75, 3.75),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    # DeepSeek-V4.1-Flash, conferido em 2026-10 na página oficial de preços. Valor de HORÁRIO DE PICO (teto):
    # fora do pico (seg-sex fora de 01-04h e 06-10h UTC, e fins de semana) cai pela metade (0,15 / 0,60).
    "deepseek-flash": (0.30, 1.20),
    "deepseek-chat": (0.30, 1.20),  # apelido que a API resolve para o deepseek-flash
}

# US$ por 1M de tokens de ENTRADA que o provedor serviu do cache (pico). Sem entrada aqui, o cache não tem desconto.
PRECOS_CACHE = {
    "deepseek-flash": 0.006,
    "deepseek-chat": 0.006,
}


def nome_curto(modelo: str | None) -> str:
    """'deepseek:deepseek-flash' -> 'deepseek-flash'."""
    return (modelo or "").split(":")[-1]


def custo_usd(modelo: str | None, tokens_entrada: int, tokens_saida: int, tokens_cache: int = 0) -> dict[str, float] | None:
    """Custo em US$ separado em entrada/saída/total, ou None se o modelo não está na tabela.

    `tokens_entrada` inclui os tokens servidos do cache (como no `prompt_tokens` do DeepSeek); eles são
    cobrados à parte, pelo preço de cache, quando ele existe."""
    chave = nome_curto(modelo)
    preco = PRECOS.get(chave)
    if preco is None:
        return None
    preco_entrada, preco_saida = preco
    if chave in PRECOS_CACHE:
        cache = min(max(tokens_cache, 0), tokens_entrada)
        entrada = ((tokens_entrada - cache) * preco_entrada + cache * PRECOS_CACHE[chave]) / 1e6
    else:
        entrada = tokens_entrada * preco_entrada / 1e6
    saida = tokens_saida * preco_saida / 1e6
    return {"input": entrada, "output": saida, "total": entrada + saida}
