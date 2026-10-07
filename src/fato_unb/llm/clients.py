import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Protocol

from dotenv import load_dotenv

load_dotenv()  # lê GEMINI_API_KEY etc. do .env, sem depender de outro módulo ter feito isso

DEFAULT_PROVIDER = "gemini"
# Padrões baratos (ver comparação de preços na conversa do projeto); troque com LLM_MODEL.
DEFAULT_MODELS = {
    "gemini": "gemini-3.1-flash-lite",  # o 2.5-flash-lite responde 404 para contas novas
    "anthropic": "claude-haiku-4-5",
    # Nome que a API lista em GET /models (DeepSeek-V4.1-Flash). "deepseek-chat" ainda é aceito, mas é um
    # apelido que a API resolve para este modelo e não aparece na listagem: pode ser aposentado.
    "deepseek": "deepseek-flash",
}
MAX_OUTPUT_TOKENS = 800  # o veredito é um JSON curto; limita custo e respostas divagantes
TIMEOUT_SECONDS = 30


class LLMError(RuntimeError):
    """Falha ao obter resposta do provedor (chave ausente, rede, limite, bloqueio).

    `transitorio=True` marca falhas que costumam passar sozinhas (503 "alta demanda", 429, timeout)
    e por isso valem uma nova tentativa; as demais (chave inválida, modelo inexistente) não."""

    def __init__(self, mensagem: str, transitorio: bool = False, retry_apos: float | None = None):
        super().__init__(mensagem)
        self.transitorio = transitorio
        self.retry_apos = retry_apos  # segundos que o próprio provedor pediu para esperar (429)


_CODIGOS_TRANSITORIOS = {408, 429, 500, 502, 503, 504}


def _retry_apos(exc: Exception) -> float | None:
    """Tempo de espera que o provedor informou na mensagem de erro ("Please retry in 8.8s", retryDelay)."""
    texto = str(exc)
    m = re.search(r"retry in ([\d.]+)\s*s", texto, re.IGNORECASE) or re.search(r"retryDelay'?\"?:\s*'?\"?([\d.]+)s", texto)
    return float(m.group(1)) if m else None


def _transitorio(exc: Exception) -> bool:
    codigo = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    texto = str(exc).lower()
    return codigo in _CODIGOS_TRANSITORIOS or "timed out" in texto or "timeout" in texto


@dataclass(frozen=True)
class LLMResposta:
    texto: str
    tokens_entrada: int | None = None
    tokens_saida: int | None = None
    modelo_real: str | None = None  # o que o provedor diz ter respondido (pode diferir de um apelido pedido)
    tokens_cache: int | None = None  # parte de tokens_entrada servida do cache do provedor
    fingerprint: str | None = None


class LLMClient(Protocol):
    nome: str

    def gerar(self, system: str, user: str) -> LLMResposta: ...


class GeminiClient:
    """Gemini via AI Studio (google-genai). A chave vem de GEMINI_API_KEY ou GOOGLE_API_KEY."""

    def __init__(self, model: str | None = None, api_key: str | None = None):
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise LLMError(
                "Chave do Gemini ausente: defina GEMINI_API_KEY (AI Studio: https://aistudio.google.com/apikey)."
            )
        from google import genai
        from google.genai import types

        self._types = types
        self.model = model or DEFAULT_MODELS["gemini"]
        self.nome = f"gemini:{self.model}"
        self._client = genai.Client(
            api_key=key, http_options=types.HttpOptions(timeout=TIMEOUT_SECONDS * 1000)
        )

    def gerar(self, system: str, user: str) -> LLMResposta:
        try:
            resp = self._client.models.generate_content(
                model=self.model,
                contents=user,
                config=self._types.GenerateContentConfig(
                    system_instruction=system,
                    response_mime_type="application/json",
                    temperature=0.0,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    automatic_function_calling=self._types.AutomaticFunctionCallingConfig(disable=True),
                ),
            )
        except Exception as exc:  # rede, cota, bloqueio: o chamador decide o que fazer
            raise LLMError(f"Gemini falhou: {exc}", transitorio=_transitorio(exc), retry_apos=_retry_apos(exc)) from exc
        usage = getattr(resp, "usage_metadata", None)
        texto = resp.text or ""
        if not texto:
            raise LLMError("Gemini devolveu resposta vazia (possível bloqueio de segurança).")
        return LLMResposta(
            texto=texto,
            tokens_entrada=getattr(usage, "prompt_token_count", None),
            tokens_saida=getattr(usage, "candidates_token_count", None),
        )


class AnthropicClient:
    """Claude via SDK oficial. A chave vem de ANTHROPIC_API_KEY (ou perfil `ant auth login`)."""

    def __init__(self, model: str | None = None, api_key: str | None = None):
        if not (api_key or os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
            raise LLMError("Chave da Anthropic ausente: defina ANTHROPIC_API_KEY.")
        import anthropic

        self._anthropic = anthropic
        self.model = model or DEFAULT_MODELS["anthropic"]
        self.nome = f"anthropic:{self.model}"
        self._client = anthropic.Anthropic(api_key=api_key, timeout=TIMEOUT_SECONDS)

    def gerar(self, system: str, user: str) -> LLMResposta:
        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=0.0,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except self._anthropic.APIError as exc:
            raise LLMError(f"Anthropic falhou: {exc}", transitorio=_transitorio(exc), retry_apos=_retry_apos(exc)) from exc
        if resp.stop_reason == "refusal":
            raise LLMError("A Anthropic recusou a requisição (stop_reason=refusal).")
        texto = "".join(b.text for b in resp.content if b.type == "text")
        if not texto:
            raise LLMError("Anthropic devolveu resposta sem texto.")
        return LLMResposta(
            texto=texto,
            tokens_entrada=resp.usage.input_tokens,
            tokens_saida=resp.usage.output_tokens,
        )


class DeepSeekClient:
    """DeepSeek via API compatível com a da OpenAI. A chave vem de DEEPSEEK_API_KEY.

    Usa `requests` (já é dependência) em vez do SDK da OpenAI, para não mexer no uv.lock."""

    URL = "https://api.deepseek.com/chat/completions"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self._key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self._key:
            raise LLMError("Chave do DeepSeek ausente: defina DEEPSEEK_API_KEY (https://platform.deepseek.com/api_keys).")
        self.model = model or DEFAULT_MODELS["deepseek"]
        self.nome = f"deepseek:{self.model}"

    def gerar(self, system: str, user: str) -> LLMResposta:
        import requests

        try:
            resp = requests.post(
                self.URL,
                headers={"Authorization": f"Bearer {self._key}"},
                json={
                    "model": self.model,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    "response_format": {"type": "json_object"},  # exige a palavra "JSON" no prompt (já tem)
                    "temperature": 0.0,
                    "max_tokens": MAX_OUTPUT_TOKENS,
                },
                timeout=TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            raise LLMError(f"DeepSeek falhou: {exc}", transitorio=_transitorio(exc)) from exc
        if resp.status_code == 402:
            raise LLMError("DeepSeek: saldo insuficiente (402). Recarregue em https://platform.deepseek.com.")
        if resp.status_code != 200:
            raise LLMError(
                f"DeepSeek falhou ({resp.status_code}): {resp.text[:200]}",
                transitorio=resp.status_code in _CODIGOS_TRANSITORIOS,
            )
        dados = resp.json()
        texto = (dados.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        if not texto:
            raise LLMError("DeepSeek devolveu resposta vazia.")
        usage = dados.get("usage") or {}
        return LLMResposta(
            texto=texto,
            tokens_entrada=usage.get("prompt_tokens"),
            tokens_saida=usage.get("completion_tokens"),
            modelo_real=dados.get("model"),
            tokens_cache=usage.get("prompt_cache_hit_tokens"),
            fingerprint=dados.get("system_fingerprint"),
        )


class ComLimiteDeTaxa:
    """Envolve um cliente e espaça as chamadas para não passar de `rpm` requisições por minuto.

    É seguro entre threads e vale também para as repetições (cada tentativa é uma requisição que conta
    na cota). Free tier do Gemini: 15 RPM por modelo; o limite de 429 se repete se isso não for respeitado."""

    def __init__(self, cliente, rpm: float, relogio=time.monotonic, dormir=time.sleep):
        if rpm <= 0:
            raise ValueError("rpm deve ser positivo")
        self.cliente = cliente
        self.nome = cliente.nome
        self._intervalo = 60.0 / rpm
        self._relogio, self._dormir = relogio, dormir
        self._lock = threading.Lock()
        self._proximo = 0.0

    def gerar(self, system: str, user: str) -> LLMResposta:
        with self._lock:  # reserva o próximo horário livre; dorme fora do lock
            agora = self._relogio()
            inicio = max(agora, self._proximo)
            self._proximo = inicio + self._intervalo
        if inicio > agora:
            self._dormir(inicio - agora)
        return self.cliente.gerar(system, user)


def criar_cliente(
    provider: str | None = None, model: str | None = None, rpm: float | None = None
) -> LLMClient:
    """Cria o cliente do provedor escolhido (LLM_PROVIDER / LLM_MODEL no ambiente).

    `rpm` (ou LLM_RPM) limita as requisições por minuto, útil no free tier (Gemini: 15 RPM por modelo)."""
    provider = (provider or os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER).lower()
    model = model or os.getenv("LLM_MODEL") or None
    rpm = rpm or float(os.getenv("LLM_RPM") or 0) or None
    if provider == "gemini":
        cliente: LLMClient = GeminiClient(model=model)
    elif provider == "anthropic":
        cliente = AnthropicClient(model=model)
    elif provider == "deepseek":
        cliente = DeepSeekClient(model=model)
    else:
        raise LLMError(f"Provedor '{provider}' não suportado. Use 'gemini', 'anthropic' ou 'deepseek'.")
    return ComLimiteDeTaxa(cliente, rpm) if rpm else cliente


def cronometrar(fn, *args):
    t0 = time.perf_counter()
    resultado = fn(*args)
    return resultado, (time.perf_counter() - t0) * 1000
