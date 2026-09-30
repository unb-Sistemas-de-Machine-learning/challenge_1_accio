import os
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
}
MAX_OUTPUT_TOKENS = 800  # o veredito é um JSON curto; limita custo e respostas divagantes
TIMEOUT_SECONDS = 30


class LLMError(RuntimeError):
    """Falha ao obter resposta do provedor (chave ausente, rede, limite, bloqueio).

    `transitorio=True` marca falhas que costumam passar sozinhas (503 "alta demanda", 429, timeout)
    e por isso valem uma nova tentativa; as demais (chave inválida, modelo inexistente) não."""

    def __init__(self, mensagem: str, transitorio: bool = False):
        super().__init__(mensagem)
        self.transitorio = transitorio


_CODIGOS_TRANSITORIOS = {408, 429, 500, 502, 503, 504}


def _transitorio(exc: Exception) -> bool:
    codigo = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    texto = str(exc).lower()
    return codigo in _CODIGOS_TRANSITORIOS or "timed out" in texto or "timeout" in texto


@dataclass(frozen=True)
class LLMResposta:
    texto: str
    tokens_entrada: int | None = None
    tokens_saida: int | None = None


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
            raise LLMError(f"Gemini falhou: {exc}", transitorio=_transitorio(exc)) from exc
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
            raise LLMError(f"Anthropic falhou: {exc}", transitorio=_transitorio(exc)) from exc
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


def criar_cliente(provider: str | None = None, model: str | None = None) -> LLMClient:
    """Cria o cliente do provedor escolhido (LLM_PROVIDER / LLM_MODEL no ambiente)."""
    provider = (provider or os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER).lower()
    model = model or os.getenv("LLM_MODEL") or None
    if provider == "gemini":
        return GeminiClient(model=model)
    if provider == "anthropic":
        return AnthropicClient(model=model)
    raise LLMError(f"Provedor '{provider}' não suportado. Use 'gemini' ou 'anthropic'.")


def cronometrar(fn, *args):
    t0 = time.perf_counter()
    resultado = fn(*args)
    return resultado, (time.perf_counter() - t0) * 1000
