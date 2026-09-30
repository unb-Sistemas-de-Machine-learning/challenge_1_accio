"""Verifica uma alegação: busca evidências na base e pede o veredito a um LLM (com guardrails).

    uv run python scripts/checar.py "O bacharelado em IA da UnB oferta 200 vagas por ano"
    uv run python scripts/checar.py "..." --dry-run          # mostra o prompt, sem chamar o LLM
    uv run python scripts/checar.py "..." --provider anthropic --modelo claude-haiku-4-5
    uv run python scripts/checar.py "..." --bruto            # mostra o prompt e a resposta crua do modelo
    uv run python scripts/checar.py -i                       # modo interativo (carrega os modelos uma vez)
    uv run python scripts/checar.py "..." --json

Provedor e modelo: LLM_PROVIDER / LLM_MODEL (padrão: gemini / gemini-3.1-flash-lite).
Chaves: GEMINI_API_KEY ou ANTHROPIC_API_KEY.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fato_unb.llm.checker import FactChecker  # noqa: E402
from fato_unb.llm.clients import LLMError, criar_cliente  # noqa: E402
from fato_unb.llm.guardrails import GuardrailConfig  # noqa: E402
from fato_unb.rag.retriever import Retriever  # noqa: E402


def imprimir(r, bruto: bool, dry_run: bool) -> None:
    v = r.veredito
    if bruto and r.prompt_usuario:
        from fato_unb.llm.prompts import SYSTEM_PROMPT

        print("\n===== PROMPT DO SISTEMA =====")
        print(SYSTEM_PROMPT)
        print("\n===== PROMPT DO USUÁRIO (alegação + evidências) =====")
        print(r.prompt_usuario)
        for i, texto in enumerate(r.respostas_brutas, 1):
            print(f"\n===== RESPOSTA BRUTA DO MODELO (tentativa {i}) =====")
            print(texto)
        print("\n===== RESULTADO APÓS OS GUARDRAILS =====")

    print(f"\nALEGAÇÃO : {v.afirmacao_analisada}")
    print(f"VEREDITO : {v.veredito.value}  (confiança {v.confianca:.2f})")
    print(f"\n{v.justificativa}\n")
    if v.fontes:
        print("FONTES:")
        for f in v.fontes:
            print(f"  - {f.title} [{f.source}]\n    {f.url}")
    print(f"\nGuardrails acionados: {', '.join(r.guardrails) or 'nenhum'}")
    if r.usou_llm:
        lat = f"{r.latencia_llm_ms:.0f} ms" if r.latencia_llm_ms is not None else "n/d"
        print(f"Modelo: {r.modelo} | tokens: {r.tokens_entrada} entrada / {r.tokens_saida} saída | {lat}")
    if dry_run and r.prompt_usuario:
        from fato_unb.llm.prompts import SYSTEM_PROMPT

        print("\n===== PROMPT DO SISTEMA =====")
        print(SYSTEM_PROMPT)
        print("\n===== PROMPT DO USUÁRIO =====")
        print(r.prompt_usuario)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("alegacao", nargs="*")
    parser.add_argument("-i", "--interativo", action="store_true", help="pergunta alegações em loop")
    parser.add_argument("--provider", choices=["gemini", "anthropic"])
    parser.add_argument("--modelo")
    parser.add_argument("--evidencias", type=int, default=GuardrailConfig.max_evidencias)
    parser.add_argument("--dry-run", action="store_true", help="não chama o LLM; mostra o prompt montado")
    parser.add_argument("--bruto", action="store_true", help="mostra o prompt enviado e a resposta crua do modelo")
    parser.add_argument("--json", action="store_true", help="imprime o VereditoJSON puro")
    args = parser.parse_args()
    if not args.alegacao and not args.interativo:
        parser.error("informe a alegação ou use -i")

    llm = None
    if not args.dry_run:
        try:
            llm = criar_cliente(args.provider, args.modelo)
        except LLMError as exc:
            sys.exit(f"Erro: {exc}\n(Use --dry-run para ver o prompt sem chamar o modelo.)")

    checker = FactChecker(Retriever.from_env(), llm, GuardrailConfig(max_evidencias=args.evidencias))

    def rodar(alegacao: str) -> None:
        r = checker.verificar(alegacao)
        if args.json:
            print(r.veredito.model_dump_json(indent=2))
        else:
            imprimir(r, args.bruto, args.dry_run)

    if args.alegacao:
        rodar(" ".join(args.alegacao))
    if args.interativo:
        print("Modo interativo. Digite uma alegação (ou 'sair').")
        while True:
            try:
                linha = input("\nalegação> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if linha.lower() in ("sair", "exit", "quit", ""):
                break
            rodar(linha)


if __name__ == "__main__":
    main()
