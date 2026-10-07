"""Demonstra a extração de texto de um PDF hospedado em uma URL."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import requests

from fato_unb.ingestion.pdf import extract_pdf_text


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("informe um número inteiro positivo") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("o valor deve ser maior que zero")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Baixa um PDF por URL, extrai suas páginas e mostra o texto."
    )
    parser.add_argument("url", help="URL HTTP(S) do PDF")
    parser.add_argument(
        "--max-pages",
        type=positive_int,
        default=5,
        help="máximo de páginas exibidas no terminal (padrão: 5)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="arquivo de texto opcional para salvar a extração completa",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        extraction = extract_pdf_text(args.url)
    except (requests.RequestException, ValueError) as exc:
        parser.exit(1, f"Falha ao extrair PDF: {exc}\n")
    except Exception as exc:
        parser.exit(1, f"Erro inesperado na extração: {exc}\n")

    print(f"URL: {extraction.source}")
    print(f"Páginas encontradas: {extraction.page_count}")

    for page in extraction.pages[: args.max_pages]:
        print(f"\n{'=' * 20} Página {page.page_number} {'=' * 20}")
        print(page.text or "[Nenhum texto extraível nesta página]")

    remaining_pages = extraction.page_count - min(
        extraction.page_count, args.max_pages
    )
    if remaining_pages:
        print(f"\n... {remaining_pages} página(s) omitida(s) no terminal.")

    if args.output:
        args.output.write_text(extraction.full_text, encoding="utf-8")
        print(f"\nExtração completa salva em: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
