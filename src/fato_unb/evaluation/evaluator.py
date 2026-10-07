from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import logging
import re
import threading
import unicodedata
from enum import Enum
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from datasets import Dataset
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from pydantic import BaseModel
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_correctness,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig

from fato_unb.rag.models import VereditoType

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = PROJECT_ROOT / "ragas_testset_local.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "resultados_avaliacao.csv"
DEFAULT_BOT_OUTPUT = Path(__file__).resolve().parent / "resultados_respostas_bot.csv"
_EVALUATION_LOCK = threading.Lock()
METRIC_COLUMNS = {
    "faithfulness": "fidelidade_ao_contexto",
    "context_recall": "cobertura_do_contexto",
    "context_precision": "precisao_do_contexto",
    "answer_correctness": "correcao_da_resposta",
}

SYSTEM_PROMPT = (
    "Você é um assistente de checagem factual. Responda sempre em português "
    "brasileiro, usando somente as informações dos contextos fornecidos. "
    "Não invente fatos. Se os contextos não forem suficientes, diga isso "
    "claramente."
)


def parse_contexts(value: Any, row_number: int) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"A linha {row_number} não contém 'contextos_de_referencia'."
        )

    try:
        contexts = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Contextos inválidos na linha {row_number}: esperado um array JSON."
        ) from exc

    if not isinstance(contexts, list) or not all(
        isinstance(context, str) and context.strip() for context in contexts
    ):
        raise ValueError(
            f"Contextos inválidos na linha {row_number}: esperado um array "
            "JSON de textos não vazios."
        )
    return contexts


def generate_answer(
    model: ChatOllama, question: str, contexts: list[str]
) -> str:
    context_text = "\n\n".join(contexts)
    result = model.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Contextos:\n{context_text}\n\n"
                    f"Pergunta: {question}\n\n"
                    "Responda de forma direta e cite os fatos relevantes "
                    "presentes nos contextos."
                )
            ),
        ]
    )

    if not isinstance(result.content, str) or not result.content.strip():
        raise ValueError("O modelo local retornou uma resposta vazia.")
    return result.content.strip()


def load_testset(input_path: Path, limit: int | None) -> pd.DataFrame:
    if not input_path.is_file():
        raise FileNotFoundError(f"Arquivo de testset não encontrado: {input_path}")

    dataframe = pd.read_csv(input_path, encoding="utf-8-sig")
    required_columns = {
        "pergunta",
        "contextos_de_referencia",
        "resposta_de_referencia",
    }
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Colunas obrigatórias ausentes no testset: {missing}")

    if dataframe.empty:
        raise ValueError(f"O testset está vazio: {input_path}")
    if limit is not None:
        dataframe = dataframe.head(limit).copy()

    if dataframe["pergunta"].isna().any():
        raise ValueError("O testset contém perguntas vazias.")
    if dataframe["resposta_de_referencia"].isna().any():
        raise ValueError("O testset contém respostas de referência vazias.")
    return dataframe


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    return re.sub(r"\s+", " ", normalized).strip().casefold()


def response_data_from_json(
    response_json: BaseModel | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(response_json, BaseModel):
        return response_json.model_dump(mode="json")
    elif isinstance(response_json, dict):
        return response_json
    raise TypeError("A resposta do bot deve ser um modelo Pydantic ou um objeto JSON.")


def answer_from_response_json(response_json: BaseModel | dict[str, Any]) -> str:
    response_data = response_data_from_json(response_json)
    answer = response_data.get("justificativa") or response_data.get("resposta")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(
            "O JSON do bot precisa conter uma justificativa ou resposta textual."
        )

    parts = []
    verdict = response_data.get("veredito")
    if verdict:
        if isinstance(verdict, Enum):
            verdict_value = verdict.value
        elif isinstance(verdict, dict):
            verdict_value = verdict.get("value", verdict)
        else:
            verdict_value = verdict
        parts.append(f"Veredito: {verdict_value}")
    parts.append(f"Justificativa: {answer.strip()}")

    confidence = response_data.get("confianca")
    if confidence is not None:
        parts.append(f"Confiança: {confidence}")

    sources = response_data.get("fontes", [])
    if not isinstance(sources, list):
        raise TypeError("As fontes da resposta do bot devem ser uma lista.")
    if sources:
        formatted_sources = []
        for source in sources:
            if not isinstance(source, dict):
                raise TypeError("Cada fonte citada deve ser um objeto JSON.")
            title = source.get("title", "")
            url = source.get("url", "")
            origin = source.get("source", "")
            formatted_sources.append(f"{title} ({origin}) {url}".strip())
        parts.append("Fontes: " + "; ".join(formatted_sources))
    return "\n".join(parts)


def expected_verdict_from_label(label: Any) -> VereditoType:
    normalized_label = str(label).strip().upper()
    verdicts = {
        "VERDADEIRO": VereditoType.CONFIRMADO_OFICIALMENTE,
        "FALSO": VereditoType.BOATO_SEM_REGISTRO,
    }
    if normalized_label in verdicts:
        return verdicts[normalized_label]
    try:
        return VereditoType(normalized_label)
    except ValueError as exc:
        raise ValueError(f"Rótulo de veredito desconhecido no testset: {label}") from exc


def reference_for_assertion(row: dict[str, Any]) -> str:
    expected_verdict = expected_verdict_from_label(row["VEREDITO"])
    reference_answer = row.get("resposta_de_referencia")
    if not isinstance(reference_answer, str) or not reference_answer.strip():
        raise ValueError("O testset contém uma resposta de referência vazia.")
    return (
        f"Veredito esperado: {expected_verdict.value}. "
        f"Referência factual: {reference_answer.strip()}"
    )


def create_evaluator_metrics() -> list[Any]:
    metrics = [
        copy.deepcopy(faithfulness),
        copy.deepcopy(context_recall),
        copy.deepcopy(context_precision),
        copy.deepcopy(answer_correctness),
    ]
    prompt_attributes = {
        "faithfulness": ("statement_generator_prompt", "nli_statements_prompt"),
        "context_recall": ("context_recall_prompt",),
        "context_precision": ("context_precision_prompt",),
        "answer_correctness": ("statement_generator_prompt", "correctness_prompt"),
    }
    for metric in metrics:
        for attribute in prompt_attributes[metric.name]:
            prompt = getattr(metric, attribute)
            if hasattr(prompt, "language"):
                prompt.language = "portuguese"
    return metrics


def run_ragas_evaluation(
    records: list[dict[str, Any]], judge_model_name: str
) -> pd.DataFrame:
    async_client_kwargs = {
        "limits": httpx.Limits(max_keepalive_connections=0),
    }
    evaluator_llm = LangchainLLMWrapper(
        ChatOllama(
            model=judge_model_name,
            temperature=0.0,
            num_gpu=0,
            format="json",
            async_client_kwargs=async_client_kwargs,
        )
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model="bge-m3", num_gpu=0)
    )
    with _EVALUATION_LOCK:
        result = evaluate(
            dataset=Dataset.from_list(records),
            metrics=create_evaluator_metrics(),
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            run_config=RunConfig(max_workers=1, max_retries=1, timeout=300),
            raise_exceptions=True,
        )
    return result.to_pandas().reset_index(drop=True)


def metric_scores_from_result(metric_results: pd.DataFrame) -> dict[str, float | None]:
    if len(metric_results) != 1:
        raise RuntimeError(
            "A avaliação de uma afirmação deveria retornar exatamente uma linha."
        )

    scores: dict[str, float | None] = {}
    for source_name, output_name in METRIC_COLUMNS.items():
        if source_name not in metric_results:
            raise RuntimeError(f"O Ragas não retornou a métrica {source_name}.")
        value = metric_results.iloc[0][source_name]
        scores[output_name] = None if pd.isna(value) else float(value)
    return scores


def checkpoint_metadata_path(output_path: Path) -> Path:
    return output_path.with_suffix(f"{output_path.suffix}.checkpoint.json")


def dataframe_fingerprint(dataframe: pd.DataFrame) -> str:
    serialized_dataframe = dataframe.to_json(
        orient="split",
        force_ascii=False,
        date_format="iso",
    )
    return hashlib.sha256(serialized_dataframe.encode("utf-8")).hexdigest()


def checkpoint_configuration(
    dataframe: pd.DataFrame,
    *,
    mode: str,
    judge_model_name: str,
    answer_model_name: str | None,
    output_columns: list[str],
    source_fingerprint: str | None = None,
) -> dict[str, Any]:
    return {
        "dataframe_sha256": source_fingerprint or dataframe_fingerprint(dataframe),
        "mode": mode,
        "judge_model": judge_model_name,
        "answer_model": answer_model_name,
        "metrics": list(METRIC_COLUMNS),
        "output_columns": output_columns,
    }


def initialize_checkpoint(
    output_path: Path | None,
    dataframe: pd.DataFrame,
    *,
    mode: str,
    judge_model_name: str,
    answer_model_name: str | None,
    output_columns: list[str],
    resume: bool,
    source_fingerprint: str | None = None,
    source_row_count: int | None = None,
) -> tuple[set[int], list[dict[str, Any]]]:
    if output_path is None:
        if resume:
            raise ValueError("A retomada exige um caminho de saída.")
        return set(), []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = checkpoint_metadata_path(output_path)
    expected_configuration = checkpoint_configuration(
        dataframe,
        mode=mode,
        judge_model_name=judge_model_name,
        answer_model_name=answer_model_name,
        output_columns=output_columns,
        source_fingerprint=source_fingerprint,
    )

    if not resume:
        pd.DataFrame(columns=output_columns).to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig",
        )
        temporary_metadata_path = metadata_path.with_suffix(
            f"{metadata_path.suffix}.tmp"
        )
        temporary_metadata_path.write_text(
            json.dumps(expected_configuration, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_metadata_path.replace(metadata_path)
        return set(), []

    if not output_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            "Não há checkpoint compatível para retomar. Execute primeiro sem "
            "--resume, usando o mesmo caminho de saída."
        )

    saved_configuration = json.loads(metadata_path.read_text(encoding="utf-8"))
    if saved_configuration != expected_configuration:
        raise ValueError(
            "O checkpoint não corresponde ao testset, modo, modelos ou métricas "
            "desta execução. Use outro arquivo de saída ou inicie sem --resume."
        )

    saved_results = pd.read_csv(output_path, encoding="utf-8-sig")
    if list(saved_results.columns) != output_columns:
        raise ValueError(
            "As colunas do CSV salvo não correspondem ao checkpoint esperado."
        )
    if saved_results.empty:
        return set(), []

    row_indices = pd.to_numeric(
        saved_results["indice_linha_testset"],
        errors="raise",
    )
    maximum_row_count = (
        len(dataframe) if source_row_count is None else source_row_count
    )
    if (
        row_indices.isna().any()
        or (row_indices % 1 != 0).any()
        or row_indices.duplicated().any()
        or (row_indices < 0).any()
        or (row_indices >= maximum_row_count).any()
    ):
        raise ValueError("O CSV do checkpoint contém índices de linha inválidos.")

    return (
        {int(row_index) for row_index in row_indices},
        saved_results.to_dict(orient="records"),
    )


def append_checkpoint_row(output_path: Path | None, row: dict[str, Any]) -> None:
    if output_path is None:
        return
    pd.DataFrame([row]).to_csv(
        output_path,
        mode="a",
        header=False,
        index=False,
        encoding="utf-8",
    )


async def avaliar_resposta_do_bot(
    afirmacao: str,
    response_json: BaseModel | dict[str, Any],
    retrieved_contexts: list[str],
    *,
    testset_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_BOT_OUTPUT,
    judge_model_name: str = "qwen2.5:3b",
) -> dict[str, float | None] | None:
    """Avalia uma resposta estruturada do bot antes de ela ser enviada.

    Só calcula pontuações quando a afirmação corresponde, sem aproximação, a
    uma afirmação do testset, para não associar uma referência incorreta.
    """
    if not isinstance(afirmacao, str) or not afirmacao.strip():
        raise ValueError("A afirmação do bot não pode ser vazia.")
    if not isinstance(retrieved_contexts, list) or not all(
        isinstance(context, str) for context in retrieved_contexts
    ):
        raise TypeError("Os contextos recuperados devem ser uma lista de textos.")

    testset = load_testset(testset_path, limit=None)
    if "afirmacao" not in testset or "VEREDITO" not in testset:
        raise ValueError("O testset deve conter as colunas 'afirmacao' e 'VEREDITO'.")
    normalized_assertion = normalize_text(afirmacao)
    matching_rows = testset[
        testset["afirmacao"].map(normalize_text) == normalized_assertion
    ]
    if matching_rows.empty:
        logger.warning(
            "Afirmação não encontrada no testset; resposta do bot não será pontuada."
        )
        return None
    if len(matching_rows) > 1:
        logger.warning(
            "Afirmação aparece mais de uma vez no testset; resposta do bot não será pontuada."
        )
        return None

    reference_row = matching_rows.iloc[0]
    answer = answer_from_response_json(response_json)
    response_data = response_data_from_json(response_json)
    verdict_value = response_data.get("veredito")
    if not verdict_value:
        raise ValueError("O JSON do bot precisa conter o campo 'veredito'.")
    if isinstance(verdict_value, Enum):
        verdict_value = verdict_value.value
    expected_verdict = expected_verdict_from_label(reference_row["VEREDITO"])
    row_data = reference_row.to_dict()
    record = {
        "user_input": afirmacao,
        "response": answer,
        "retrieved_contexts": retrieved_contexts,
        "reference": reference_for_assertion(row_data),
    }
    metric_results = await asyncio.to_thread(
        run_ragas_evaluation,
        [record],
        judge_model_name,
    )
    scores = metric_scores_from_result(metric_results)
    scores["acuracia_do_veredito"] = float(verdict_value == expected_verdict.value)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_row = {
        "afirmacao": afirmacao,
        "veredito_esperado": expected_verdict.value,
        "veredito_do_bot": verdict_value,
        "resposta_avaliada": answer,
        "fontes_citadas": json.dumps(
            response_data.get("fontes", []),
            ensure_ascii=False,
        ),
        "contextos_recuperados": json.dumps(
            retrieved_contexts, ensure_ascii=False
        ),
        "resposta_de_referencia": reference_row["resposta_de_referencia"],
        **scores,
    }
    result_frame = pd.DataFrame([result_row])
    with _EVALUATION_LOCK:
        result_frame.to_csv(
            output_path,
            mode="a",
            header=not output_path.exists(),
            index=False,
            encoding="utf-8-sig",
        )
    return scores


def evaluate_bot_testset(
    dataframe: pd.DataFrame,
    judge_model_name: str,
    *,
    output_path: Path | None = None,
    resume: bool = False,
    source_fingerprint: str | None = None,
    source_row_count: int | None = None,
) -> tuple[pd.DataFrame, dict[str, float | None]]:
    from fato_unb.bots.telegram_bot import (
        verificar_afirmacao_provisoria_com_contextos,
    )

    if "afirmacao" not in dataframe or "VEREDITO" not in dataframe:
        raise ValueError("O testset do bot deve conter 'afirmacao' e 'VEREDITO'.")
    if dataframe.empty:
        raise ValueError("O testset do bot não pode estar vazio.")

    generated_columns = [
        "resposta_do_bot",
        "veredito_esperado",
        "veredito_do_bot",
        "acerto_do_veredito",
        "fontes_citadas",
        "contextos_recuperados",
        *METRIC_COLUMNS.values(),
    ]
    output_columns = list(
        dict.fromkeys(
            ["indice_linha_testset", *dataframe.columns, *generated_columns]
        )
    )
    completed_indices, evaluated_rows = initialize_checkpoint(
        output_path,
        dataframe,
        mode="bot",
        judge_model_name=judge_model_name,
        answer_model_name=None,
        output_columns=output_columns,
        resume=resume,
        source_fingerprint=source_fingerprint,
        source_row_count=source_row_count,
    )
    for row_index, row in enumerate(dataframe.to_dict(orient="records")):
        if row_index in completed_indices:
            continue
        row_number = row_index + 2
        evaluated_row: dict[str, Any] | None = None
        try:
            assertion = row["afirmacao"]
            if not isinstance(assertion, str) or not assertion.strip():
                raise ValueError("A afirmação está vazia ou inválida.")

            verdict, contexts = verificar_afirmacao_provisoria_com_contextos(
                assertion,
                raise_on_error=True,
            )
            expected_verdict = expected_verdict_from_label(row["VEREDITO"])
            response_data = verdict.model_dump(mode="json")
            record = {
                "user_input": assertion,
                "response": answer_from_response_json(response_data),
                "retrieved_contexts": contexts,
                "reference": reference_for_assertion(row),
            }
            scores = metric_scores_from_result(
                run_ragas_evaluation([record], judge_model_name)
            )
            evaluated_row = {
                "indice_linha_testset": row_index,
                **row,
                "resposta_do_bot": record["response"],
                "veredito_esperado": expected_verdict.value,
                "veredito_do_bot": verdict.veredito.value,
                "acerto_do_veredito": verdict.veredito == expected_verdict,
                "fontes_citadas": json.dumps(
                    [
                        source.model_dump(mode="json")
                        for source in verdict.fontes
                    ],
                    ensure_ascii=False,
                ),
                "contextos_recuperados": json.dumps(
                    contexts, ensure_ascii=False
                ),
                **scores,
            }
        except Exception:
            logger.exception(
                "Falha ao avaliar a linha %d do testset do bot; linha ignorada.",
                row_number,
            )
            continue

        if evaluated_row is None:
            raise RuntimeError("A avaliação terminou sem produzir uma linha.")
        append_checkpoint_row(output_path, evaluated_row)
        evaluated_rows.append(evaluated_row)

    output = pd.DataFrame(evaluated_rows, columns=output_columns)
    summary: dict[str, float | None] = {}
    for column in METRIC_COLUMNS.values():
        mean_score = output[column].mean() if not output.empty else float("nan")
        summary[column] = None if pd.isna(mean_score) else float(mean_score)
    accuracy = (
        output["acerto_do_veredito"].mean() if not output.empty else float("nan")
    )
    summary["acuracia_do_veredito"] = (
        None if pd.isna(accuracy) else float(accuracy)
    )
    return output, summary


def evaluate_testset(
    dataframe: pd.DataFrame,
    answer_model_name: str,
    judge_model_name: str,
    *,
    output_path: Path | None = None,
    resume: bool = False,
    source_fingerprint: str | None = None,
    source_row_count: int | None = None,
) -> tuple[pd.DataFrame, dict[str, float | None]]:
    async_client_kwargs = {
        "limits": httpx.Limits(max_keepalive_connections=0),
    }
    answer_model = ChatOllama(
        model=answer_model_name,
        temperature=0.0,
        num_gpu=0,
        async_client_kwargs=async_client_kwargs,
    )

    generated_columns = ["resposta_do_modelo", *METRIC_COLUMNS.values()]
    output_columns = list(
        dict.fromkeys(
            ["indice_linha_testset", *dataframe.columns, *generated_columns]
        )
    )
    completed_indices, evaluated_rows = initialize_checkpoint(
        output_path,
        dataframe,
        mode="local",
        judge_model_name=judge_model_name,
        answer_model_name=answer_model_name,
        output_columns=output_columns,
        resume=resume,
        source_fingerprint=source_fingerprint,
        source_row_count=source_row_count,
    )
    for row_index, row in enumerate(dataframe.to_dict(orient="records")):
        if row_index in completed_indices:
            continue
        row_number = row_index + 2
        evaluated_row: dict[str, Any] | None = None
        try:
            question = row["pergunta"]
            if not isinstance(question, str) or not question.strip():
                raise ValueError("A pergunta está vazia ou inválida.")
            contexts = parse_contexts(row["contextos_de_referencia"], row_number)
            answer = generate_answer(answer_model, question, contexts)
            record = {
                "user_input": question,
                "response": answer,
                "retrieved_contexts": contexts,
                "reference": row["resposta_de_referencia"],
            }
            scores = metric_scores_from_result(
                run_ragas_evaluation([record], judge_model_name)
            )
            evaluated_row = {
                "indice_linha_testset": row_index,
                **row,
                "resposta_do_modelo": answer,
                **scores,
            }
        except Exception:
            logger.exception(
                "Falha ao avaliar a linha %d do testset; linha ignorada.",
                row_number,
            )
            continue

        if evaluated_row is None:
            raise RuntimeError("A avaliação terminou sem produzir uma linha.")
        append_checkpoint_row(output_path, evaluated_row)
        evaluated_rows.append(evaluated_row)

    output = pd.DataFrame(evaluated_rows, columns=output_columns)
    summary: dict[str, float | None] = {}
    for column in METRIC_COLUMNS.values():
        mean_score = output[column].mean() if not output.empty else float("nan")
        summary[column] = None if pd.isna(mean_score) else float(mean_score)
    return output, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Avalia fidelidade e cobertura com Ragas. Use --bot para avaliar "
            "as respostas reais do verificador sobre afirmações do testset."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"CSV do testset (padrão: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"CSV com respostas e métricas (padrão: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Avalia somente as primeiras N linhas, útil para um teste rápido.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Retoma do CSV de saída, ignorando linhas já avaliadas com a mesma configuração.",
    )
    parser.add_argument(
        "--answer-model",
        default="qwen2.5:7b",
        help="Modelo Ollama que responderá às perguntas no modo padrão.",
    )
    parser.add_argument(
        "--judge-model",
        default="qwen2.5:3b",
        help="Modelo Ollama usado pelo Ragas como avaliador.",
    )
    parser.add_argument(
        "--bot",
        action="store_true",
        help="Avalia afirmações com o verificador do bot e registra veredito/fontes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit deve ser maior que zero.")

    full_testset = load_testset(args.input, None)
    source_fingerprint = dataframe_fingerprint(full_testset)
    testset = (
        full_testset.head(args.limit).copy()
        if args.limit is not None
        else full_testset
    )
    if args.bot:
        print(f"Avaliando {len(testset)} afirmações com o verificador do bot.")
        _, summary = evaluate_bot_testset(
            testset,
            judge_model_name=args.judge_model,
            output_path=args.output,
            resume=args.resume,
            source_fingerprint=source_fingerprint,
            source_row_count=len(full_testset),
        )
    else:
        print(f"Avaliando {len(testset)} perguntas com modelos locais.")
        _, summary = evaluate_testset(
            testset,
            answer_model_name=args.answer_model,
            judge_model_name=args.judge_model,
            output_path=args.output,
            resume=args.resume,
            source_fingerprint=source_fingerprint,
            source_row_count=len(full_testset),
        )

    summary_path = args.output.with_name(f"{args.output.stem}_resumo.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Resultados por pergunta: {args.output}")
    print(f"Médias das métricas: {summary_path}")
    for metric_name, score in summary.items():
        formatted_score = "indisponível" if score is None else f"{score:.3f}"
        print(f"- {metric_name}: {formatted_score}")


if __name__ == "__main__":
    main()