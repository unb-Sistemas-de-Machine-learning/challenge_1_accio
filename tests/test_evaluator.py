import asyncio
import json

import pandas as pd

from fato_unb.rag.models import FonteCitada, VereditoJSON, VereditoType
from fato_unb.evaluation import evaluator


def test_normalize_assertion_ignores_case_accents_and_extra_spaces():
    assert evaluator.normalize_text("  PÓS-graduação  ") == "pos-graduacao"


def test_response_json_includes_verdict_justification_and_sources():
    assert evaluator.answer_from_response_json(
        {
            "veredito": "CONFIRMADO_OFICIALMENTE",
            "justificativa": "A informação consta na fonte oficial.",
            "fontes": [
                {
                    "title": "Notícia da UnB",
                    "url": "https://noticias.unb.br/teste",
                    "source": "UnB Notícias",
                }
            ],
        }
    ) == (
        "Veredito: CONFIRMADO_OFICIALMENTE\n"
        "Justificativa: A informação consta na fonte oficial.\n"
        "Fontes: Notícia da UnB (UnB Notícias) https://noticias.unb.br/teste"
    )


def test_binary_testset_labels_map_to_verdict_categories():
    assert (
        evaluator.expected_verdict_from_label("VERDADEIRO")
        == VereditoType.CONFIRMADO_OFICIALMENTE
    )
    assert (
        evaluator.expected_verdict_from_label("FALSO")
        == VereditoType.BOATO_SEM_REGISTRO
    )


def test_bot_response_uses_testset_reference_and_persists_metrics(
    tmp_path, monkeypatch
):
    testset_path = tmp_path / "testset.csv"
    pd.DataFrame(
        [
            {
                "pergunta": "Qual é a afirmação avaliada?",
                "afirmacao": "A UnB publicou o calendário.",
                "VEREDITO": "VERDADEIRO",
                "contextos_de_referencia": json.dumps(
                    ["O calendário foi publicado pela UnB."],
                    ensure_ascii=False,
                ),
                "resposta_de_referencia": "Sim, a UnB publicou o calendário.",
            }
        ]
    ).to_csv(testset_path, index=False, encoding="utf-8-sig")

    def fake_run_ragas(records, judge_model_name):
        assert judge_model_name == "qwen2.5:3b"
        assert records == [
            {
                "user_input": " A UNB publicou o calendário. ",
                "response": (
                    "Veredito: CONFIRMADO_OFICIALMENTE\n"
                    "Justificativa: Sim, a fonte oficial publicou o calendário."
                ),
                "retrieved_contexts": ["O calendário foi publicado pela UnB."],
                "reference": (
                    "Veredito esperado: CONFIRMADO_OFICIALMENTE. "
                    "Referência factual: Sim, a UnB publicou o calendário."
                ),
            }
        ]
        return pd.DataFrame(
            [
                {
                    "faithfulness": 0.9,
                    "context_recall": 0.7,
                    "context_precision": 0.8,
                    "answer_correctness": 0.85,
                }
            ]
        )

    monkeypatch.setattr(evaluator, "run_ragas_evaluation", fake_run_ragas)
    output_path = tmp_path / "scores.csv"

    scores = asyncio.run(
        evaluator.avaliar_resposta_do_bot(
            afirmacao=" A UNB publicou o calendário. ",
            response_json={
                "veredito": "CONFIRMADO_OFICIALMENTE",
                "justificativa": "Sim, a fonte oficial publicou o calendário.",
            },
            retrieved_contexts=["O calendário foi publicado pela UnB."],
            testset_path=testset_path,
            output_path=output_path,
        )
    )

    assert scores == {
        "fidelidade_ao_contexto": 0.9,
        "cobertura_do_contexto": 0.7,
        "precisao_do_contexto": 0.8,
        "correcao_da_resposta": 0.85,
        "acuracia_do_veredito": 1.0,
    }
    saved = pd.read_csv(output_path, encoding="utf-8-sig")
    assert len(saved) == 1
    assert saved.loc[0, "resposta_avaliada"] == (
        "Veredito: CONFIRMADO_OFICIALMENTE\n"
        "Justificativa: Sim, a fonte oficial publicou o calendário."
    )


def test_bot_response_is_not_scored_without_exact_testset_match(
    tmp_path, monkeypatch
):
    testset_path = tmp_path / "testset.csv"
    pd.DataFrame(
        [
            {
                "pergunta": "Pergunta conhecida?",
                "afirmacao": "Afirmação conhecida.",
                "VEREDITO": "VERDADEIRO",
                "contextos_de_referencia": json.dumps(["Contexto."]),
                "resposta_de_referencia": "Referência.",
            }
        ]
    ).to_csv(testset_path, index=False, encoding="utf-8-sig")

    def unexpected_evaluation(*args, **kwargs):
        raise AssertionError("Pergunta ausente do testset não deve ser pontuada.")

    monkeypatch.setattr(evaluator, "run_ragas_evaluation", unexpected_evaluation)

    result = asyncio.run(
        evaluator.avaliar_resposta_do_bot(
            afirmacao="Outra afirmação.",
            response_json={"justificativa": "Resposta."},
            retrieved_contexts=["Contexto."],
            testset_path=testset_path,
            output_path=tmp_path / "scores.csv",
        )
    )

    assert result is None


def test_bot_testset_evaluation_uses_assertion_and_bot_response(monkeypatch):
    dataframe = pd.DataFrame(
        [
            {
                "pergunta": "Pergunta que não deve ser usada.",
                "afirmacao": "A UnB publicou o calendário.",
                "VEREDITO": "VERDADEIRO",
                "resposta_de_referencia": "A UnB publicou o calendário.",
            }
        ]
    )
    verdict = VereditoJSON(
        veredito=VereditoType.INCONCLUSIVO,
        justificativa="Ainda não consigo confirmar a afirmação.",
        fontes=[
            FonteCitada(
                title="Notícia da UnB",
                url="https://noticias.unb.br/calendario",
                source="UnB Notícias",
            )
        ],
        confianca=0.8,
        afirmacao_analisada="A UnB publicou o calendário.",
    )

    from fato_unb.bots import telegram_bot

    def fake_verifier(assertion, *, raise_on_error):
        assert assertion == "A UnB publicou o calendário."
        assert raise_on_error is True
        return verdict, ["O calendário foi publicado pela UnB."]

    monkeypatch.setattr(
        telegram_bot,
        "verificar_afirmacao_provisoria_com_contextos",
        fake_verifier,
    )

    def fake_run_ragas(records, judge_model_name):
        assert records == [
            {
                "user_input": "A UnB publicou o calendário.",
                "response": (
                    "Veredito: INCONCLUSIVO\n"
                    "Justificativa: Ainda não consigo confirmar a afirmação.\n"
                    "Confiança: 0.8\n"
                    "Fontes: Notícia da UnB (UnB Notícias) "
                    "https://noticias.unb.br/calendario"
                ),
                "retrieved_contexts": ["O calendário foi publicado pela UnB."],
                "reference": (
                    "Veredito esperado: CONFIRMADO_OFICIALMENTE. "
                    "Referência factual: A UnB publicou o calendário."
                ),
            }
        ]
        return pd.DataFrame(
            [
                {
                    "faithfulness": 0.75,
                    "context_recall": 0.9,
                    "context_precision": 0.8,
                    "answer_correctness": 0.7,
                }
            ]
        )

    monkeypatch.setattr(evaluator, "run_ragas_evaluation", fake_run_ragas)
    output, summary = evaluator.evaluate_bot_testset(dataframe, "qwen2.5:3b")

    assert output.loc[0, "afirmacao"] == "A UnB publicou o calendário."
    assert output.loc[0, "veredito_esperado"] == "CONFIRMADO_OFICIALMENTE"
    assert output.loc[0, "veredito_do_bot"] == "INCONCLUSIVO"
    assert not output.loc[0, "acerto_do_veredito"]
    assert "https://noticias.unb.br/calendario" in output.loc[0, "fontes_citadas"]
    assert output.loc[0, "precisao_do_contexto"] == 0.8
    assert output.loc[0, "correcao_da_resposta"] == 0.7
    assert summary == {
        "fidelidade_ao_contexto": 0.75,
        "cobertura_do_contexto": 0.9,
        "precisao_do_contexto": 0.8,
        "correcao_da_resposta": 0.7,
        "acuracia_do_veredito": 0.0,
    }


def test_batch_evaluation_exports_one_version_of_metric_columns(monkeypatch):
    dataframe = pd.DataFrame(
        [
            {
                "pergunta": "Pergunta?",
                "contextos_de_referencia": json.dumps(["Contexto."]),
                "resposta_de_referencia": "Referência.",
            }
        ]
    )

    monkeypatch.setattr(
        evaluator,
        "generate_answer",
        lambda model, question, contexts: "Resposta local.",
    )
    monkeypatch.setattr(
        evaluator,
        "run_ragas_evaluation",
        lambda records, judge_model_name: pd.DataFrame(
            [
                {
                    "faithfulness": 0.9,
                    "context_recall": 0.8,
                    "context_precision": 0.7,
                    "answer_correctness": 0.6,
                }
            ]
        ),
    )

    output, summary = evaluator.evaluate_testset(
        dataframe,
        answer_model_name="qwen2.5:7b",
        judge_model_name="qwen2.5:3b",
    )

    assert output.columns.tolist() == [
        "indice_linha_testset",
        "pergunta",
        "contextos_de_referencia",
        "resposta_de_referencia",
        "resposta_do_modelo",
        "fidelidade_ao_contexto",
        "cobertura_do_contexto",
        "precisao_do_contexto",
        "correcao_da_resposta",
    ]
    assert summary == {
        "fidelidade_ao_contexto": 0.9,
        "cobertura_do_contexto": 0.8,
        "precisao_do_contexto": 0.7,
        "correcao_da_resposta": 0.6,
    }


def test_bot_batch_resume_skips_saved_rows_and_retries_failed_rows(
    tmp_path, monkeypatch
):
    dataframe = pd.DataFrame(
        [
            {
                "afirmacao": "Afirmação 1.",
                "VEREDITO": "VERDADEIRO",
                "resposta_de_referencia": "Referência 1.",
            },
            {
                "afirmacao": "Afirmação 2.",
                "VEREDITO": "FALSO",
                "resposta_de_referencia": "Referência 2.",
            },
        ]
    )
    output_path = tmp_path / "scores.csv"
    from fato_unb.bots import telegram_bot

    def fake_verifier(assertion, *, raise_on_error):
        assert raise_on_error is True
        return (
            VereditoJSON(
                veredito=VereditoType.INCONCLUSIVO,
                justificativa="Ainda inconclusivo.",
                confianca=0.5,
                afirmacao_analisada=assertion,
            ),
            ["Contexto recuperado."],
        )

    evaluated_assertions = []

    def first_run_ragas(records, judge_model_name):
        assertion = records[0]["user_input"]
        evaluated_assertions.append(assertion)
        if assertion == "Afirmação 2.":
            raise ValueError("Falha simulada.")
        return pd.DataFrame(
            [
                {
                    "faithfulness": 0.8,
                    "context_recall": 0.7,
                    "context_precision": 0.6,
                    "answer_correctness": 0.5,
                }
            ]
        )

    monkeypatch.setattr(
        telegram_bot,
        "verificar_afirmacao_provisoria_com_contextos",
        fake_verifier,
    )
    monkeypatch.setattr(evaluator, "run_ragas_evaluation", first_run_ragas)

    partial_output, _ = evaluator.evaluate_bot_testset(
        dataframe,
        "qwen2.5:3b",
        output_path=output_path,
    )
    assert partial_output["indice_linha_testset"].tolist() == [0]
    assert evaluated_assertions == ["Afirmação 1.", "Afirmação 2."]
    assert pd.read_csv(output_path, encoding="utf-8-sig")[
        "indice_linha_testset"
    ].tolist() == [0]

    evaluated_assertions.clear()

    def resumed_run_ragas(records, judge_model_name):
        evaluated_assertions.append(records[0]["user_input"])
        return pd.DataFrame(
            [
                {
                    "faithfulness": 0.9,
                    "context_recall": 0.8,
                    "context_precision": 0.7,
                    "answer_correctness": 0.6,
                }
            ]
        )

    monkeypatch.setattr(evaluator, "run_ragas_evaluation", resumed_run_ragas)
    resumed_output, _ = evaluator.evaluate_bot_testset(
        dataframe,
        "qwen2.5:3b",
        output_path=output_path,
        resume=True,
    )

    assert evaluated_assertions == ["Afirmação 2."]
    assert resumed_output["indice_linha_testset"].tolist() == [0, 1]
    saved = pd.read_csv(output_path, encoding="utf-8-sig")
    assert saved["indice_linha_testset"].tolist() == [0, 1]


def test_resume_rejects_changed_testset(tmp_path):
    original = pd.DataFrame(
        [{"pergunta": "Pergunta original?", "resposta_de_referencia": "Resposta."}]
    )
    changed = original.copy()
    changed.loc[0, "pergunta"] = "Pergunta diferente?"
    output_path = tmp_path / "scores.csv"
    output_columns = [
        "indice_linha_testset",
        *original.columns,
        "resposta_do_modelo",
        *evaluator.METRIC_COLUMNS.values(),
    ]
    evaluator.initialize_checkpoint(
        output_path,
        original,
        mode="local",
        judge_model_name="qwen2.5:3b",
        answer_model_name="qwen2.5:7b",
        output_columns=output_columns,
        resume=False,
    )

    import pytest

    with pytest.raises(ValueError, match="não corresponde ao testset"):
        evaluator.initialize_checkpoint(
            output_path,
            changed,
            mode="local",
            judge_model_name="qwen2.5:3b",
            answer_model_name="qwen2.5:7b",
            output_columns=output_columns,
            resume=True,
        )


def test_checkpoint_can_resume_with_a_larger_limit(tmp_path):
    full_dataframe = pd.DataFrame(
        [
            {"pergunta": "Pergunta 1?", "resposta_de_referencia": "Resposta 1."},
            {"pergunta": "Pergunta 2?", "resposta_de_referencia": "Resposta 2."},
        ]
    )
    first_batch = full_dataframe.head(1)
    output_path = tmp_path / "scores.csv"
    output_columns = [
        "indice_linha_testset",
        *full_dataframe.columns,
        "resposta_do_modelo",
        *evaluator.METRIC_COLUMNS.values(),
    ]
    source_fingerprint = evaluator.dataframe_fingerprint(full_dataframe)
    evaluator.initialize_checkpoint(
        output_path,
        first_batch,
        mode="local",
        judge_model_name="qwen2.5:3b",
        answer_model_name="qwen2.5:7b",
        output_columns=output_columns,
        resume=False,
        source_fingerprint=source_fingerprint,
        source_row_count=len(full_dataframe),
    )
    evaluator.append_checkpoint_row(
        output_path,
        {
            "indice_linha_testset": 0,
            "pergunta": "Pergunta 1?",
            "resposta_de_referencia": "Resposta 1.",
            "resposta_do_modelo": "Resposta gerada.",
            "fidelidade_ao_contexto": 0.9,
            "cobertura_do_contexto": 0.8,
            "precisao_do_contexto": 0.7,
            "correcao_da_resposta": 0.6,
        },
    )

    completed_indices, saved_rows = evaluator.initialize_checkpoint(
        output_path,
        full_dataframe,
        mode="local",
        judge_model_name="qwen2.5:3b",
        answer_model_name="qwen2.5:7b",
        output_columns=output_columns,
        resume=True,
        source_fingerprint=source_fingerprint,
        source_row_count=len(full_dataframe),
    )

    assert completed_indices == {0}
    assert len(saved_rows) == 1


def test_all_ragas_metric_prompts_are_configured_for_portuguese():
    metrics = evaluator.create_evaluator_metrics()

    assert {metric.name for metric in metrics} == set(evaluator.METRIC_COLUMNS)
    for metric in metrics:
        for name in (
            "statement_generator_prompt",
            "nli_statements_prompt",
            "context_recall_prompt",
            "context_precision_prompt",
            "correctness_prompt",
        ):
            prompt = getattr(metric, name, None)
            if prompt is not None:
                assert prompt.language == "portuguese"


def test_ragas_judge_uses_json_mode_and_all_four_metrics(monkeypatch):
    calls = {}

    class FakeEvaluation:
        def to_pandas(self):
            return pd.DataFrame(
                [
                    {
                        "faithfulness": 0.8,
                        "context_recall": 0.7,
                        "context_precision": 0.6,
                        "answer_correctness": 0.5,
                    }
                ]
            )

    def fake_chat_ollama(**kwargs):
        calls["chat_ollama"] = kwargs
        return "chat-model"

    def fake_evaluate(**kwargs):
        calls["evaluate"] = kwargs
        return FakeEvaluation()

    monkeypatch.setattr(evaluator, "ChatOllama", fake_chat_ollama)
    monkeypatch.setattr(
        evaluator,
        "LangchainLLMWrapper",
        lambda model: ("llm-wrapper", model),
    )
    monkeypatch.setattr(
        evaluator,
        "OllamaEmbeddings",
        lambda **kwargs: ("embeddings", kwargs),
    )
    monkeypatch.setattr(
        evaluator,
        "LangchainEmbeddingsWrapper",
        lambda embeddings: ("embeddings-wrapper", embeddings),
    )
    monkeypatch.setattr(evaluator, "evaluate", fake_evaluate)

    result = evaluator.run_ragas_evaluation(
        [
            {
                "user_input": "Afirmação",
                "response": "Resposta",
                "retrieved_contexts": ["Contexto"],
                "reference": "Referência",
            }
        ],
        "qwen2.5:3b",
    )

    assert calls["chat_ollama"]["model"] == "qwen2.5:3b"
    assert calls["chat_ollama"]["format"] == "json"
    assert {
        metric.name for metric in calls["evaluate"]["metrics"]
    } == set(evaluator.METRIC_COLUMNS)
    assert set(result.columns) == set(evaluator.METRIC_COLUMNS)


def test_bot_batch_skips_failed_row_and_continues(monkeypatch):
    dataframe = pd.DataFrame(
        [
            {
                "afirmacao": "Afirmação que falha.",
                "VEREDITO": "VERDADEIRO",
                "resposta_de_referencia": "Referência 1.",
            },
            {
                "afirmacao": "Afirmação que passa.",
                "VEREDITO": "FALSO",
                "resposta_de_referencia": "Referência 2.",
            },
        ]
    )
    from fato_unb.bots import telegram_bot

    def fake_verifier(assertion, *, raise_on_error):
        assert raise_on_error is True
        return (
            VereditoJSON(
                veredito=VereditoType.INCONCLUSIVO,
                justificativa="Ainda inconclusivo.",
                confianca=0.4,
                afirmacao_analisada=assertion,
            ),
            ["Contexto recuperado."],
        )

    def fake_run_ragas(records, judge_model_name):
        assertion = records[0]["user_input"]
        if assertion == "Afirmação que falha.":
            raise ValueError("Falha simulada na avaliação RAGAS.")
        return pd.DataFrame(
            [
                {
                    "faithfulness": 0.8,
                    "context_recall": 0.7,
                    "context_precision": 0.6,
                    "answer_correctness": 0.5,
                }
            ]
        )

    monkeypatch.setattr(
        telegram_bot,
        "verificar_afirmacao_provisoria_com_contextos",
        fake_verifier,
    )
    monkeypatch.setattr(evaluator, "run_ragas_evaluation", fake_run_ragas)

    output, summary = evaluator.evaluate_bot_testset(dataframe, "qwen2.5:3b")

    assert output["afirmacao"].tolist() == ["Afirmação que passa."]
    assert summary["fidelidade_ao_contexto"] == 0.8
    assert summary["acuracia_do_veredito"] == 0.0


def test_local_batch_skips_failed_row_and_continues(monkeypatch):
    dataframe = pd.DataFrame(
        [
            {
                "pergunta": "Pergunta que falha?",
                "contextos_de_referencia": json.dumps(["Contexto 1."]),
                "resposta_de_referencia": "Referência 1.",
            },
            {
                "pergunta": "Pergunta que passa?",
                "contextos_de_referencia": json.dumps(["Contexto 2."]),
                "resposta_de_referencia": "Referência 2.",
            },
        ]
    )
    monkeypatch.setattr(
        evaluator,
        "ChatOllama",
        lambda **kwargs: "fake-chat-model",
    )
    monkeypatch.setattr(
        evaluator,
        "generate_answer",
        lambda model, question, contexts: f"Resposta para {question}",
    )

    def fake_run_ragas(records, judge_model_name):
        question = records[0]["user_input"]
        if question == "Pergunta que falha?":
            raise ValueError("Falha simulada na avaliação RAGAS.")
        return pd.DataFrame(
            [
                {
                    "faithfulness": 0.8,
                    "context_recall": 0.7,
                    "context_precision": 0.6,
                    "answer_correctness": 0.5,
                }
            ]
        )

    monkeypatch.setattr(evaluator, "run_ragas_evaluation", fake_run_ragas)

    output, summary = evaluator.evaluate_testset(
        dataframe,
        answer_model_name="qwen2.5:7b",
        judge_model_name="qwen2.5:3b",
    )

    assert output["pergunta"].tolist() == ["Pergunta que passa?"]
    assert output["resposta_do_modelo"].tolist() == [
        "Resposta para Pergunta que passa?"
    ]
    assert summary["correcao_da_resposta"] == 0.5
