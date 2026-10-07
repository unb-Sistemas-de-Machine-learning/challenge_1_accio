import asyncio

from langchain_core.exceptions import OutputParserException
from langchain_core.documents import Document
from ragas.testset.graph import Node, NodeType
from ragas.testset.transforms import Parallel
from ragas.testset.transforms.extractors.llm_based import NERExtractor, SummaryExtractor
from ragas.testset.transforms.relationship_builders import OverlapScoreBuilder

from fato_unb.evaluation import generateTestset as generate_testset_module
from fato_unb.evaluation.generateTestset import (
    AfirmacaoGerada,
    RESPOSTA_FALSA,
    RESPOSTA_VERDADEIRA,
    SummaryExtractorResiliente,
    VEREDITO_CONFIRMADO,
    VEREDITO_FALSO,
    gerar_afirmacao,
    localizar_fontes,
    preparar_amostra_bot,
    preparar_transforms,
    tentar_gerar_afirmacao,
)


def test_summary_extractor_uses_plain_text_when_model_breaks_json_format():
    class PromptComFalha:
        async def generate(self, llm, data):
            raise OutputParserException(
                "Saída não é JSON",
                llm_output="Resumo em texto simples, mas válido.",
            )

    extractor = SummaryExtractorResiliente(
        llm=object(),
        prompt=PromptComFalha(),
    )
    node = Node(
        type=NodeType.DOCUMENT,
        properties={"page_content": "Conteúdo de teste."},
    )

    resultado = asyncio.run(extractor.extract(node))

    assert resultado == (
        "summary",
        "Resumo em texto simples, mas válido.",
    )


def test_summary_extractor_relanca_falha_sem_resposta_textual():
    class PromptComFalha:
        async def generate(self, llm, data):
            raise OutputParserException(
                "Saída não é JSON",
                llm_output=" ",
            )

    extractor = SummaryExtractorResiliente(
        llm=object(),
        prompt=PromptComFalha(),
    )
    node = Node(
        type=NodeType.DOCUMENT,
        properties={"page_content": "Conteúdo de teste."},
    )

    try:
        asyncio.run(extractor.extract(node))
    except OutputParserException:
        pass
    else:
        raise AssertionError("A falha sem texto bruto deveria ser relançada.")


def test_preparar_transforms_preserva_extratores_necessarios_para_sintese():
    resumo = SummaryExtractor(llm=object())
    named_entities = NERExtractor(llm=object())
    overlap_builder = OverlapScoreBuilder()
    parallel = Parallel(named_entities, overlap_builder)

    transforms = preparar_transforms([resumo, parallel])

    assert isinstance(transforms[0], SummaryExtractorResiliente)
    assert len(transforms[1].transformations) == 2
    assert isinstance(transforms[1].transformations[0], NERExtractor)
    assert isinstance(transforms[1].transformations[1], OverlapScoreBuilder)


def test_preparar_amostra_bot_cria_afirmacao_rotulada_com_fonte_oficial():
    documento = Document(
        page_content="Título: Calendário UnB\n\nA matrícula começa em março.",
        metadata={
            "doc_id": "calendario-1",
            "title": "Calendário UnB",
            "url": "https://deG.unb.br/calendario#content",
            "source": "DEG",
            "published_at": "2026-01-01",
        },
    )
    contextos = [
        "<1-hop>\n\nT\uFFFDtulo: Calendário UnB\n\nA matrícula começa em março."
    ]
    fontes_por_contexto = localizar_fontes(contextos, [documento])
    linha = {
        "user_input": "Quando começa a matrícula?",
        "reference": "A matrícula começa em março.",
        "reference_contexts": contextos,
        "fontes_dos_contextos": fontes_por_contexto,
    }

    amostra = preparar_amostra_bot(linha)

    assert amostra is not None
    assert amostra["user_input"] == "A matrícula começa em março."
    assert amostra["pergunta_original"] == "Quando começa a matrícula?"
    assert amostra["afirmacao"] == "A matrícula começa em março."
    assert amostra["veredito_de_referencia"] == VEREDITO_CONFIRMADO
    assert amostra["requer_revisao_humana"] is True
    assert amostra["fontes_de_referencia"] == fontes_por_contexto[0]["fontes"]


def test_preparar_amostra_bot_nao_rotula_sem_fonte_oficial():
    linha = {
        "user_input": "Afirmação original?",
        "reference": "Afirmação factual.",
        "fontes_dos_contextos": [
            {
                "contexto": "<1-hop>",
                "fontes": [
                    {
                        "titulo": "Fonte externa",
                        "url": "https://example.org/fato",
                    }
                ],
            }
        ],
    }

    assert preparar_amostra_bot(linha) is None


def test_gerar_afirmacao_retorna_apenas_fontes_dos_contextos_citados(monkeypatch):
    contextos = [
        "<1-hop>\n\nO edital informa matrícula entre 10 e 15 de março.",
        "<2-hop>\n\nUma notícia geral sobre a universidade.",
    ]
    fontes_por_contexto = [
        {
            "contexto": "<1-hop>",
            "fontes": [
                {
                    "titulo": "Edital DEG",
                    "url": "https://deg.unb.br/edital",
                }
            ],
        },
        {
            "contexto": "<2-hop>",
            "fontes": [
                {
                    "titulo": "Notícia externa",
                    "url": "https://example.org/noticia",
                }
            ],
        },
    ]

    monkeypatch.setattr(
        generate_testset_module.random,
        "randint",
        lambda minimo, maximo: 1,
    )

    class FakeRunnable:
        def invoke(self, messages):
            prompt = messages[0].content
            assert "Quando ocorre a matrícula?" in prompt
            assert "afirmação verdadeira" in prompt
            assert "CONTEXTO 1" in prompt
            assert "CONTEXTO 2" not in prompt
            return AfirmacaoGerada(
                afirmacao="A matrícula ocorre entre 10 e 15 de março.",
                indices_contextos=[1],
            )

    class FakeLLM:
        def with_structured_output(self, schema):
            assert schema is AfirmacaoGerada
            return FakeRunnable()

    linha = {
        "user_input": "Quando ocorre a matrícula?",
        "reference_contexts": contextos,
        "fontes_dos_contextos": fontes_por_contexto,
    }

    resultado = gerar_afirmacao(linha, FakeLLM())

    assert resultado == {
        "afirmacao": "A matrícula ocorre entre 10 e 15 de março.",
        "fontes": [
            {
                "titulo": "Edital DEG",
                "url": "https://deg.unb.br/edital",
            }
        ],
        "numero_sorteado": 1,
        "resposta": RESPOSTA_VERDADEIRA,
    }
    linha["afirmacao"] = resultado["afirmacao"]
    linha["fontes_da_afirmacao"] = resultado["fontes"]
    amostra_bot = preparar_amostra_bot(linha)

    assert amostra_bot is not None
    assert amostra_bot["user_input"] == resultado["afirmacao"]
    assert amostra_bot["reference"] == resultado["afirmacao"]
    assert amostra_bot["fontes_de_referencia"] == resultado["fontes"]
    assert amostra_bot["resposta"] == RESPOSTA_VERDADEIRA
    assert amostra_bot["veredito_de_referencia"] == VEREDITO_CONFIRMADO
    assert "fontes_da_afirmacao" not in amostra_bot


def test_gerar_afirmacao_sorteia_falsa_e_rotula_resposta(monkeypatch):
    contextos = ["<1-hop>\n\nA matrícula começa em março."]
    fontes_por_contexto = [
        {
            "contexto": "<1-hop>",
            "fontes": [
                {
                    "titulo": "Calendário UnB",
                    "url": "https://unb.br/calendario",
                }
            ],
        }
    ]
    sorteios = []

    def sortear(minimo, maximo):
        sorteios.append((minimo, maximo))
        return 0

    monkeypatch.setattr(generate_testset_module.random, "randint", sortear)

    class FakeRunnable:
        def invoke(self, messages):
            assert "afirmação falsa" in messages[0].content
            assert "diretamente contradita" in messages[0].content
            return AfirmacaoGerada(
                afirmacao="A matrícula começa em setembro.",
                indices_contextos=[1],
            )

    class FakeLLM:
        def with_structured_output(self, schema):
            assert schema is AfirmacaoGerada
            return FakeRunnable()

    resultado = gerar_afirmacao(
        {
            "user_input": "Quando começa a matrícula?",
            "reference_contexts": contextos,
            "fontes_dos_contextos": fontes_por_contexto,
        },
        FakeLLM(),
    )
    amostra_bot = preparar_amostra_bot(
        {
            "user_input": "Quando começa a matrícula?",
            "afirmacao": resultado["afirmacao"],
            "fontes_dos_contextos": fontes_por_contexto,
            "fontes_da_afirmacao": resultado["fontes"],
            "resposta": resultado["resposta"],
        }
    )

    assert sorteios == [(0, 1)]
    assert resultado["numero_sorteado"] == 0
    assert resultado["resposta"] == RESPOSTA_FALSA
    assert amostra_bot["resposta"] == RESPOSTA_FALSA
    assert amostra_bot["veredito_de_referencia"] == VEREDITO_FALSO
    assert amostra_bot["fontes_de_referencia"] == fontes_por_contexto[0]["fontes"]


def test_gerar_afirmacao_rejeita_indice_de_contexto_nao_disponivel():
    class FakeRunnable:
        def invoke(self, messages):
            return AfirmacaoGerada(
                afirmacao="Uma afirmação.",
                indices_contextos=[2],
            )

    class FakeLLM:
        def with_structured_output(self, schema):
            return FakeRunnable()

    linha = {
        "user_input": "Pergunta?",
        "reference_contexts": ["<1-hop>\n\nFato oficial."],
        "fontes_dos_contextos": [
            {
                "contexto": "<1-hop>",
                "fontes": [
                    {
                        "titulo": "Fonte oficial",
                        "url": "https://unb.br/fato",
                    }
                ],
            }
        ],
    }

    try:
        gerar_afirmacao(linha, FakeLLM())
    except ValueError as exc:
        assert "contexto inexistente" in str(exc)
    else:
        raise AssertionError("A afirmação sem uma fonte válida foi aceita.")


def test_tentar_gerar_afirmacao_descarta_indice_invalido_sem_interromper():
    class FakeRunnable:
        def invoke(self, messages):
            return AfirmacaoGerada(
                afirmacao="Uma afirmação.",
                indices_contextos=[2],
            )

    class FakeLLM:
        def with_structured_output(self, schema):
            return FakeRunnable()

    linha = {
        "user_input": "Pergunta?",
        "reference_contexts": ["<1-hop>\n\nFato oficial."],
        "fontes_dos_contextos": [
            {
                "contexto": "<1-hop>",
                "fontes": [
                    {
                        "titulo": "Fonte oficial",
                        "url": "https://unb.br/fato",
                    }
                ],
            }
        ],
    }

    assert tentar_gerar_afirmacao(linha, FakeLLM()) is None


def test_localizar_fontes_deduplica_urls_com_fragmentos():
    documentos = [
        Document(
            page_content="Título: Aviso UnB\n\nConteúdo.",
            metadata={
                "doc_id": "aviso-1",
                "title": "Aviso UnB",
                "url": "https://unb.br/aviso",
                "source": "UnB",
            },
        ),
        Document(
            page_content="Título: Aviso UnB\n\nConteúdo.",
            metadata={
                "doc_id": "aviso-2",
                "title": "Aviso UnB",
                "url": "https://unb.br/aviso#main",
                "source": "UnB",
            },
        ),
    ]

    fontes = localizar_fontes(
        ["<1-hop>\n\nTítulo: Aviso UnB\n\nTrecho."],
        documentos,
    )

    assert len(fontes) == 1
    assert len(fontes[0]["fontes"]) == 1
