import json
import httpx
import logging
import pandas as pd
from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import OllamaEmbeddings, ChatOllama

from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from ragas.testset.transforms import Parallel
from ragas.testset.transforms.default import default_transforms
from ragas.testset.transforms.extractors.llm_based import (
    TextWithExtractionLimit,
    ThemesExtractor,
)
from ragas.testset.transforms.filters import CustomNodeFilter
from ragas.testset.transforms.splitters import HeadlineSplitter

logger = logging.getLogger(__name__)

# para fazer a resposta do CSV vim em portugues e quais colunas criar
INSTRUCAO_PORTUGUES = (
    "Responda sempre em português brasileiro. Todo texto natural gerado, "
    "incluindo resumos, temas, personas, perguntas e respostas, deve estar em "
    "português brasileiro. Preserve nomes próprios, siglas, URLs e o conteúdo "
    "factual das fontes. Siga também o formato estruturado solicitado. "
    "Quando o formato solicitado for JSON, retorne apenas JSON válido: não "
    "adicione comentários, como // ou /* ... */, nem texto fora do JSON."
)


class ChatOllamaPortugues(ChatOllama):
    def _convert_messages_to_ollama_messages(self, messages):
        return super()._convert_messages_to_ollama_messages(
            [SystemMessage(content=INSTRUCAO_PORTUGUES), *messages]
        )


class ThemesExtractorResiliente(ThemesExtractor):
    async def extract(self, node):
        node_text = node.get_property("page_content")
        if node_text is None:
            return self.property_name, []

        chunks = self.split_text_by_token_limit(node_text, self.max_token_limit)
        themes = []
        for indice, chunk in enumerate(chunks, start=1):
            try:
                result = await self.prompt.generate(
                    self.llm,
                    data=TextWithExtractionLimit(
                        text=chunk,
                        max_num=self.max_num_themes,
                    ),
                )
            except OutputParserException as exc:
                logger.warning(
                    "Ignorando trecho %d/%d do nó %s na extração de temas "
                    "após %s: %s",
                    indice,
                    len(chunks),
                    node.id,
                    type(exc).__name__,
                    exc,
                )
                continue
            themes.extend(result.output)

        return self.property_name, themes


def _tornar_transforms_resilientes(transform):
    if isinstance(transform, Parallel):
        transform.transformations = [
            _tornar_transforms_resilientes(child)
            for child in transform.transformations
        ]
        return transform

    if not isinstance(transform, ThemesExtractor) or isinstance(
        transform, ThemesExtractorResiliente
    ):
        return transform

    return ThemesExtractorResiliente(
        name=transform.name,
        filter_nodes=transform.filter_nodes,
        llm=transform.llm,
        merge_if_possible=transform.merge_if_possible,
        max_token_limit=transform.max_token_limit,
        tokenizer=transform.tokenizer,
        property_name=transform.property_name,
        prompt=transform.prompt,
        max_num_themes=transform.max_num_themes,
    )


def gerar_afirmacao(pergunta, contextos, llm, veredito):
    if veredito not in ("VERDADEIRO", "FALSO"):
        raise ValueError(f"Veredito inválido para gerar afirmação: {veredito}")
    if not isinstance(pergunta, str) or not pergunta.strip():
        raise ValueError("Não é possível gerar afirmação sem uma pergunta.")
    if not contextos:
        raise ValueError(
            "Não é possível gerar afirmação sem contextos de referência."
        )

    contextos_texto = "\n\n".join(
        contexto for contexto in contextos if isinstance(contexto, str)
    )
    if not contextos_texto.strip():
        raise ValueError("Os contextos de referência não contêm texto.")

    instrucao_veredito = (
        "Crie uma afirmação verdadeira, sustentada diretamente pelos contextos."
        if veredito == "VERDADEIRO"
        else (
            "Crie uma afirmação falsa, contradita diretamente por pelo menos "
            "um dos contextos. Para isso, altere ou negue um fato explícito "
            "nos contextos; não invente fatos externos."
        )
    )
    resultado = llm.invoke(
        [
            HumanMessage(
                content=(
                    f"{instrucao_veredito} Transforme a pergunta em uma "
                    "única frase declarativa curta. Não inclua explicações, "
                    "rótulos ou aspas. Retorne apenas a afirmação.\n\n"
                    f"Pergunta: {pergunta}\n\n"
                    f"Contextos de referência:\n{contextos_texto}"
                )
            )
        ]
    )
    if not isinstance(resultado.content, str) or not resultado.content.strip():
        raise ValueError("O modelo retornou uma afirmação vazia ou inválida.")
    return resultado.content.strip()


def distribuir_vereditos(total_amostras):
    if total_amostras <= 0:
        raise ValueError(
            "A quantidade de amostras precisa ser maior que zero."
        )
    print(f"total de amostras: ${total_amostras}, verdadeiras = ${(total_amostras + 1) // 2}, falsas = ${total_amostras // 2}")
    quantidade_verdadeiras = (total_amostras + 1) // 2
    quantidade_falsas = total_amostras // 2
    return (
        ["VERDADEIRO"] * quantidade_verdadeiras
        + ["FALSO"] * quantidade_falsas
    )


def localizar_fontes(contextos, documentos):
    fontes_por_contexto = []
    for contexto in contextos or []:
        texto_contexto = contexto.split("\n\n", 1)[-1]
        fontes = []
        for documento in documentos:
            if texto_contexto and texto_contexto in documento.page_content:
                fontes.append(
                    {
                        "id_documento": documento.metadata.get("doc_id"),
                        "titulo": documento.metadata.get("title"),
                        "url": documento.metadata.get("url"),
                        "origem": documento.metadata.get("source"),
                        "data_publicacao": documento.metadata.get("published_at"),
                    }
                )
        fontes_por_contexto.append(
            {
                "contexto": contexto.split("\n", 1)[0],
                "fontes": fontes,
            }
        )
    return fontes_por_contexto


def serializar_valores(valor):
    if isinstance(valor, (list, dict)):
        return json.dumps(valor, ensure_ascii=False)
    return valor


ESTILOS_EM_PORTUGUES = {
    "MISSPELLED": "com erros ortográficos",
    "PERFECT_GRAMMAR": "gramática correta",
    "POOR_GRAMMAR": "gramática informal",
    "WEB_SEARCH_LIKE": "estilo de busca na web",
}

COMPRIMENTOS_EM_PORTUGUES = {
    "LONG": "longa",
    "MEDIUM": "média",
    "SHORT": "curta",
}

SINTETIZADORES_EM_PORTUGUES = {
    "single_hop_specific_query_synthesizer": "pergunta específica de etapa única",
    "multi_hop_abstract_query_synthesizer": "pergunta abstrata de múltiplas etapas",
    "multi_hop_specific_query_synthesizer": "pergunta específica de múltiplas etapas",
}


# ---------------------------------------------------------
# 1. Carregamento e Deduplicação dos Dados
# ---------------------------------------------------------
input_filename = "dados.txt"
documents = []
seen_doc_ids = set()

with open(input_filename, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        
        data = json.loads(line)
        doc_id = data.get("doc_id")
        
        if doc_id and doc_id in seen_doc_ids: #não adiciona com IDs repetidos
            continue
        if doc_id:
            seen_doc_ids.add(doc_id)
        
        content = f"Título: {data.get('title', '')}\n\n{data.get('content', '')}"
        
        doc = Document(
            page_content=content,
            metadata={
                "title": data.get("title"),
                "url": data.get("url"),
                "source": data.get("source"),
                "published_at": data.get("published_at"),
                "doc_id": doc_id
            }
        )
        documents.append(doc)

print(f"Total de documentos únicos carregados: {len(documents)}")

# ---------------------------------------------------------
# 2. Configuração dos modelos locais via Ollama
# ---------------------------------------------------------
ollama_async_client_kwargs = {
    "limits": httpx.Limits(max_keepalive_connections=0),
}

generator_chat_model = ChatOllamaPortugues(
    model="qwen2.5:7b",
    temperature=0.0,    # temperatura 0 para reduzir variancia nas respostas
    async_client_kwargs=ollama_async_client_kwargs,
)
generator_llm = LangchainLLMWrapper(generator_chat_model)
transforms_llm = LangchainLLMWrapper(
    ChatOllamaPortugues(
        model="qwen2.5:3b",
        temperature=0.0,
        async_client_kwargs=ollama_async_client_kwargs,
    )
)

embeddings = LangchainEmbeddingsWrapper(
    OllamaEmbeddings(model="bge-m3", num_gpu=0) # gpu desativado pq estava retornando embedding = NaN
)

run_config = RunConfig(
    max_workers=1,      
    max_retries=3,      
    timeout=80
)

# ---------------------------------------------------------
# 3. Inicialização e Geração do Testset
# ---------------------------------------------------------
generator = TestsetGenerator(
    llm=generator_llm,
    embedding_model=embeddings,
    llm_context=INSTRUCAO_PORTUGUES,
)

transforms = default_transforms(
    documents=documents,
    llm=transforms_llm,
    embedding_model=embeddings,
)
transforms = [
    transform
    for transform in transforms
    if not isinstance(transform, CustomNodeFilter)
]
transforms = [
    _tornar_transforms_resilientes(transform)
    for transform in transforms
]
for transform in transforms:
    if isinstance(transform, HeadlineSplitter):
        transform.filter_nodes = lambda node: node.get_property("headlines") is not None

testset = generator.generate_with_langchain_docs(
    documents=documents,
    testset_size=30,
    transforms=transforms,
    run_config=run_config  
)
if not testset.samples:
    raise RuntimeError(
        "O Ragas concluiu a geração sem produzir amostras. "
        "Verifique se os documentos permitem criar cenários e se os transforms "
        "de extração de entidades e relações foram concluídos."
    )
vereditos = distribuir_vereditos(len(testset.samples))

# ---------------------------------------------------------
# 4. Exportação dos Resultados
# ---------------------------------------------------------
linhas_csv = []
for amostra, veredito in zip(testset.samples, vereditos, strict=True):
    linha = amostra.eval_sample.model_dump(exclude_none=True)
    linha["tipo_de_sintetizador"] = SINTETIZADORES_EM_PORTUGUES.get(
        amostra.synthesizer_name,
        amostra.synthesizer_name,
    )
    linha["identificador_tecnico_do_sintetizador"] = amostra.synthesizer_name
    if "query_style" in linha:
        linha["query_style"] = ESTILOS_EM_PORTUGUES.get(
            linha["query_style"],
            linha["query_style"],
        )
    if "query_length" in linha:
        linha["query_length"] = COMPRIMENTOS_EM_PORTUGUES.get(
            linha["query_length"],
            linha["query_length"],
        )
    linha["personas_geradas"] = [
        {
            "nome": persona.name,
            "descricao_do_papel": persona.role_description,
        }
        for persona in generator.persona_list or []
    ]
    linha["fontes_dos_contextos"] = localizar_fontes(
        linha.get("reference_contexts", []),
        documents,
    )
    linha["afirmacao"] = gerar_afirmacao(
        linha.get("user_input"),
        linha.get("reference_contexts", []),
        generator_chat_model,
        veredito,
    )
    linha["VEREDITO"] = veredito
    linhas_csv.append(linha)

df_testset = pd.DataFrame(linhas_csv)
df_testset = df_testset.rename(
    columns={
        "user_input": "pergunta",
        "retrieved_contexts": "contextos_recuperados",
        "reference_contexts": "contextos_de_referencia",
        "retrieved_context_ids": "ids_contextos_recuperados",
        "reference_context_ids": "ids_contextos_de_referencia",
        "response": "resposta_gerada",
        "multi_responses": "respostas_alternativas",
        "reference": "resposta_de_referencia",
        "rubrics": "criterios_de_avaliacao",
        "persona_name": "nome_da_persona",
        "query_style": "estilo_da_pergunta",
        "query_length": "comprimento_da_pergunta",
        "reference_tool_calls": "chamadas_de_ferramenta_de_referencia",
        "reference_topics": "topicos_de_referencia",
        "tipo_de_sintetizador": "tipo_de_sintetizador",
        "identificador_tecnico_do_sintetizador": "identificador_tecnico_do_sintetizador",
        "personas_geradas": "personas_geradas_na_execucao",
        "fontes_dos_contextos": "fontes_dos_contextos_de_referencia",
    }
)
df_testset = df_testset.map(serializar_valores)
df_testset.to_csv("ragas_testset_local.csv", index=False, encoding="utf-8-sig")

print("Testset gerado com sucesso! Salvo em 'ragas_testset_local.csv'.")