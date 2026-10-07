from datetime import UTC, datetime

import pytest
from pydantic import HttpUrl

from fato_unb.ingestion.models import RawDocument, SourceType


@pytest.fixture
def mock_noticia_ru():
    return RawDocument(
        title="Funcionamento do RU no Feriado",
        content="A Diretoria do RU informa que nos dias 1 e 2 de maio os refeitórios do Darcy Ribeiro abrirão apenas no almoço, das 11h30 às 14h. O jantar estará suspenso.",
        url=HttpUrl("https://noticias.unb.br/76-institucional/5500-ru-feriado"),
        source="UnB Notícias",
        source_type=SourceType.RSS_NEWS,
        published_at=datetime(2026, 4, 28, 10, 0, tzinfo=UTC),
        semester_ref="2026/1",
    )


@pytest.fixture
def mock_edital_extenso():
    texto_base = (
        "Art. 1º O período de confirmação de matrícula no SIGAA ocorre entre os dias 10 e 15 de março. "
        "Art. 2º O trancamento geral justificado pode ser solicitado até o encerramento do semestre. "
        "Art. 3º É vedada a matrícula em disciplinas com sobreposição de horário no mesmo campus. "
    )
    return RawDocument(
        title="Circular Normativa DEG nº 02/2026",
        content=(
            texto_base * 40
        ),  # Cria um documento longo para acionar múltiplos chunks
        url=HttpUrl("https://deg.unb.br/atos/circular-02-2026"),
        source="DEG",
        source_type=SourceType.HTML_PAGE,
        published_at=datetime(2026, 2, 15, 8, 30, tzinfo=UTC),
        semester_ref="2026/1",
    )
