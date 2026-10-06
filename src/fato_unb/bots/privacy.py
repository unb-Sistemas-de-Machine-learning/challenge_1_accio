import logging
import re

CPF_FORMATADO = re.compile(r"(?<!\d)\d{3}\.\d{3}\.\d{3}-\d{2}(?!\d)")
TELEFONE_FORMATADO = re.compile(r"(?<!\d)(?:\+55[\s-]?)?\(?\d{2}\)?[\s-]?9?\d{4}-\d{4}(?!\d)")
NUMERO_SEM_PONTUACAO = re.compile(r"(?<!\d)\d{8,11}(?!\d)")
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")


def _rotular_numero_sem_pontuacao(match: re.Match) -> str:
    tamanho = len(match.group())
    if tamanho == 11:
        return "[CPF]"
    if tamanho == 9:
        return "[MATRICULA]"
    return "[TELEFONE]"


def scrub(texto: str) -> str:
    if not texto:
        return texto
    texto = CPF_FORMATADO.sub("[CPF]", texto)
    texto = TELEFONE_FORMATADO.sub("[TELEFONE]", texto)
    texto = NUMERO_SEM_PONTUACAO.sub(_rotular_numero_sem_pontuacao, texto)
    texto = EMAIL.sub("[EMAIL]", texto)
    return texto


class FiltroPII(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = scrub(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: scrub(v) if isinstance(v, str) else v for k, v in record.args.items()}
            else:
                record.args = tuple(scrub(a) if isinstance(a, str) else a for a in record.args)
        return True
