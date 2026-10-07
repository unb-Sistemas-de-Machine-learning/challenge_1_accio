import pytest

from fato_unb.rag.model_cache import carregar_modelo, materializar_symlinks


def _cache_com_symlink(tmp_path):
    snapshot = tmp_path / "models--org--modelo" / "snapshots" / "abc"
    blobs = tmp_path / "models--org--modelo" / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    (blobs / "hash1").write_bytes(b"dados do modelo")
    (snapshot / "model.onnx_data").symlink_to("../../blobs/hash1")
    return snapshot, blobs


def test_materializar_troca_symlink_por_arquivo_real(tmp_path):
    snapshot, blobs = _cache_com_symlink(tmp_path)

    assert materializar_symlinks(tmp_path) == 1

    arquivo = snapshot / "model.onnx_data"
    assert not arquivo.is_symlink()
    assert arquivo.read_bytes() == b"dados do modelo"
    assert (blobs / "hash1").exists()  # o blob original não é tocado


def test_materializar_sem_symlinks_nao_faz_nada(tmp_path):
    assert materializar_symlinks(tmp_path) == 0


def test_carregar_modelo_tenta_de_novo_apos_erro_do_onnxruntime(tmp_path, monkeypatch):
    snapshot, _ = _cache_com_symlink(tmp_path)
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(tmp_path))
    chamadas = []

    def fabrica():
        chamadas.append(1)
        if (snapshot / "model.onnx_data").is_symlink():
            raise RuntimeError("External data path escapes model directory")
        return "modelo"

    assert carregar_modelo(fabrica) == "modelo"
    assert len(chamadas) == 2


def test_carregar_modelo_repassa_outros_erros(tmp_path, monkeypatch):
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(tmp_path))

    def fabrica():
        raise ValueError("outro problema")

    with pytest.raises(ValueError, match="outro problema"):
        carregar_modelo(fabrica)


def test_carregar_modelo_repassa_erro_se_nao_havia_symlink(tmp_path, monkeypatch):
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(tmp_path))

    def fabrica():
        raise RuntimeError("External data path escapes model directory")

    with pytest.raises(RuntimeError, match="External data path"):
        carregar_modelo(fabrica)
