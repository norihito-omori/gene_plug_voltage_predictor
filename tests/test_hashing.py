from pathlib import Path

from gene_plug_voltage_predictor.utils.hashing import sha256_of_file


def test_sha256_of_file_returns_hex_digest_with_prefix(tmp_path: Path) -> None:
    f = tmp_path / "sample.csv"
    f.write_bytes(b"col1,col2\n1,2\n")
    result = sha256_of_file(f)
    assert result.startswith("sha256:")
    assert len(result) == len("sha256:") + 64


def test_sha256_of_file_is_deterministic(tmp_path: Path) -> None:
    f = tmp_path / "sample.csv"
    f.write_bytes(b"abc")
    assert sha256_of_file(f) == sha256_of_file(f)


def test_sha256_of_file_differs_for_different_content(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    a.write_bytes(b"abc")
    b.write_bytes(b"abd")
    assert sha256_of_file(a) != sha256_of_file(b)
