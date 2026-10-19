"""
Unit tests for data parsing and text preprocessing modules.
"""

import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocess import clean_text, parse_pubmed_rct


def test_clean_text():
    sample = "  To investigate the   efficacy of 6 weeks  of daily   prednisolone .\n "
    cleaned = clean_text(sample)
    assert cleaned == "To investigate the efficacy of 6 weeks of daily prednisolone ."


def test_parse_pubmed_rct_single_and_multi_abstract():
    content = (
        "###1001\n"
        "BACKGROUND\tThis is the background sentence.\n"
        "OBJECTIVE\tThis is the objective sentence.\n"
        "METHODS\tA randomized trial was conducted.\n"
        "RESULTS\tSignificant improvement was observed (p < 0.05).\n"
        "CONCLUSIONS\tThe treatment is effective.\n"
        "\n"
        "###1002\n"
        "BACKGROUND\tAnother background sentence.\n"
        "METHODS\tMethods for abstract two.\n"
        "RESULTS\tResults for abstract two.\n"
        "CONCLUSIONS\tConclusions for abstract two.\n"
    )

    with tempfile.NamedTemporaryFile("w+", delete=False, encoding="utf-8", suffix=".txt") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        data = parse_pubmed_rct(tmp_path)
        assert len(data) == 9

        # Handle list vs DataFrame
        if isinstance(data, list):
            abstract_ids = list(dict.fromkeys(r["abstract_id"] for r in data))
            labels = list(dict.fromkeys(r["label"] for r in data))
            assert abstract_ids == ["1001", "1002"]
            assert labels == ["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"]

            abs_1001 = [r for r in data if r["abstract_id"] == "1001"]
            assert len(abs_1001) == 5
            assert [r["line_number"] for r in abs_1001] == [0, 1, 2, 3, 4]
            assert [r["total_lines"] for r in abs_1001] == [5, 5, 5, 5, 5]
            assert abs_1001[0]["relative_position"] == 0.0
            assert abs_1001[4]["relative_position"] == 0.8
        else:
            assert list(data["abstract_id"].unique()) == ["1001", "1002"]
            assert list(data["label"].unique()) == ["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"]

            df_1001 = data[data["abstract_id"] == "1001"]
            assert len(df_1001) == 5
            assert list(df_1001["line_number"]) == [0, 1, 2, 3, 4]
            assert list(df_1001["total_lines"]) == [5, 5, 5, 5, 5]
            assert df_1001.iloc[0]["relative_position"] == 0.0
            assert df_1001.iloc[4]["relative_position"] == 0.8
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_parse_pubmed_rct_no_trailing_newline():
    content = (
        "###2001\n"
        "BACKGROUND\tSentence without trailing newline.\n"
        "CONCLUSIONS\tFinal line at EOF."
    )

    with tempfile.NamedTemporaryFile("w+", delete=False, encoding="utf-8", suffix=".txt") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        data = parse_pubmed_rct(tmp_path)
        assert len(data) == 2
        last_item = data[-1] if isinstance(data, list) else data.iloc[-1].to_dict()
        assert last_item["text"] == "Final line at EOF."
        assert last_item["line_number"] == 1
        assert last_item["total_lines"] == 2
    finally:
        Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    test_clean_text()
    test_parse_pubmed_rct_single_and_multi_abstract()
    test_parse_pubmed_rct_no_trailing_newline()
    print("[OK] All data parser unit tests passed successfully!")
