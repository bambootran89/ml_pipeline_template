from mlproject.src.data.dataloader import create_windows


def test_create_windows_shape():
    import pandas as pd

    idx = pd.date_range("2020-01-01", periods=50, freq="H")
    df = pd.DataFrame({"value": range(50)}, index=idx)
    X, y = create_windows(
        df.rename(columns={"value": "HUFL"}), "HUFL", input_chunk=8, output_chunk=2
    )
    assert X.ndim == 3
