import pandas as pd
from .search import search_track_files


def test_search_track_files():
    df_test = pd.DataFrame({'file_name': ['track1', 'track2', 'example1', 'example2'],
                            'file_path': ['path1', 'path2', 'path3', 'path4']})

    # Test that a match is found
    result, _isMatch = search_track_files(df_test, "example")
    assert len(result) == 2
    assert result.iloc[0]['file_name'] == 'example1'
    assert result.iloc[0]['file_path'] == 'path3'
    assert _isMatch is True

    # Test that no match is found
    result, _isMatch = search_track_files(df_test, "no_match")
    assert isinstance(result, pd.DataFrame)  # assert that result is a DataFrame
    assert result.empty  # assert that result is empty
    assert _isMatch is False
