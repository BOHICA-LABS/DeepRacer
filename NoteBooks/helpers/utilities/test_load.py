import os
import shutil
import numpy as np
from shapely.geometry import LineString, Polygon

import pandas as pd
from .load import load_available_tracks


def test_load_available_tracks():
    # Create a temporary directory with test files
    data_dir = 'tmp_test_dir'
    os.makedirs(os.path.join(data_dir, 'tracks'), exist_ok=True)
    open(os.path.join(data_dir, 'tracks', 'file1.npy'), 'a').close()
    open(os.path.join(data_dir, 'tracks', 'file2.npy'), 'a').close()
    open(os.path.join(data_dir, 'tracks', 'file3.npy'), 'a').close()

    # Test that function returns a sorted DataFrame
    expected_df = pd.DataFrame({'file_name': ['file1', 'file2', 'file3'],
                                'file_path': [os.path.join(data_dir, 'tracks', 'file1.npy'),
                                              os.path.join(data_dir, 'tracks', 'file2.npy'),
                                              os.path.join(data_dir, 'tracks', 'file3.npy')]})
    result_df = load_available_tracks(f'{data_dir}/tracks')
    pd.testing.assert_frame_equal(result_df, expected_df)

    # Test with an empty directory
    empty_data_dir = 'tmp_empty_dir'
    os.makedirs(os.path.join(empty_data_dir, 'tracks'), exist_ok=True)
    result_df_empty = load_available_tracks(f'{empty_data_dir}/tracks')
    expected_df_empty = pd.DataFrame({'file_name': [], 'file_path': []})
    pd.testing.assert_frame_equal(result_df_empty, expected_df_empty)

    # Clean up temporary directories
    shutil.rmtree(data_dir, ignore_errors=True)
    shutil.rmtree(empty_data_dir, ignore_errors=True)