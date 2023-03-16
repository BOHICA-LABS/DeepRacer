import glob
import os
import pandas as pd
import numpy as np

__all__ = [
    "load_available_tracks",
    "load_track_waypoints",
]


def load_available_tracks(data_dir):
    """Load available track files from the data directory."""

    available_track_files = glob.glob(f"{data_dir}/*.npy")
    available_tracks = []

    if len(available_track_files) > 0:
        for file_path in available_track_files:
            file_name = os.path.basename(file_path).split('.npy')[0]
            available_tracks.append({'file_name': file_name, 'file_path': file_path})

        df_available_tracks = pd.DataFrame(available_tracks, columns=['file_name', 'file_path'])
        df_available_tracks_sorted = df_available_tracks.sort_values(by=['file_name'])
        df_available_tracks_sorted = df_available_tracks_sorted.reset_index(drop=True)
    else:
        df_available_tracks_sorted = pd.DataFrame({'file_name': [], 'file_path': []})

    return df_available_tracks_sorted


def load_track_waypoints(track_file):
    """Load track waypoints from a track file."""

    track_waypoints = np.load(track_file)
    return track_waypoints
