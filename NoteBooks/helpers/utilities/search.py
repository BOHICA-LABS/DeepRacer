import os
import glob
import pandas as pd

__all__ = [
    "search_track_files"
]


def search_track_files(df_tracks, search_string):
    """Search for track files with a given string in the file name."""

    matching_tracks = df_tracks[df_tracks['file_name'].str.contains(search_string)]

    if matching_tracks.empty:
        return matching_tracks, False
    else:
        return matching_tracks, True
