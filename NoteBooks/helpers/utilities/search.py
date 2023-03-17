import os
import glob
import pandas as pd

__all__ = [
    "search_track_files",
    "search_list"
]


def search_track_files(df_tracks, search_string):
    """Search for track files with a given string in the file name."""

    matching_tracks = df_tracks[df_tracks['file_name'].str.contains(search_string)]

    if matching_tracks.empty:
        return matching_tracks, False
    else:
        return matching_tracks, True


def search_list(search_term, lists):
    """Search for a term in a list."""

    matching_items = [item for item in lists if search_term in item]

    return matching_items
