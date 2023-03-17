import json
import os
import pickle
import gzip
import uuid
import numpy as np
import shutil
from shapely.geometry import Polygon
from PIL import Image, ImageDraw
from datetime import datetime
from .search import search_list

__all__ = [
    "save_racepack",
    "load_package",
]


def dict_to_json(dict):
    """Convert a list of dictionaries to a JSON string."""

    return json.dumps(dict)


def create_folder(folder_path):
    """Create a folder if it doesn't exist."""

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return True


# TODO: Increase the verbosity of the vaildation function
def validate_race_package(race_package):
    """Validate a dictionary representing a race package."""

    required_keys = ['VERSIONS', 'GLOBAL_VARS', 'TRACK_VARS', 'RACE_LINE_VARS']
    for key in required_keys:
        if key not in race_package:
            return False
    return True


def create_compressed_pickle(file_path, data):
    """Create a compressed pickle file from a dictionary."""

    with gzip.open(file_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_package(file_path):
    """Load a compressed pickle file into a dictionary."""

    with gzip.open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def generate_uuid():
    """Generate a UUIDv4 and return it as a string."""
    uuid_v4 = uuid.uuid4()
    uuid_str = str(uuid_v4)
    return uuid_str


def write_file(file_path, data):
    """Write data to a file."""

    with open(file_path, 'w') as f:
        f.write(data)


def copy_file(src_path, dst_path):
    """Copy a file from src_path to dst_path."""

    # Create the destination directory if it doesn't exist
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    # Copy the file from src_path to dst_path
    shutil.copy(src_path, dst_path)


def write_optimals(optimal_path, optimal_data):
    """Write the optimal data to a file."""

    # Write to txt file
    with open(optimal_path, 'w') as f:
        f.write("[")
        for line in optimal_data:
            f.write("%s" % line)
            if line != optimal_data[-1]:
                f.write(",\n")
        f.write("]")


def save_polygon_as_png(poly, file_path, image_size=(200, 200), bg_color=(0, 0, 0, 0), line_color=(0, 0, 0)):
    """Save a Shapely Polygon object as a PNG image file."""

    # Create an image object
    image = Image.new('RGBA', image_size, bg_color)

    # Draw the polygon on the image
    draw = ImageDraw.Draw(image)
    draw.polygon(poly.exterior.coords, outline=line_color)

    # Save the image as a PNG file
    image.save(file_path, 'PNG')


def save_racepack(racepack: dict):
    """Save a racepack to disk."""

    # Error checking
    if not validate_race_package(racepack):
        raise ValueError("Invalid racepack")

    now = datetime.now()
    package_id = generate_uuid()
    package_name = f'{package_id}_{now.strftime("%Y%m%d_%H%M%S")}'

    racepack_dir = racepack['GLOBAL_VARS']['RACEPACK_DIR']
    track_name = racepack['TRACK_VARS']['TRACK_FILE_NAME']
    track_racepack_root = f"{racepack_dir}/{track_name}"
    package_dir = f"{track_racepack_root}/{package_name}"

    # Create Metadata

    metedata = {
        'created': now.strftime("%Y-%m-%d %H:%M:%S"),
        'package_id': package_id,
        'package_name': package_name,
        'package_dir': package_dir,
        'track_name': track_name,
        'track_racepack_root': track_racepack_root,
        'details': {
            'version': racepack['VERSIONS'],
            'track': {
                'name': racepack['TRACK_VARS']['TRACK_FILE_NAME'],
                'reduced': racepack['TRACK_VARS']['TRACK_REDUCE_PERC'],
                'fig': f'../track.original.fig.png',
                #'poly': f'../track.original.poly.png',
                #'reduced_poly': f'./package.reduced.poly.png',
                'overlay': f'./package.overlay.fig.png',
            },
            "race_line": {
                'xi_iterations': racepack['RACE_LINE_VARS']['RACE_LINE_XI_ITERATIONS'],
                'line_iterations': racepack['RACE_LINE_VARS']['RACE_LINE_LINE_ITERATIONS'],
                'original_length': racepack['RACE_LINE_VARS']['RACE_LINE_LENGTH_ORIGINAL'],
                'improved_length': racepack['RACE_LINE_VARS']['RACE_LINE_IMPROVED_LENGTH'],
                'fig': f'./package.improved.raceline.fig.png',
                'bin': f'./package.array.bin.npy',
                'txt': f'./package.array.txt.py',
            },
            'optimal': {
                'fig': f'./package.optimal.fig.png',
                'txt': f'./package.optimal.txt',
            },
            'reward_function': f'./package.reward_function.py',
            'generator_notebook': f'./package.generator.ipynb',
        }
    }

    # Create the racepack directory if it doesn't exist
    isCreated = create_folder(track_racepack_root)
    if isCreated:
        # save original track polygon
        # TODO: fix this, it generates a blank image
        #save_polygon_as_png(
        #    racepack['TRACK_VARS']['TRACK_SHAPELY']['road_poly'],
        #    f"{track_racepack_root}/track.original.poly.png"
        #)
        racepack['TRACK_VARS']['TRACK_FIGURE'].savefig(f'{track_racepack_root}/track.original.fig.png')

    # create package folder
    _ = create_folder(package_dir)

    # create compressed pickle
    racepack['METADATA'] = metedata
    create_compressed_pickle(f"{package_dir}/package.racepack.pickle.gz", racepack)

    # Write numpy array text to file
    write_file(
        f"{package_dir}/package.array.txt.py",
        np.array_repr(racepack['RACE_LINE_VARS']['RACE_LINE_IMPROVED_LOOP_RACE_LINE'])
    )

    # Write numpy array binary to file
    np.save(
        f"{package_dir}/package.array.bin.npy",
        racepack['RACE_LINE_VARS']['RACE_LINE_IMPROVED_LOOP_RACE_LINE']
    )

    # save reduced track polygon
    # TODO: fix this, it generates a blank image
    #save_polygon_as_png(
    #    racepack['TRACK_VARS']['TRACK_REDUCED_SHAPELY']['road_poly'],
    #    f"{package_dir}/package.reduced.poly.png"
    #)

    # save overlay track figure
    racepack['TRACK_VARS']['TRACK_FIGURE_OVERLAY'].savefig(f'{package_dir}/package.overlay.fig.png')

    # save race line figure
    racepack['RACE_LINE_VARS']['RACE_LINE_IMPROVED_FIG'].savefig(f'{package_dir}/package.improved.raceline.fig.png')

    # copy reward function
    copy_file(racepack['GLOBAL_VARS']['REWARD_FUNCTION'], f"{package_dir}/package.reward_function.py")

    # copy notebook used to generate racepack
    copy_file(racepack['GLOBAL_VARS']['NOTEBOOK'], f"{package_dir}/package.generator.ipynb")

    # Search for look ahead package
    look_ahead_package = search_list(
        racepack['OPTIMAL_SPEED_VARS']['OPTIMAL_SPEED_TARGET_LOOK_AHEAD_POINTS'],
        list(racepack['OPTIMAL_SPEED_VARS']['OPTIMAL_SPEED_VELOCITY'].keys()),
    )
    if len(look_ahead_package) > 1:
        raise ValueError("Multiple look ahead packages found")
    else:
        look_ahead_name = look_ahead_package[0]
        look_ahead_package = racepack['OPTIMAL_SPEED_VARS']['OPTIMAL_SPEED_VELOCITY'][look_ahead_package[0]]
        metedata['details']['race_line']['optimal_time'] = look_ahead_package['TOTAL_TIME']
        write_optimals(f"{package_dir}/package.optimal.txt", look_ahead_package['RACE_PACKAGE'])
        as_json = look_ahead_package['ACTION_SPACE_E'][["steering","velocity"]].copy()
        as_json = as_json.round(2)  # TODO: should this be 4; originally 4?
        as_json.columns = ["steering_angle", "speed"]
        as_json["index"] = as_json.index
        as_json = json.dumps(json.loads(as_json.to_json(orient="records", lines=False)), indent=4)
        write_file(f"{package_dir}/package.action_space.json", as_json)
        racepack['OPTIMAL_SPEED_VARS']["OPTIMAL_SPEED_VELOCITY_FIGS"][look_ahead_name].savefig(f'{package_dir}/package.optimal.fig.png')



    # create track.package.json
    package = dict_to_json(metedata)
    write_file(f"{package_dir}/package.json", package)


    return metedata
