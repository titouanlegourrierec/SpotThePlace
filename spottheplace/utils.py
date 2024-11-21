import os
import pandas as pd


def data_to_dataframe(directory: str) -> pd.DataFrame:

    list_files = _list_files_in_directory(directory)

    ORIENTATIONS_DICT = {
        "90h": "E",
        "180h": "S",
        "270h": "W",
        "360h": "N"
    }
    image_paths = []
    countries = []
    longs = []
    lats = []
    orientations = []

    for file in list_files:
        if file.endswith(".png"):
            image_paths.append(file)

            image_name = file.split("/")[-1]

            countries.append(image_name.split("_")[0])
            longs.append(float(image_name.split("_")[1]))
            lats.append(float(image_name.split("_")[2]))

            orientations.append(ORIENTATIONS_DICT.get(image_name.split("_")[3].split(".")[0]))

    df = pd.DataFrame({
        "label": countries,
        "longitude": longs,
        "latitude": lats,
        "orientation": orientations,
        "image_path": image_paths
        })

    df.to_csv(os.path.join(directory, f"{os.path.basename(directory)}.csv"), index=False)

    return df


def _list_files_in_directory(directory: str) -> list:
    """
    Retourne une liste triée de tous les fichiers dans un répertoire donné.

    Args:
        directory (str): Le chemin du répertoire à explorer.

    Returns:
        list: Une liste triée des chemins complets des fichiers dans le répertoire.
    """
    files_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            files_list.append(os.path.join(root, file))
    files_list.sort()

    return files_list
