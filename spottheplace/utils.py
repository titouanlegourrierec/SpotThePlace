import os
from itertools import product

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import box, Point


def data_to_dataframe(directory: str) -> pd.DataFrame:
    """
    Converts image data in a directory to a pandas DataFrame.

    Args:
        - directory (str): The path to the directory containing the image files.

    Returns:
        - pd.DataFrame: A DataFrame containing the image data with columns for labels, longitude, latitude, orientation, and image paths.
    """
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
        if file.endswith(".png") or file.endswith(".jpg"):
            image_paths.append(file)

            image_name = file.split("/")[-1]

            countries.append(image_name.split("_")[0])
            longs.append(float(image_name.split("_")[1]))
            lats.append(float(image_name.split("_")[2]))

            orientations.append(ORIENTATIONS_DICT.get(image_name.split("_")[3].split(".")[0]))

    df = pd.DataFrame({
        "label": countries,
        "long": longs,
        "lat": lats,
        "orientation": orientations,
        "image_path": image_paths
        })

    df.to_csv(os.path.join(directory, f"{os.path.basename(directory)}.csv"), index=False)

    return df


def france_grid_to_dataframe(directory: str, n: int = 10) -> pd.DataFrame:
    """
    Generate a grid of n x n cells over the metropolitan area of France.

    Parameters:
        - n (int): The number of cells along each dimension of the grid. Default is 10.
        - show (bool): If True, display a plot of the grid overlaid on the map of France. Default is False.

    Returns:
        - gpd.GeoDataFrame: A GeoDataFrame containing the grid cells that intersect with the metropolitan area of France.
    """
    list_files = _list_files_in_directory(directory)
    grid = france_grid(n)

    ORIENTATIONS_DICT = {
        "90h": "E",
        "180h": "S",
        "270h": "W",
        "360h": "N"
    }
    image_paths = []
    labels = []
    longs = []
    lats = []
    orientations = []

    for file in list_files:
        if file.endswith(".png") or file.endswith(".jpg"):
            image_name = file.split("/")[-1]

            long = float(image_name.split("_")[1])
            lat = float(image_name.split("_")[2])
            point = Point(long, lat)
            match = grid[grid.geometry.contains(point)]

            if not match.empty:
                image_paths.append(file)
                labels.append(match.iloc[0]['label'])
                longs.append(long)
                lats.append(lat)
                orientations.append(ORIENTATIONS_DICT.get(image_name.split("_")[3].split(".")[0]))

    df = pd.DataFrame({
        "label": labels,
        "long": longs,
        "lat": lats,
        "orientation": orientations,
        "image_path": image_paths
        })

    df.to_csv(os.path.join(directory, "France_Grid.csv"), index=False)

    return df


def france_region_to_dataframe(directory: str) -> pd.DataFrame:
    """
    Converts image data in a directory to a pandas DataFrame with region labels for France.

    Args:
        - directory (str): The path to the directory containing the image files.

    Returns:
        - pd.DataFrame: A DataFrame containing the image data with columns for labels, longitude, latitude, orientation, and image paths.
    """
    list_files = _list_files_in_directory(directory)

    # Load the world map and filter the country of France and its regions that are located in the metropolitan area
    world = gpd.read_file("./ne_10m_admin_1_states_provinces/ne_10m_admin_1_states_provinces.shp")
    country = world[world["admin"] == "France"]
    regions = country.dissolve(by="region")
    regions = regions[regions.geometry.bounds['miny'] > 31]

    ORIENTATIONS_DICT = {
        "90h": "E",
        "180h": "S",
        "270h": "W",
        "360h": "N"
    }
    image_paths = []
    labels = []
    longs = []
    lats = []
    orientations = []

    for file in list_files:
        if file.endswith(".png") or file.endswith(".jpg"):
            image_name = file.split("/")[-1]

            long = float(image_name.split("_")[1])
            lat = float(image_name.split("_")[2])
            point = Point(long, lat)
            match = regions[regions.geometry.contains(point)]

            if not match.empty:
                image_paths.append(file)
                labels.append(match.index[0])
                longs.append(long)
                lats.append(lat)
                orientations.append(ORIENTATIONS_DICT.get(image_name.split("_")[3].split(".")[0]))

    df = pd.DataFrame({
        "label": labels,
        "long": longs,
        "lat": lats,
        "orientation": orientations,
        "image_path": image_paths
        })

    df.to_csv(os.path.join(directory, "France_Regions.csv"), index=False)

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


def france_grid(n: int = 10, show: bool = False) -> gpd.GeoDataFrame:

    world = gpd.read_file("./ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp")

    # Filter Metropole France
    country = world[world["NAME"] == "France"].explode(index_parts=True)
    country = country[country.geometry.bounds['miny'] > 31]

    # Get the bounds of the country and calculate the grid cell dimensions
    minx, miny, maxx, maxy = country.total_bounds
    grid_width = (maxx - minx) / n
    grid_height = (maxy - miny) / n

    # Create a grid of n x n cells
    grid = [
        box(minx + j * grid_width, miny + (n - i - 1) * grid_height,
            minx + (j + 1) * grid_width, miny + (n - i) * grid_height)
        for i, j in product(range(n), repeat=2)
    ]

    # Create labels for the grid cells
    labels = [f"{chr(65 + j)}{i+1}" for i, j in product(range(n), range(n))]

    # Create a GeoDataFrame from the grid
    grid = gpd.GeoDataFrame({'geometry': grid, 'label': labels})

    # Filter the grid cells that intersect with the country
    grid = grid[grid.intersects(country.unary_union)]

    if show:
        _, ax = plt.subplots(figsize=(20, 20))
        country.plot(ax=ax, color='white', edgecolor='black')
        grid.boundary.plot(ax=ax, color='red')
        plt.axis('off')
        plt.show()

    return grid
