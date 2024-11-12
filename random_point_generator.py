import random

import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt


class RandomPointGenerator:
    # Made with Natural Earth. Free vector and raster map data @ naturalearthdata.com.
    SHAPEFILE_PATH = "ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp"

    def __init__(self, random_state: int = 1234):
        # Load the world shapefile
        self.world = gpd.read_file(RandomPointGenerator.SHAPEFILE_PATH)
        # Set the random state for reproducibility
        self.random_state = random_state

    def generate_points_in_country(self,
                                   country_name: str,
                                   num_points: int,
                                   markersize: float = 5,
                                   show: bool = True) -> gpd.GeoDataFrame:
        """
        Generates random points within the boundaries of a specified country and plots them.

        This method filters the world shapefile to obtain the geometry of the specified country,
        generates random points within the country's boundaries, and returns these points as a
        GeoDataFrame with additional columns for latitude, longitude, and country name. It also
        plots the generated points on the map of the specified country.

        Parameters:
            - country_name (str): The name of the country in which to generate points.
            - num_points (int): The number of random points to generate.
            - markersize (float): The size of the markers for the points. Defaults to 5.

        Returns:
            - gpd.GeoDataFrame: A GeoDataFrame containing the generated points with columns for
                            'id', 'long', 'lat', and 'country'.

        Example:
            >>> generator = RandomPointGenerator()
            >>> points_gdf = generator.generate_points_in_country('France', 100)
        """

        random.seed(self.random_state)

        # Filter to obtain the shapefile of the specified country
        country = self.world[self.world['NAME'] == country_name]

        if country.empty:
            raise ValueError(f"The country '{country_name}' was not found in the shapefile.")

        # Extract the bounds of the country's geometry
        minx, miny, maxx, maxy = country.total_bounds
        country_polygon = country.geometry.iloc[0]

        # Generate random points within the country boundaries
        points = []
        while len(points) < num_points:
            random_point = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
            if country_polygon.contains(random_point):
                points.append(random_point)

        # Convert to a GeoDataFrame
        points_gdf = gpd.GeoDataFrame(geometry=points, crs=country.crs)

        # Add columns for latitude, longitude, and country name
        points_gdf["id"] = range(1, num_points + 1)
        points_gdf["long"] = points_gdf.geometry.x
        points_gdf["lat"] = points_gdf.geometry.y
        points_gdf["country"] = country_name

        if show:
            # Plot the country boundaries and the random points
            ax = country.plot(color='#CACACA', edgecolor='black')
            points_gdf.plot(ax=ax, color='red', markersize=markersize)
            ax.set_axis_off()
            plt.show()

        return points_gdf
