import geopandas as gpd
from shapely.geometry import Point
import random

class RandomPointGenerator:
    SHAPEFILE_PATH = "ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp"

    def __init__(self, random_state=1234):
        # Load the world shapefile
        self.world = gpd.read_file(RandomPointGenerator.SHAPEFILE_PATH)
        # Set the random state for reproducibility
        self.random_state = random_state
    
    def generate_points_in_country(self, country_name: str, num_points: int) -> gpd.GeoDataFrame:

        random.seed(self.random_state)

        # Filter to obtain the shapefile of the specified country
        country = self.world[self.world['NAME'] == country_name]

        if country.empty:
            raise ValueError(f"Le pays '{country_name}' n'a pas été trouvé dans le shapefile.")
        
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
        
        return points_gdf

    def plot_points_in_country(self, country_name: str, num_points: int, markersize:float = 5) -> None:
        # Generate random points in the specified country
        points_gdf = self.generate_points_in_country(country_name, num_points)
        
        # Plot the country boundaries and the random points
        ax = self.world[self.world['NAME'] == country_name].plot(color='#CACACA', edgecolor='black')
        points_gdf.plot(ax=ax, color='red', markersize=markersize)
        ax.set_axis_off()
