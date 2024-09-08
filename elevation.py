import rasterio
from point import Point
from typing import List


class Elevation(Point):
    def __init__(self, longitude, latitude):
        super().__init__(longitude, latitude)

    def get_elevation(self, dem=None) -> float:
        lon = int(self.longitude)
        lat = int(self.latitude)
        path_to_file = ""
        if lat > 0:
            path_to_file += f"N{lat}"
        else:
            path_to_file += f"S{lat}"
        if lon > 0:
            path_to_file += f"E{lon}"
        else:
            path_to_file += f"W{lon}"
        if not dem:
            dem = rasterio.open(f'data/{path_to_file}_FABDEM_V1-2.tif')
        dem_data = dem.read(1)
        row, col = dem.index(self.longitude, self.latitude)
        return round(dem_data[row, col].__float__(), 2)

    @staticmethod
    def get_elevation_for_multiple_points(points: List[Point] = None) -> List[dict]:
        lon = int(points[0].longitude)
        lat = int(points[0].latitude)
        path_to_file = ""
        if lat > 0:
            path_to_file += f"N{lat}"
        else:
            path_to_file += f"S{lat}"
        if lon > 0:
            path_to_file += f"E{lon}"
        else:
            path_to_file += f"W{lon}"
        dem = rasterio.open(f'data/{path_to_file}_FABDEM_V1-2.tif')
        data_elevation = []
        for point in points:
            elevation = Elevation(longitude=point.longitude, latitude=point.latitude).get_elevation(dem=dem)
            data_elevation.append(
                {
                    "longitude": point.longitude,
                    "latitude": point.latitude,
                    "elevation": elevation
                }
            )
        return data_elevation
