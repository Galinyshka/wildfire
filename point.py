class Point:
    def __init__(self, longitude: float, latitude: float):
        self.longitude = longitude
        self.latitude = latitude

    def __repr__(self):
        return f"Point({self.longitude}, {self.latitude})"