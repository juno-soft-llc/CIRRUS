import os
import shutil
from osgeo import gdal
from PIL import Image
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling


def define_image_size(path_geotiff) -> tuple:
    """
    Defines the size of the image and returns its dimensions and the image object.

    Parameters:
    - path_geotiff (str): The file path to the GeoTIFF image.

    Returns:
    - tuple: A tuple containing the width, height, and the image object.
    """
    Image.MAX_IMAGE_PIXELS = None  # Disable the limit on image pixel size
    img = Image.open(path_geotiff)
    width = img.size[0]
    height = img.size[1]
    return width, height, img


def get_coordinates_from_geotiff(file_path: str) -> list[int]:
    """
    Retrieves the geographical coordinates from a GeoTIFF file.

    Parameters:
    - file_path (str): The file path to the GeoTIFF image.

    Returns:
    - list[int]: A list containing the minimum and maximum coordinates [min_x, max_y, max_x, min_y].
    """
    dataset = gdal.Open(file_path, gdal.GA_ReadOnly)

    geo_transform = dataset.GetGeoTransform()
    min_x = geo_transform[0]
    max_y = geo_transform[3]
    pixel_width = geo_transform[1]
    pixel_height = geo_transform[5]

    width = dataset.RasterXSize
    height = dataset.RasterYSize

    max_x = min_x + width * pixel_width
    min_y = max_y + height * pixel_height

    coordinates = [min_x, max_y, max_x, min_y]
    return coordinates


def transform_geotiff(name_geotiff: str):
    """
    Transforms a GeoTIFF file to a specified coordinate reference system (EPSG:3857).

    Parameters:
    - name_geotiff (str): The name of the GeoTIFF file (without extension).

    Returns:
    - str: The path to the transformed GeoTIFF file or the original path if no transformation is needed.
    """
    root_path = 'geotiff'
    path_geotiff = f'{root_path}/{name_geotiff}.tif'

    with rasterio.open(path_geotiff) as src:
        src_crs = src.crs
        if src_crs == 'EPSG:3857':
            return path_geotiff  # No transformation needed

    try:
        with rasterio.open(path_geotiff) as src:
            src_crs = src.crs
            src_transform = src.transform
            src_width = src.width
            src_height = src.height

            # Define the target projection (WGS 84 / Pseudo-Mercator)
            dst_crs = 'EPSG:3857'  # Pseudo-Mercator
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src_crs, dst_crs, src_width, src_height, *src.bounds)

            with rasterio.open(f'{root_path}/{name_geotiff}_transformed.tif', 'w', driver='GTiff',
                               height=dst_height, width=dst_width,
                               count=src.count, dtype=src.dtypes[0],
                               crs=dst_crs, transform=dst_transform) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src_transform,
                        src_crs=src_crs,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.bilinear)
    except Exception as e:
        print(e)
        return path_geotiff  # Return the original path in case of an error

    return f'{root_path}/{name_geotiff}_transformed.tif'


def init(path_geotiff: str) -> tuple:
    """
    Initializes the environment for processing the GeoTIFF file.

    Parameters:
    - path_geotiff (str): The file path to the GeoTIFF image.

    Returns:
    - tuple: A tuple containing the coordinates, differences in coordinates, and schema for shapefiles.
    """
    coordinates = get_coordinates_from_geotiff(path_geotiff)

    # Round coordinates to three decimal places
    for i in range(len(coordinates)):
        coordinates[i] = round(coordinates[i], 3)

    diff_x = round(coordinates[2] - coordinates[0], 3)
    diff_y = round(coordinates[3] - coordinates[1], 3)

    # Clean up previous shapefiles if they exist
    if os.path.exists("shapefiles"):
        shutil.rmtree("shapefiles/")

    os.makedirs("shapefiles/", mode=0o777, exist_ok=True)
    os.makedirs("predicted_geojson/", mode=0o777, exist_ok=True)

    # Define the schema for the shapefile
    schema = {
        "geometry": "Polygon",
        "properties": {"class_name": "str", "confidence": "float", "area": "float"},
    }

    return coordinates, diff_x, diff_y, schema
