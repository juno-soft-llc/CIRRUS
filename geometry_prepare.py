import numpy as np
import geopandas as gpd

def white_tile_check(tile) -> bool:
    """
    Checks if a tile is predominantly white.

    Parameters:
    - tile: The image tile to be checked.

    Returns:
    - bool: True if the tile is predominantly white (95% or more white pixels), False otherwise.
    """
    tile_bw = tile.convert("L")  # Convert the tile to grayscale
    image_array = np.array(tile_bw)  # Convert the image to a NumPy array
    white_pixels = np.sum(image_array == 255)  # Count the number of white pixels
    total_pixels = image_array.size  # Get the total number of pixels
    percentage_white_pixels = (white_pixels / total_pixels) * 100  # Calculate the percentage of white pixels

    return percentage_white_pixels < 95  # Return True if less than 95% of the pixels are white

def postprocessing_geom(
    gdf_geom: gpd.GeoDataFrame, name_geotiff: str, task_object='buildings', flag_save=False
) -> None:
    """
    Post-processes geometries in a GeoDataFrame.

    Parameters:
    - gdf_geom (gpd.GeoDataFrame): The GeoDataFrame containing geometries to be processed.
    - name_geotiff (str): The name of the GeoTIFF file for saving purposes.
    - task_object (str): The task object name (default is 'buildings').
    - flag_save (bool): If True, saves the processed geometries to a GeoJSON file.
    """
    buffer = 1  # Define a buffer distance
    gdf_geom["geometry"] = gdf_geom["geometry"].buffer(buffer)  # Apply buffer to geometries
    gdf_dissolve = gdf_geom.dissolve(by="class_name", as_index=False)  # Dissolve geometries by class name
    gdf_explode = gdf_dissolve.explode(index_parts=True)  # Explode the dissolved geometries
    gdf_geom = gpd.sjoin(gdf_geom, gdf_explode, how="inner", predicate="intersects")  # Spatial join
    gdf_geom = gdf_geom.dissolve(
        by=["class_name_left", "index_right1"],
        aggfunc={"confidence_left": ["mean"]},
        as_index=False,
    )  # Dissolve again by class name and index

    gdf_geom["geometry"] = gdf_geom["geometry"].buffer(-buffer)  # Remove the buffer
    gdf_geom["geometry"] = gdf_geom["geometry"].simplify(tolerance=1)  # Simplify geometries

    # Rename columns for clarity
    gdf_final = gdf_geom.rename(
        columns={
            "": "confidence",
            "class_name_left": "class_name",
            "index_right1": "index",
        }
    )
    gdf_final["confidence"] = gdf_final[("confidence_left", "mean")].apply(
        lambda x: round(x, 4)  # Round confidence values to 4 decimal places
    )
    gdf_final = gdf_final[["confidence", "class_name", "index", "geometry"]]  # Select relevant columns
    gdf_final = gdf_final.set_crs(3857, allow_override=True)  # Set the coordinate reference system

    # Save the processed GeoDataFrame to a GeoJSON file if specified
    if flag_save:
        gdf_final.to_file(
            f"predicted_geojson/{task_object}_{name_geotiff}.geojson"
        )
