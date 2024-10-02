from tqdm import tqdm
import time
from shapely.geometry import mapping, Polygon
from ultralytics import YOLO
import fiona

from init import *
from geometry_prepare import *


def inference(inference_voc: dict, image_save=False) -> gpd.GeoDataFrame:
    """
    Performs inference on images from a GeoTIFF file using the YOLO model.

    Parameters:
    - inference_voc (dict): A dictionary containing parameters for inference, including file paths and settings.
    - image_save (bool): If True, saves images with inference results.

    Returns:
    - gpd.GeoDataFrame: Geospatial data obtained from the inference.
    """
    start_time = time.time()

    # Extract parameters from the input dictionary
    path_geotiff = inference_voc["path_geotiff"]
    name_geotiff = inference_voc["name_geotiff"]
    path_weights = inference_voc["task_object"]["path_weights"]
    imgsz = inference_voc["task_object"]["imgsz"]
    confidence = inference_voc["task_object"]["confidence"]
    task = inference_voc["task_object"]["task"]

    # Initialize coordinates and image dimensions
    coordinates, diff_x, diff_y, schema = init(path_geotiff)
    width, height, img = define_image_size(path_geotiff)
    tile_size = (imgsz, imgsz)
    step = int(imgsz / 2)

    # Iterate over the image tiles
    for i in tqdm(range(0, width - tile_size[0] + step, step)):
        for j in range(0, height - tile_size[1] + step, step):
            box = (i, j, i + tile_size[0], j + tile_size[1])
            tile = img.crop(box)
            # Check for white pixels in the tile
            if white_tile_check(tile):
                model = YOLO(path_weights)
                result = model(
                    tile,
                    save=False,
                    show_boxes=False,
                    imgsz=imgsz,
                    conf=confidence,
                    show_labels=False,
                    verbose=False,
                )[0]

                # Save the image if specified
                if image_save:
                    os.makedirs(f"results_{task}", mode=0o777, exist_ok=True)
                    result.save(
                        filename=f"results_{task}/tile_{i}_{j}.jpg",
                        boxes=False,
                    )

                # Process the inference results
                for counter, r in enumerate(result):
                    cls_id = int(result.boxes[counter].cls.item())
                    cls_name = model.names[cls_id]
                    mask = r.masks[0]
                    mask_polygon_array = np.array([])

                    # Convert the mask to a polygon
                    for points in mask.xy:
                        is_first = True
                        first_x = None
                        first_y = None
                        for point in points:
                            point_x = round(
                                coordinates[0] + (point[0] + i) / width * diff_x,
                                3,
                            )
                            point_y = round(
                                coordinates[1] + (point[1] + j) / height * diff_y,
                                3,
                            )
                            if is_first:
                                first_x = point_x
                                first_y = point_y
                                is_first = False
                            mask_polygon_array = np.append(
                                mask_polygon_array,
                                np.array([point_x, point_y]),
                                axis=0,
                            )

                    # Close the polygon
                    mask_polygon_array = np.append(
                        mask_polygon_array, np.array([first_x, first_y]), axis=0
                    )
                    mask_polygon_array = mask_polygon_array.reshape(-1, 2)

                    # Create the polygon
                    try:
                        for row in mask_polygon_array:
                            if None in row:
                                raise ValueError("The array contains a None value")
                        poly = Polygon(mask_polygon_array)
                    except ValueError as error:
                        print(error)

                    # Write to shapefile
                    if not os.path.exists(f"shapefiles/{task}.shp"):
                        # Create a new file if it does not exist
                        with fiona.open(
                                f"shapefiles/{task}.shp",
                                "w",
                                "ESRI Shapefile",
                                schema,
                        ):
                            pass

                    # Open the file for writing
                    with fiona.open(
                            f"shapefiles/{task}.shp",
                            "a",
                            "ESRI Shapefile",
                            schema,
                    ) as c:
                        c.write(
                            {
                                "geometry": mapping(poly),
                                "properties": {
                                    "class_name": cls_name,
                                    "confidence": round(
                                        float(r.boxes.conf.cpu().numpy()[0]), 4
                                    ),
                                    "area": poly.area,
                                },
                            }
                        )

    # Read and process the shapefile
    if os.path.exists("shapefiles"):
        gdf = gpd.read_file(f"shapefiles/{task}.shp")
        postprocessing_geom(gdf, task, name_geotiff, flag_save=True)
        shutil.rmtree("shapefiles")
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Time taken for GeoTIFF recognition and geometry processing: {execution_time:.1f} seconds")
        return gdf
