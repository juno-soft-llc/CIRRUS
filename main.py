from download_geotiff import download_image
from init import transform_geotiff
from predict import inference
from train import train

# Flag to determine whether to run prediction or training
flag_predict = False

if flag_predict:
    # Download and transform the GeoTIFF file
    url = "https://oin-hotosm.s3.amazonaws.com/5d4aa573ad3e9500059b3805/0/5d4aa573ad3e9500059b3810.tif"
    name_geotiff = 'geotiff_example'
    download_image(url)
    path_geotiff = transform_geotiff(name_geotiff)  # Transform the GeoTIFF file

    # Prepare the inference configuration
    inference_voc = {
        "path_geotiff": path_geotiff,
        "name_geotiff": name_geotiff,
        "task_object": {
            "task": "buildings",  # Specify the task type
            "path_weights": "weights/best.pt",  # Path to the model weights
            "confidence": 0.4,  # Confidence threshold for predictions
            "imgsz": 640,  # Image size for inference
        },
    }

    # Run inference with the prepared configuration
    inference(inference_voc)
else:
    # If prediction is not enabled, prompt for the Roboflow API key
    api_key = input('Enter your personal API key from Roboflow: ')

    # Prepare the training configuration
    train_voc = {
        "epoch": 1,  # Number of training epochs
        "batch": 4,  # Batch size for training
        "imgsz": 640,  # Image size for training
    }

    # Attempt to train the model and handle any exceptions
    try:
        train(api_key, train_voc)  # Train the model with the provided API key and training config
    except Exception as error:
        print(error)  # Print any errors that occur during training
