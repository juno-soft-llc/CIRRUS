from ultralytics import YOLO
from roboflow import Roboflow
import torch
import os


def download_dataset(personal_api_key: str) -> str:
    """
    Downloads a dataset from Roboflow using the provided API key.

    Parameters:
    - personal_api_key (str): Your personal API key for Roboflow.

    Returns:
    - str: The path to the downloaded dataset.
    """
    # Initialize Roboflow with your API key
    rf = Roboflow(api_key=personal_api_key)

    # Get the project and version
    project = rf.workspace("roboflow-universe-projects").project("buildings-instance-segmentation")
    version = project.version(4)

    # Specify the path to the folder for downloading
    dataset_path_init = "datasets/"
    if not os.path.exists(dataset_path_init):
        os.makedirs(dataset_path_init)  # Create the directory if it doesn't exist

    # Download the dataset to the specified folder
    dataset = version.download("yolov8", location=dataset_path_init, overwrite=True)
    dataset_path = dataset.location
    return dataset_path


def correct_yaml(path_project: str, path_yaml: str) -> None:
    """
    Corrects the paths in the YAML configuration file for the dataset.

    Parameters:
    - path_project (str): The path to the project directory.
    - path_yaml (str): The path to the YAML configuration file.
    """
    with open(path_yaml, 'r') as file:
        lines = file.readlines()
        # Update the paths for test, train, and validation images
        lines[9] = "test: " + path_project + "/test/images\n"
        lines[10] = "train: " + path_project + "/train/images\n"
        lines[11] = "val: " + path_project + "/valid/images\n"

    with open(path_yaml, 'w') as file:
        file.writelines(lines)  # Write the updated lines back to the YAML file


def train(personal_api_key: str, train_voc: dict, experiment: str = 'yolov8_nano') -> None:
    """
    Trains a YOLO model using the specified training parameters.

    Parameters:
    - personal_api_key (str): Your personal API key for Roboflow.
    - train_voc (dict): A dictionary containing training parameters such as epochs, image size, and batch size.
    - experiment (str): The name of the experiment (default is 'yolov8_nano').

    Returns:
    - None: The function performs training and returns nothing.
    """
    current_directory = os.getcwd()
    if not os.path.exists('datasets/'):
        path_dataset = download_dataset(personal_api_key)  # Download dataset if it doesn't exist
    else:
        path_dataset = f'{current_directory}/datasets'

    path_yaml = f'{path_dataset}/data.yaml'
    correct_yaml(path_dataset, path_yaml)  # Correct the YAML file paths

    model = YOLO("yolov8n-seg.yaml").load("yolov8n-seg.pt")  # Load the YOLO model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Set device to GPU if available

    # Train the model with the specified parameters
    results = model.train(
        data=path_yaml,
        epochs=train_voc["epoch"],
        imgsz=train_voc["imgsz"],
        batch=train_voc["batch"],
        device=device,
        project=current_directory,
        name=experiment,
        exist_ok=True,
    )
