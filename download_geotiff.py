import os
import requests

def download_image(url, folder='geotiff', filename='geotiff_example.tif'):
    """
    Downloads an image from the specified URL and saves it in the specified folder.

    :param url: URL of the image to download
    :param folder: Folder to save the image (default is "geotiff")
    :param filename: Filename for saving (default is "image.tif")
    """
    os.makedirs(folder, exist_ok=True)  # Create the folder if it does not exist
    file_path = os.path.join(folder, filename)  # Path to the file

    # Check if the file exists
    if os.path.exists(file_path):
        print(f"The file {file_path} already exists. No download needed.")
        return

    # Download the image
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f"The image has been successfully downloaded and saved to {file_path}")
    else:
        print("Error downloading the image:", response.status_code)
