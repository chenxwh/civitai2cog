"""
adpated from https://github.com/camenduru/sd-civitai-browser/blob/main/scripts/civitai-api.py
"""

import requests
import json
import time
import threading
import urllib.request
import os
from tqdm import tqdm
import re
from requests.exceptions import ConnectionError


def download_file(url, file_name):
    # Maximum number of retries
    max_retries = 5

    # Delay between retries (in seconds)
    retry_delay = 10

    while True:
        # Check if the file has already been partially downloaded
        if os.path.exists(file_name):
            # Get the size of the downloaded file
            downloaded_size = os.path.getsize(file_name)

            # Set the range of the request to start from the current size of the downloaded file
            headers = {"Range": f"bytes={downloaded_size}-"}
        else:
            downloaded_size = 0
            headers = {}

        # Split filename from included path
        tokens = re.split(re.escape("\\"), file_name)
        file_name_display = tokens[-1]

        # Initialize the progress bar
        progress = tqdm(
            total=1000000000,
            unit="B",
            unit_scale=True,
            desc=f"Downloading {file_name_display}",
            initial=downloaded_size,
            leave=False,
        )

        # Open a local file to save the download
        with open(file_name, "ab") as f:
            while True:
                try:
                    # Send a GET request to the URL and save the response to the local file
                    response = requests.get(url, headers=headers, stream=True)

                    # Get the total size of the file
                    total_size = int(response.headers.get("Content-Length", 0))

                    # Update the total size of the progress bar if the `Content-Length` header is present
                    if total_size == 0:
                        total_size = downloaded_size
                    progress.total = total_size

                    # Write the response to the local file and update the progress bar
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                            progress.update(len(chunk))

                    downloaded_size = os.path.getsize(file_name)
                    # Break out of the loop if the download is successful
                    break
                except ConnectionError as e:
                    # Decrement the number of retries
                    max_retries -= 1

                    # If there are no more retries, raise the exception
                    if max_retries == 0:
                        raise e

                    # Wait for the specified delay before retrying
                    time.sleep(retry_delay)

        # Close the progress bar
        progress.close()
        downloaded_size = os.path.getsize(file_name)
        # Check if the download was successful
        if downloaded_size >= total_size:
            print(f"{file_name_display} successfully downloaded.")
            break
        else:
            print(f"Error: File download failed. Retrying... {file_name_display}")


def download_file_thread(url, file_name, content_type, use_new_folder, model_name):
    if content_type == "Checkpoint":
        folder = "models/Stable-diffusion"
        new_folder = "models/Stable-diffusion/new"
    elif content_type == "Hypernetwork":
        folder = "models/hypernetworks"
        new_folder = "models/hypernetworks/new"
    elif content_type == "TextualInversion":
        folder = "embeddings"
        new_folder = "embeddings/new"
    elif content_type == "AestheticGradient":
        folder = (
            "extensions/stable-diffusion-webui-aesthetic-gradients/aesthetic_embeddings"
        )
        new_folder = "extensions/stable-diffusion-webui-aesthetic-gradients/aesthetic_embeddings/new"
    elif content_type == "VAE":
        folder = "models/VAE"
        new_folder = "models/VAE/new"
    elif content_type == "LORA":
        folder = "extensions/sd-webui-additional-networks/models/lora"
        new_folder = "extensions/sd-webui-additional-networks/models/lora/new"
    if (
        content_type == "TextualInversion"
        or content_type == "VAE"
        or content_type == "AestheticGradient"
    ):
        if use_new_folder:
            model_folder = new_folder
            if not os.path.exists(new_folder):
                os.makedirs(new_folder)

        else:
            model_folder = folder
            if not os.path.exists(model_folder):
                os.makedirs(model_folder)
    else:
        if use_new_folder:
            model_folder = os.path.join(
                new_folder,
                model_name.replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace("|", ""),
            )
            if not os.path.exists(new_folder):
                os.makedirs(new_folder)
            if not os.path.exists(model_folder):
                os.makedirs(model_folder)

        else:
            model_folder = os.path.join(
                folder,
                model_name.replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace("|", ""),
            )
            if not os.path.exists(model_folder):
                os.makedirs(model_folder)

    path_to_new_file = os.path.join(model_folder, file_name)

    thread = threading.Thread(target=download_file, args=(url, path_to_new_file))

    # Start the thread
    thread.start()


url = "https://civitai.com/api/download/models/5365?type=Model&format=PickleTensor"
file_name = "MAGIFACTORYTShirt_magifactoryTShirt.ckpt"
content_type = "VAE"
use_new_folder = False
model_name = "t_shirt_sd"
download_file_thread(
    url=url,
    file_name=file_name,
    content_type=content_type,
    use_new_folder=use_new_folder,
    model_name=model_name,
)
