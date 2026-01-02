import os
import subprocess
import urllib.request

def main():
    url = "https://www.kaggle.com/api/v1/datasets/download/deasadiqbal/private-data-1"
    zip_path = "./dataset/ISPRS-Postdam/ISPRS-Postdam.zip"
    extract_dir = "./dataset/ISPRS-Postdam/"

    os.makedirs(extract_dir, exist_ok=True)

    if not os.path.exists(zip_path):
        print("Downloading dataset...")
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete.")
    else:
        print("Zip file already exists. Skipping download.")

    print("Extracting dataset...")
    subprocess.run(["7z", "x", "-aos", zip_path, f"-o{extract_dir}"], check=True)
    print("Extraction complete.")

if __name__ == "__main__":
    main()
