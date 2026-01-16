import os
import subprocess
import urllib.request

def main():
    urls ={
        'small': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth',
        'base': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth',
        'large': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth'
    }
    extract_dir = "./pretrained/"

    os.makedirs(extract_dir, exist_ok=True)

    for model_size, url in urls.items():
        pth_path = os.path.join(extract_dir, f"dinov2_{model_size}.pth")

        if not os.path.exists(pth_path):
            print("Downloading pretrained...")
            urllib.request.urlretrieve(url, pth_path)
            print("Download complete.")
        else:
            print("Pretrained already exists. Skipping download.")

if __name__ == "__main__":
    main()
