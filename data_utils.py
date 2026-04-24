import os
import subprocess
import zipfile  
def download_tiny_imagenet(target_dir="./data"):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    subprocess.run(["kaggle", "datasets", "download", "-d", "akash2sharma/tiny-imagenet", "-p", target_dir], check=True)
    zip_path = os.path.join(target_dir, "tiny-imagenet.zip")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
def prepare_val_folder(val_dir):
    pass 
def download_cifar10(target_dir="./data"):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    subprocess.run(["kaggle", "datasets", "download", "-d", "cifar/cifar-10", "-p", target_dir], check=True)
    zip_path = os.path.join(target_dir, "cifar-10.zip")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
if __name__ == "__main__":
    download_tiny_imagenet()