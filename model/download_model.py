import gdown


def download_model_from_google_drive(folder_url):
    if folder_url is None:
        with open('.model_url', 'r') as f:
            folder_url = f.readline()
    gdown.download_folder(folder_url)


if __name__ == '__main__':
    url = input()
    if len(url) == 0:
        url = None
    download_model_from_google_drive(url)
