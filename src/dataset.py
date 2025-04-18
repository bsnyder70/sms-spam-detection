import requests
import json


def download_data():
    """
    Downloads data from our chosen meme dataset into a format ready for training.
    """

    dataset_url = "https://raw.githubusercontent.com/eujhwang/meme-cap/refs/heads/main/data/memes-test.json"

    json_data = json.loads(requests.get(dataset_url).text)

    image_names = []
    meme_captions = []
    for example in json_data:
        continue


download_data()