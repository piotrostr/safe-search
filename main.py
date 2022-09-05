import argparse
import io
import time

from typing import Callable

from google.cloud import vision

def track_time(func: Callable, *args):
    s = time.time()
    func(*args)
    e = time.time()
    print(f"\nTime elapsed: {e.__sub__(s):.3f} sec")

def check_image(client: vision.ImageAnnotatorClient, path: str):
    with io.open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.safe_search_detection(image=image)
    safe = response.safe_search_annotation

    # Names of likelihood from google.cloud.vision.enums
    likelihood_name = (
        "UNKNOWN",
        "VERY_UNLIKELY",
        "UNLIKELY",
        "POSSIBLE",
        "LIKELY",
        "VERY_LIKELY",
    )
    print("Safe search:")

    print("adult:    {}".format(likelihood_name[safe.adult]))
    print("medical:  {}".format(likelihood_name[safe.medical]))
    print("spoofed:  {}".format(likelihood_name[safe.spoof]))
    print("violence: {}".format(likelihood_name[safe.violence]))
    print("racy:     {}".format(likelihood_name[safe.racy]))

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(
                response.error.message
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-path",
        type=str,
        default="./berkeley-county-tape-library.jpg",
    )
    args = parser.parse_args()

    client = vision.ImageAnnotatorClient()
    track_time(check_image, client, args.image_path)
