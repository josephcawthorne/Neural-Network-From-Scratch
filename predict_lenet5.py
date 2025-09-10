# In this file, I am going to try and load my saved model and then ask the user for an image path.
# I am going to try and convert that image into the same format I used before, and then predict coat or sneaker.

import os
from lenet5_core import LeNet5Binary
from utils_fashion_mnist import pad_to_32x32, normalize_0_to_1

MODEL_FILE = os.path.join(os.path.dirname(__file__), 'model_params_lenet5.json')

def _load_image_to_28x28_list(file_path):
    # I am going to try to use Pillow just for opening the image and getting pixels.
    try:
        from PIL import Image
    except:
        print("I could not import Pillow (PIL). please install it first: pip install Pillow")
        return None

    # sometimes the user might type the path with extra quotes, I am going to strip those out
    if file_path.startswith('"') and file_path.endswith('"'):
        file_path = file_path[1:-1]
    if file_path.startswith("'") and file_path.endswith("'"):
        file_path = file_path[1:-1]

    try:
        # open, convert to grayscale "L", and resize to 28x28 because that is what the dataset uses
        img = Image.open(file_path).convert("L").resize((28, 28))
    except OSError as e:
        print("I could not open the image at:", file_path)
        print("reason:", e)
        return None

    pixels = list(img.getdata())
    # convert flat list to 28x28 nested list
    image28 = []
    idx = 0
    for r in range(28):
        row = []
        for c in range(28):
            row.append(pixels[idx])
            idx += 1
        image28.append(row)
    return image28

if __name__ == '__main__':
    print("Type the full file path of the image you want me to predict (28x28 grayscale works best):")
    file_path = input().strip()
    if file_path == "":
        print("I don't have a file path to read. I am going to stop now.")
    else:
        image28 = _load_image_to_28x28_list(file_path)
        if image28 is None:
            # I already printed the reason above.
            pass
        else:
            # pad and normalize just like I did for the dataset
            img32 = pad_to_32x32(image28)
            img32 = normalize_0_to_1(img32)

            # load model
            model = LeNet5Binary()
            loaded = model.load(MODEL_FILE)
            if not loaded:
                print("I do not have a saved model yet. please run train_lenet5.py first.")
            else:
                # predict
                logits, probs = model.forward(img32)
                pred = 1 if probs[1] > probs[0] else 0
                label_name = "sneaker" if pred == 1 else "coat"
                print("prediction:", label_name, " probs:", probs)
