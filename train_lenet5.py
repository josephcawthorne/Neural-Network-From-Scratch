# In this file, I am going to try and write the training loop for my sneaker vs coat model.
# I am going to try and re-use the helpers from the other files I wrote here.

import os
import random
import time  # I am adding this so I can time how long this is running
from utils_fashion_mnist import load_fashion_mnist_folder, filter_to_sneaker_vs_coat, pad_to_32x32, normalize_0_to_1, make_minibatches
from lenet5_core import LeNet5Binary

#dataset_folder = 'Dataset'
dataset_folder = os.path.join(os.path.dirname(__file__), 'Dataset')
MODEL_FILE = os.path.join(os.path.dirname(__file__), 'model_params_lenet5.json')

def preprocess_all(images, labels):
    # I am going to pad to 32x32 and then normalize 0..1
    new_images = []
    new_labels = []
    for i in range(len(images)):
        padded = pad_to_32x32(images[i])
        normed = normalize_0_to_1(padded)
        new_images.append(normed)
        new_labels.append(labels[i])
    return new_images, new_labels

def accuracy(model, images, labels):
    # I want to just do a basic accuracy on the full list
    correct = 0
    for i in range(len(images)):
        logits, probs = model.forward(images[i])
        pred = 0
        if probs[1] > probs[0]:
            pred = 1
        if pred == labels[i]:
            correct += 1
    if len(images) == 0:
        return 0.0
    return correct / len(images)

def train():
    # I am going to ask the user how many minutes this should run for.
    print("Type the max minutes you want me to run for (or press Enter for no limit):")
    _user_text = input().strip()
    max_minutes = None
    if _user_text != "":
        try:
            max_minutes = float(_user_text)
        except:
            max_minutes = None  # if I can't parse it I will just continue as normal

    start_time = time.time()

    def time_is_up():
        # this is just a helper so I can check the time in my loop
        if max_minutes is None:
            return False
        return (time.time() - start_time) >= (max_minutes * 60.0)

    # load and filter the dataset to only the 2 classes I want
    (train_images, train_labels), (test_images, test_labels) = load_fashion_mnist_folder(dataset_folder)
    train_images, train_labels = filter_to_sneaker_vs_coat(train_images, train_labels)
    test_images, test_labels = filter_to_sneaker_vs_coat(test_images, test_labels)

    # preprocess
    train_images, train_labels = preprocess_all(train_images, train_labels)
    test_images, test_labels = preprocess_all(test_images, test_labels)

    # create model
    model = LeNet5Binary()
    # if I have a saved model I will load it
    if os.path.exists(MODEL_FILE):
        model.load(MODEL_FILE)

    # basic training settings
    lr = 0.005
    epochs = 2  # I am keeping this small because I just want to see it run
    batch_size = 8

    # I will do a very simple shuffle each epoch
    indices = list(range(len(train_images)))

    for epoch in range(epochs):
        random.shuffle(indices)
        # build shuffled mini-batches
        shuffled_imgs = [train_images[i] for i in indices]
        shuffled_lbls = [train_labels[i] for i in indices]

        batches = make_minibatches(shuffled_imgs, shuffled_lbls, batch_size)

        # train over batches
        total = 0
        correct = 0
        for (imgs, lbls) in batches:
            # I will just loop within the batch and do SGD updates
            for i in range(len(imgs)):
                x = imgs[i]
                y = lbls[i]
                logits, probs = model.forward(x)
                # track accuracy
                pred = 1 if probs[1] > probs[0] else 0
                if pred == y:
                    correct += 1
                total += 1
                # backprop one sample (this is my small task)
                model.backward(logits, probs, y, lr)

                # after finishing this small task, I will check if I need to stop now
                if time_is_up():
                    model.save(MODEL_FILE)
                    print("time limit reached, I have saved what I learnt to:", MODEL_FILE)
                    return

        # at the end of epoch I will measure train/test accuracy
        train_acc = accuracy(model, train_images[:200], train_labels[:200])  # small slice just to be quick
        test_acc = accuracy(model, test_images[:200], test_labels[:200])
        print("epoch:", epoch+1, " train_acc:", train_acc, " test_acc:", test_acc)

        # I might also check time here so I can save between epochs if needed
        if time_is_up():
            model.save(MODEL_FILE)
            print("time limit reached after epoch, I have saved what I learnt to:", MODEL_FILE)
            return

    # save the model so I can re-use it
    model.save(MODEL_FILE)
    print("model saved to:", MODEL_FILE)

if __name__ == '__main__':
    train()
