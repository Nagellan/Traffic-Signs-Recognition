from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np
import os
from random import shuffle, choice, randrange
from time import time
from cv2 import resize
from albumentations import PadIfNeeded, Rotate, RandomBrightnessContrast
import csv


TEST_PATH = "GTSRB/Final_Test/Images"
TRAINING_PATH = "GTSRB/Final_Training/Images"


def read_training_data(rootpath):
    """
    Function for reading the images for training.
    :param rootpath:  path to the traffic sign data
    :return:          list of images, list of corresponding image information: width, height, class, track
    """
    images = []     # images
    img_info = []   # corresponding labels
    # loop over all classes
    for img_class in os.listdir(rootpath):
        prefix = rootpath + '/' + img_class + '/'                   # subdirectory for class
        gt_file = open(prefix + 'GT-' + img_class + '.csv')         # annotations file
        gt_reader = csv.reader(gt_file, delimiter=';')              # csv parser for annotations file
        next(gt_reader)                                             # skip header
        # loop over all images in current annotations file
        for row in gt_reader:
            images.append(plt.imread(prefix + row[0]))              # numpy array representation of image
            img_info.append([int(row[1]), int(row[2]),
                             img_class, row[0][:5]])                # width, height, class, track
        gt_file.close()
    return images, img_info


def read_test_data(rootpath):
    """
    Function for reading the images for testing.
    :param rootpath:  path to the testing traffic sign data
    :return:          list of images, list of corresponding image information: width, height, class, filename
    """
    images = []     # images
    img_info = []   # corresponding labels
    gt_file = open(rootpath + '/GT-final_test.csv')                # annotations file
    gt_reader = csv.reader(gt_file, delimiter=';')                 # csv parser for annotations file
    next(gt_reader)                                                # skip header
    # loop over all images in current annotations file
    for row in gt_reader:
        images.append(plt.imread(rootpath + '/' + row[0]))         # numpy array representation of image
        img_info.append([int(row[1]), int(row[2]),
                         format(int(row[7]), '05d'), row[0][:5]])  # width, height, class, filename
    gt_file.close()
    return images, img_info


def make_square(image, img_info):
    """
    Function for transforming images into square shape by adding padding to the smallest dimension.
    :param image:     image to be transformed
    :param img_info:  image data containing width and height etc
    :return:          processed image of square shape
    """
    if img_info[0] == img_info[1]:  # if image is already square
        return image
    max_side = max(img_info[0], img_info[1])    # figure's maximum dimension size
    pad = PadIfNeeded(max_side, max_side)
    return pad(image=image)['image']    # add padding to least dimension so that its size becomes equal to max's one


def split_train_images(images, img_info):
    """
    Function for splitting images set into training and validation sets with 4 to 1 proportion.
    :param images:    set of images to be splitted
    :param img_info:  set of images' information to be splitted accordingly
    :return:          training and validation sets with their information sets
    """
    tracks_unique = list(set([img_i[2] + '_' + img_i[3] for img_i in img_info]))  # unique tracks' names among classes
    shuffle(tracks_unique)
    proportion = len(tracks_unique)//5  # 80%-20% proportion for train and validation tracks accordingly
    validation_tracks = tracks_unique[:proportion]  # set of unique track names for validation
    t_set, v_set, t_set_info, v_set_info = [], [], [], []
    # loop through the whole set
    for i, i_info in enumerate(img_info):
        if i_info[2] + '_' + i_info[3] in validation_tracks:    # if current image belongs to validation set's track
            v_set.append(images[i])
            v_set_info.append((i_info[2], i_info[3]))
        else:
            t_set.append(images[i])
            t_set_info.append((i_info[2], i_info[3]))
    return t_set, v_set, t_set_info, v_set_info


def show_class_frequencies(t_set_info, rootpath, set_name):
    """
    Function for calculating the frequencies for images of each certain class.
    :param t_set_info:  images set information
    :param rootpath:    path to images set
    :param set_name:    name of a set used
    :return:            frequencies of each class and lists of image indices by classes
    """
    freq = {img_class[:5]: 0 for img_class in os.listdir(rootpath)}
    img_indices_by_classes = {img_class[:5]: [] for img_class in os.listdir(rootpath)}
    # calculate frequencies of each class and collect lists of image indices by classes through given information set
    for i, (img_class, img_track) in enumerate(t_set_info):
        freq[img_class] += 1
        img_indices_by_classes[img_class].append(i)
    # plot the bar chart of calculated frequencies
    plt.bar(freq.keys(), freq.values(), color='orange')
    plt.xlabel("Classes")
    plt.xticks(rotation=65, fontsize=8)
    plt.ylabel("Amount of images")
    plt.title("Class frequencies in " + set_name + " set")
    plt.show()
    return freq, img_indices_by_classes


def augment_image(img, rotate, rand_brightness_contrast):
    """
    Function for augmenting single image.
    :param img:                       image to be augmented
    :param rotate:                    rotation function
    :param rand_brightness_contrast:  brightness & contrast altering function
    :return:                          augmented image
    """
    new_img = img.copy()
    # alter image's brightness on ±0.3 and contrast on ±0.2
    new_img = rand_brightness_contrast(image=new_img)['image']
    new_img = rotate(image=new_img)['image']  # rotate image on ±20deg
    return new_img


def augment_images(t_set, t_set_info, freq, img_indices_by_classes):
    """
    Function for images augmentation.
    :param t_set:                   training set of images
    :param t_set_info:              training set images information
    :param freq:                    frequencies of each class
    :param img_indices_by_classes:  lists of image indices by classes
    :return:                        nothing, data is processed by reference
    """
    rotate, rand_brightness_contrast = \
        Rotate(limit=20, p=1), RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.2, p=1)
    max_num_images = max(freq.values())     # get maximum amount of images through all classes
    # loop through all image classes
    for img_class, num_of_images in freq.items():
        # add lacking amount of images to the training set
        for _ in range(max_num_images - num_of_images):
            index = choice(img_indices_by_classes[img_class])   # choose random image index of required class
            new_img = augment_image(t_set[index], rotate, rand_brightness_contrast)
            t_set.append(new_img)
            t_set_info.append((t_set_info[index][0], t_set_info[index][1]))


def get_test_set(rootpath, img_size):
    """
    Function for reading and processing test data.
    :param rootpath: path to test data
    :param img_size: size of images to be transformed to
    :return:         list of test images, list of corresponding image information: class, filename
    """
    test_set, test_set_info = read_test_data(rootpath)                                     # read test data images
    test_set = [make_square(image, test_set_info[i]) for i, image in enumerate(test_set)]  # make images square
    test_set = [resize(image, (img_size, img_size)) for image in test_set]                 # lead images to one size
    test_set_info = [(img_info[2], img_info[3]) for img_info in test_set_info]  # keep only image classes and filenames
    return test_set, test_set_info


def show_incorrect_prediction(img_set, correct_ans, pred_ans):
    """
    Function for detecting incorrect prediction and plotting incorrectly predicted image.
    :param img_set:      image set for taking the image from
    :param correct_ans:  correct predefined answers
    :param pred_ans:     predicted answers
    :return:             nothing
    """
    for i in range(len(correct_ans)):
        if correct_ans[i] != pred_ans[i]:
            print("Some incorrect prediction: expected", correct_ans[i], "received", pred_ans[i])
            plt.imshow(img_set[i])
            plt.title("Image predicted incorrectly: expected " + correct_ans[i] + " received " + pred_ans[i])
            plt.show()
            return


def show_recall_precision(set_to_predict_info, pred_set, classes):
    """
    Function for computing and plotting recall and precision.
    :param set_to_predict_info:  list of predefined truly correct answers
    :param pred_set:             list of predicted answers
    :param classes:              lists of classes
    :return:                     dictionaries with recall and precision scores by classes
    """
    recall = recall_score(set_to_predict_info, pred_set, average=None)
    precision = precision_score(set_to_predict_info, pred_set, average=None)
    # Plot recall bar chart
    plt.bar(classes, recall, color='orange')
    plt.xlabel("Classes")
    plt.xticks(rotation=65, fontsize=8)
    plt.ylabel("Recall score")
    plt.title("Recall score through classes")
    plt.show()
    # Plot precision bar chart
    plt.bar(classes, precision, color='blue')
    plt.xlabel("Classes")
    plt.xticks(rotation=65, fontsize=8)
    plt.ylabel("Precision score")
    plt.title("Precision score through classes")
    plt.show()
    return recall, precision


def train_model(img_size, augmented=True, test_data=True):
    """
    Function for training the model.
    :param img_size:   size of images to be transformed to
    :param augmented:  use augmentation technique on images or not
    :param test_data:  make predictions on test data or validation data
    :return:           nothing
    """
    init_time = time()
    # Step 1: Read all images from training dataset
    start_time = time()
    print("Step 1: Start reading all images from training dataset...")
    train_images, train_img_info = read_training_data(TRAINING_PATH)
    print("Step 1: Completed in", str(round(time() - start_time, 4)) + "s!")

    random_img_index = randrange(len(train_images))     # get index of random image from training dataset
    plt.imshow(train_images[random_img_index])  # plot image
    plt.title("Initial image")
    plt.show()

    # Step 2: Transform images to square shape
    start_time = time()
    print("Step 2: Start transforming images to square shape...")
    train_images = [make_square(image, train_img_info[i]) for i, image in enumerate(train_images)]
    print("Step 2: Completed in", str(round(time() - start_time, 4)) + "s!")

    plt.imshow(train_images[random_img_index])  # plot image made square
    plt.title("Image after making squared")
    plt.show()

    # Step 3: Transform images to same size
    start_time = time()
    print("Step 3: Start transforming images to same size...")
    train_images = [resize(image, (img_size, img_size)) for image in train_images]
    print("Step 3: Completed in", str(round(time() - start_time, 4)) + "s!")

    plt.imshow(train_images[random_img_index])  # plot image resized
    plt.title("Image after resizing")
    plt.show()

    # Step 4: Split data
    start_time = time()
    print("Step 4: Start splitting data...")
    train_set, valid_set, train_set_info, valid_set_info = split_train_images(train_images, train_img_info)
    print("Step 4: Completed in", str(round(time() - start_time, 4)) + "s!")

    # Choose between validation and test sets
    set_to_predict, set_to_predict_info = valid_set, valid_set_info
    if test_data:
        start_time = time()
        print("Start processing test data(reading, reshaping, resizing)...")
        set_to_predict, set_to_predict_info = get_test_set(TEST_PATH, img_size)
        print("Completed in", str(round(time() - start_time, 4)) + "s!")

    # Step 5: Find class frequencies in the resulting training set
    start_time = time()
    print("Step 5: Start finding class frequencies in the resulting training set...")
    frequencies, image_indices_by_classes = show_class_frequencies(train_set_info, TRAINING_PATH, "training")
    print("Step 5: Completed in", str(round(time() - start_time, 4)) + "s!")

    if augmented:
        # Step 6: Augmentation - produce synthetic images from the existing ones
        start_time = time()
        print("Step 6: Start augmentation process...")
        augment_images(train_set, train_set_info, frequencies, image_indices_by_classes)
        print("Step 6: Completed in", str(round(time() - start_time, 4)) + "s!")

        # plot augmented image
        plt.imshow(augment_image(train_images[random_img_index], Rotate(limit=20, p=1),
                                 RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.2, p=1)))
        plt.title("Image after augmentation")
        plt.show()

    # Find class frequencies in set to predict
    start_time = time()
    print("Start finding class frequencies in set to predict...")
    image_indices_by_classes = show_class_frequencies(set_to_predict_info, TRAINING_PATH, "to predict")[1]
    print("Completed in", str(round(time() - start_time, 4)) + "s!")

    # Step 7: Find class frequencies in the resulting training set after augmentation
    start_time = time()
    print("Step 7: Start finding class frequencies in the resulting training set after augmentation...")
    show_class_frequencies(train_set_info, TRAINING_PATH, "augmented training")
    print("Step 7: Completed in", str(round(time() - start_time, 4)) + "s!")

    init_set_to_predict = set_to_predict  # save images set before normalization and transforming into matrix

    # Step 8: Normalize images from both training set and set to predict
    start_time = time()
    print("Step 8: Start normalizing images from both training set and set to predict...")
    train_set, set_to_predict = [img/255 for img in train_set], [img/255 for img in set_to_predict]
    print("Step 8: Completed in", str(round(time() - start_time, 4)) + "s!")

    # Step 9: Transform training set and set to predict into matrix containing each image as 1D vector
    start_time = time()
    print("Step 9: Start transforming training set and set to predict into matrix...")
    train_set, train_set_info = [np.ravel(img) for img in train_set], [img_info[0] for img_info in train_set_info]
    set_to_predict = [np.ravel(img) for img in set_to_predict]
    set_to_predict_info = [img_info[0] for img_info in set_to_predict_info]
    print("Step 9: Completed in", str(round(time() - start_time, 4)) + "s!")

    print("Total time spent on data preprocessing:", str(round(time() - init_time, 4)) + "s!")

    # Step 10: Train the model
    start_time = time()
    print("Step 10: Start training the model...")
    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
    clf.fit(train_set, train_set_info)
    model_fit_time = round(time() - start_time, 4)
    print("Step 10: Completed in", str(model_fit_time) + "s!")

    # Step 11: Predict the data
    start_time = time()
    print("Step 11: Start predicting the data...")
    prediction = clf.predict(set_to_predict)
    print("Step 11: Completed in", str(round(time() - start_time, 4)) + "s!")

    # Step 12: Validate the model
    start_time = time()
    print("Step 12: Start validating the model...")
    accuracy = accuracy_score(set_to_predict_info, prediction)
    print("Model's accuracy:", accuracy)
    show_incorrect_prediction(init_set_to_predict, set_to_predict_info, prediction)
    classes_recall, classes_precision = \
        show_recall_precision(set_to_predict_info, prediction, image_indices_by_classes.keys())
    print("Model classes recall:", classes_recall, "\nModel classes precision:", classes_precision)
    print("Step 12: Completed in", str(round(time() - start_time, 4)) + "s!")

    print("\nTotal time spent:", str(round(time() - init_time, 4)) + "s.")

    return img_size, accuracy, model_fit_time


# list containing models' accuracy and overall model fit time
models_info = [
    train_model(30, test_data=False),  # train model on 30x30 images and validate using validation set
    train_model(30, augmented=False),  # train model on 30x30 images without augmentation, and validate using test set
    train_model(30),                   # train model on 30x30 images and validate using test set
    train_model(35),
    train_model(40),
    train_model(45),
    train_model(50)
][2:]

# plot precision score depending on image size
plt.plot([info[0] for info in models_info], [info[1] for info in models_info], color='orange')
plt.xlabel("Image sizes")
plt.ylabel("Accuracy score")
plt.title("Accuracy score depending on image size")
plt.show()

# plot model fit time depending on image size
plt.plot([info[0] for info in models_info], [info[2] for info in models_info], color='blue')
plt.xlabel("Image sizes")
plt.ylabel("Model fit time")
plt.title("Model fit time depending on image size")
plt.show()
