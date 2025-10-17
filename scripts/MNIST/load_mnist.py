import numpy as np
import imageio
import glob
import os
from tqdm import tqdm

#taken from the Deeplearning course at Uppsala University
def load_mnist(testOnly = False):
    # Loads the MNIST dataset from png images
    #
    # Return
    # X_train - Training input 
    # Y_train - Training output (one-hot encoded)
    # X_test - Test input
    # Y_test - Test output (one-hot encoded)

    # Each of them uses rows as data point dimension.
 
    NUM_LABELS = 10    
    #need this for pathing reasons
    BASE_DIR = os.path.dirname(__file__)  # directory of load_mnist.py    
    # create list of image objects
    test_images = []
    test_labels = []    
    
    print("Retrieving test images")
    for label in tqdm(range(NUM_LABELS)):
        #pathing so it works anywhere
        test_dir = os.path.join(BASE_DIR, "../../data/MNIST/Test", str(label), "*.png")
        for image_path in tqdm(glob.glob(test_dir)):
            image = imageio.imread(image_path)
            test_images.append(image)
            letter = [0 for _ in range(0,NUM_LABELS)]    
            letter[label] = 1
            test_labels.append(letter)  
            
    # create list of image objects
    train_images = []
    train_labels = []    
    if testOnly:
        return None, None, np.array(test_images).reshape(-1,784)/255.0, np.array(test_labels)
    print("Retrieving train images")
    for label in tqdm(range(NUM_LABELS)):
        #pathing so it works anywhere
        train_dir = os.path.join(BASE_DIR, "../../data/MNIST/Train", str(label), "*.png")
        for image_path in tqdm(glob.glob(train_dir)):
            image = imageio.imread(image_path)
            train_images.append(image)
            letter = [0 for _ in range(0,NUM_LABELS)]    
            letter[label] = 1
            train_labels.append(letter)                  
            
    X_train= np.array(train_images).reshape(-1,784)/255.0
    Y_train= np.array(train_labels)
    X_test= np.array(test_images).reshape(-1,784)/255.0
    Y_test= np.array(test_labels)
    
    return X_train, Y_train, X_test, Y_test
