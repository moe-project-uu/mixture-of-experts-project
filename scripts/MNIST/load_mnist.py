import numpy as np
import imageio
import glob
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
    # create list of image objects
    test_images = []
    test_labels = []    
    
    print("Retrieving test images")
    for label in range(NUM_LABELS):
        for image_path in glob.glob("../../data/MNIST/Test/" + str(label) + "/*.png"):
            print("adding image")
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
    for label in range(NUM_LABELS):
        for image_path in glob.glob("../../data/MNIST/Train/" + str(label) + "/*.png"):
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
