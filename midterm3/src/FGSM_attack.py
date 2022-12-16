import numpy as np
from matplotlib import pyplot as plt
from keras.models import load_model
from CNN import load_dataset, prep_pixels
from FGSM import generate_image_adversary

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# load dataset
_, _, testX, testY = load_dataset()
# prepare pixel data
testX = prep_pixels(testX)

model = load_model('final_model.h5')

# loop over a sample of our testing images
for i in np.random.choice(np.arange(0, len(testX)), size=(10,)):
    # grab the current image and label
    image = testX[i]
    label = testY[i]
    # generate an image adversary for the current image and make a prediction on the adversary
    adversary = generate_image_adversary(model, image.reshape(1, 32, 32, 3), label, eps=0.1)
    plt.imshow(image.reshape(1, 32, 32, 3)[0])
    plt.show()
    plt.imshow((adversary[0] * 255).astype(np.uint8))
    plt.show()
    pred = model.predict(adversary)
    classes_x = np.argmax(pred, axis=1)
    print("Model prediction on the adversary: ",classes_x)
    pred2 = model.predict(image.reshape(1, 32, 32, 3))
    classes_x_2 = np.argmax(pred2, axis=1)
    print("Model prediction on the original image: ",classes_x_2)
    label_real = np.argmax(label)
    print("Target (original image): ",label_real)
    print("---")