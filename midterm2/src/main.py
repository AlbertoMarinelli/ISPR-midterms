import numpy as np
from matplotlib import pyplot as plt
from load_dataset import load_data
from rbm import RBM

imgs, _, _, _ = load_data(directory="./datasets/")
rbm = RBM(v_units=len(imgs[0]), h_units=20, path_directory='./datasets/')
model_path = 'model_14-04-22_11-55.pickle'
train_model = False

# TRAIN NEW MODEL
if (train_model):
    rbm.fit(epochs=1, lr=0.1, k=1, batch_size=1, save=True, fit_cl=True, save_weights=False)

# LOAD MODEL
else:
    rbm.load_weights(model_path)

    rbm.fit_classifier(load_boltz_weights=True, w_path=model_path) #Train
    rbm.test_classifier() #Test

    # Confusion matrix
    rbm.confusion_matrix()

    # Plot a reconstruction for each digit
    indexes = []
    curr = 0
    while curr < 10:
        for i, label in enumerate(rbm.tr_labels):
            if label == curr:
                indexes.append(i)
                curr += 1
                break
    fig, ax = plt.subplots(2, 10, figsize=(5, 2))
    for i in range(20):
        if i < 10:
            ax[0, i].imshow(np.reshape(rbm.tr_imgs[indexes[i]], newshape=(28, 28)))
            ax[0, i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        else:
            v_sample = np.random.binomial(n=1, p=rbm.tr_imgs[indexes[i - 10]], size=len(rbm.tr_imgs[indexes[i - 10]]))
            _, h_sample = rbm.hidden_visible(v_sample)
            v_probs, _ = rbm.visible_hidden(h_sample)
            ax[1, i - 10].imshow(np.reshape(v_probs, newshape=(28, 28)))
            ax[1, i - 10].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.show()
