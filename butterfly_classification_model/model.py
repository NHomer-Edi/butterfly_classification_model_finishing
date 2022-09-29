import os
import requests
import numpy as np
from skimage.transform import resize
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
# prevent annoying tensorflow warning

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import warnings
warnings.simplefilter("ignore")

CLASS_LABELS = ['ADONIS',
 'AFRICAN_GIANT_SWALLOWTAIL',
 'AMERICAN_SNOOT',
 'AN_88',
 'APPOLLO',
 'ATALA',
 'BANDED_ORANGE_HELICONIAN',
 'BANDED_PEACOCK',
 'BECKERS_WHITE',
 'BLACK_HAIRSTREAK',
 'BLUE_MORPHO',
 'BLUE_SPOTTED_CROW',
 'BROWN_SIPROETA',
 'CABBAGE_WHITE',
 'CAIRNS_BIRDWING',
 'CHECQUERED_SKIPPER',
 'CHESTNUT',
 'CLEOPATRA',
 'CLODIUS_PARNASSIAN',
 'CLOUDED_SULPHUR',
 'COMMON_BANDED_AWL',
 'COMMON_WOOD-NYMPH',
 'COPPER_TAIL',
 'CRECENT',
 'CRIMSON_PATCH',
 'DANAID_EGGFLY',
 'EASTERN_COMA',
 'EASTERN_DAPPLE_WHITE',
 'EASTERN_PINE_ELFIN',
 'ELBOWED_PIERROT',
 'GOLD_BANDED',
 'GREAT_EGGFLY',
 'GREAT_JAY',
 'GREEN_CELLED_CATTLEHEART',
 'GREY_HAIRSTREAK',
 'INDRA_SWALLOW',
 'IPHICLUS_SISTER',
 'JULIA',
 'LARGE_MARBLE',
 'MALACHITE',
 'MANGROVE_SKIPPER',
 'MESTRA',
 'METALMARK',
 'MILBERTS_TORTOISESHELL',
 'MONARCH',
 'MOURNING_CLOAK',
 'ORANGE_OAKLEAF',
 'ORANGE_TIP',
 'ORCHARD_SWALLOW',
 'PAINTED_LADY',
 'PAPER_KITE',
 'PEACOCK',
 'PINE_WHITE',
 'PIPEVINE_SWALLOW',
 'POPINJAY',
 'PURPLE_HAIRSTREAK',
 'PURPLISH_COPPER',
 'QUESTION_MARK',
 'RED_ADMIRAL',
 'RED_CRACKER',
 'RED_POSTMAN',
 'RED_SPOTTED_PURPLE',
 'SCARCE_SWALLOW',
 'SILVER_SPOT_SKIPPER',
 'SLEEPY_ORANGE',
 'SOOTYWING',
 'SOUTHERN_DOGFACE',
 'STRAITED_QUEEN',
 'TROPICAL_LEAFWING',
 'TWO_BARRED_FLASHER',
 'ULYSES',
 'VICEROY',
 'WOOD_SATYR',
 'YELLOW_SWALLOW_TAIL',
 'ZEBRA_LONG_WING']


def load_model_with_weights(url):
    ### TODO ####
    ### Need to download the weights file from url, if it's not already
    ### present, and put the downloaded filename into a variable
    ### called weights_filename
    model = tf.keras.models.load_model(url)
    return model

def preprocess_image(image):
    """
    Ensure that an input image is the correct size, and
    has the expected shape, to be used by the predict function

    parameters
    ----------
    image: np.ndarray, shape(npix_x,npix_y,3)

    returns
    -------
    image: np.ndarray, shape(None, 150, 150, 3)
    """
    image = resize(image, (300, 300),
                   preserve_range=True,
                   anti_aliasing=True)
    image = np.expand_dims(image, 0)
    return image


class efficientNetB3:
    ### Add a constructor to this class that calls the function
    ### to download the model weights, load the model, and assign
    ### to self.model
    def __init__(self, url):
        self.model = load_model_with_weights(url)

    def predict(self, image: np.ndarray):
        ### make sure the image is the correct size, and has
        ### the dimensions expected by the model.
        image = preprocess_image(image)
        result = self.model.predict(image)
        
        ### Find the highest weight, and, using the list of CLASS_LABELS
        ### get the corresponding class name.
        class_name = CLASS_LABELS[np.argmax(result)]
        return class_name



if __name__ == "__main__":
    pass
