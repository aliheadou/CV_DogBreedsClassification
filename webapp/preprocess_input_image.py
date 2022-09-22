import cv2
import tensorflow as tf
import tensorflow.keras as keras

import numpy as np

# preprocessing
def prep_images(x):
    dst_image=[]
    for i in range(x.shape[0]):
        img = x.numpy()[i].astype("uint8")
        img_YUV = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
        # Equalization
        img_YUV[:,:,0] = cv2.equalizeHist(img_YUV[:,:,0])
        img_equ = cv2.cvtColor(img_YUV, cv2.COLOR_YUV2RGB)
        # Apply non-local means filter on test img
        dst_img = cv2.fastNlMeansDenoisingColored(
            src=img_equ,
            dst=None,
            h=10,
            hColor=10,
            templateWindowSize=7,
            searchWindowSize=21)
        dst_image.append(dst_img)
    out_img = tf.convert_to_tensor(np.array(dst_image), dtype=tf.float32)
    return out_img

# rescale data
def data_rescale(img, images_shape):
    data_rescale_layer = tf.keras.Sequential([
        keras.layers.Input(shape=images_shape),
        keras.layers.experimental.preprocessing.Rescaling(
            1./(255),
            input_shape=images_shape
    )])
    return data_rescale_layer(img)

# Class names
def class_names(pos):
    class_names = [
        'Afghan_hound', 'African_hunting_dog', 'Airedale', 'American_Staffordshire_terrier', 'Appenzeller', 'Australian_terrier',
        'Bedlington_terrier', 'Bernese_mountain_dog', 'Blenheim_spaniel', 'Border_collie', 'Border_terrier', 'Boston_bull', 
        'Bouvier_des_Flandres', 'Brabancon_griffon', 'Brittany_spaniel', 'Cardigan', 'Chesapeake_Bay_retriever', 'Chihuahua', 
        'Dandie_Dinmont', 'Doberman', 'English_foxhound', 'English_setter', 'English_springer', 'EntleBucher', 'Eskimo_dog',
        'French_bulldog', 'German_shepherd', 'German_short', 'Gordon_setter', 'Great_Dane', 'Great_Pyrenees', 'Greater_Swiss_Mountain_dog',
        'Ibizan_hound', 'Irish_setter', 'Irish_terrier', 'Irish_water_spaniel', 'Irish_wolfhound', 'Italian_greyhound', 'Japanese_spaniel',
        'Kerry_blue_terrier', 'Labrador_retriever', 'Lakeland_terrier', 'Leonberg', 'Lhasa', 'Maltese_dog', 'Mexican_hairless',
        'Newfoundland', 'Norfolk_terrier', 'Norwegian_elkhound', 'Norwich_terrier', 'Old_English_sheepdog', 'Pekinese', 'Pembroke',
        'Pomeranian', 'Rhodesian_ridgeback', 'Rottweiler', 'Saint_Bernard', 'Saluki', 'Samoyed', 'Scotch_terrier', 'Scottish_deerhound',
        'Sealyham_terrier', 'Shetland_sheepdog', 'Shih', 'Siberian_husky', 'Staffordshire_bullterrier', 'Sussex_spaniel', 'Tibetan_mastiff',
        'Tibetan_terrier', 'Walker_hound', 'Weimaraner', 'Welsh_springer_spaniel', 'West_Highland_white_terrier', 'Yorkshire_terrier',
        'affenpinscher', 'basenji', 'basset', 'beagle', 'black', 'bloodhound', 'bluetick', 'borzoi', 'boxer', 'briard', 'bull_mastiff',
        'cairn', 'chow', 'clumber', 'cocker_spaniel', 'collie', 'curly', 'dhole', 'dingo', 'flat', 'giant_schnauzer', 'golden_retriever',
        'groenendael', 'keeshond', 'kelpie', 'komondor', 'kuvasz', 'malamute', 'malinois', 'miniature_pinscher', 'miniature_poodle',
        'miniature_schnauzer', 'otterhound', 'papillon', 'pug', 'redbone', 'schipperke', 'silky_terrier', 'soft', 'standard_poodle', 
        'standard_schnauzer', 'toy_poodle', 'toy_terrier', 'vizsla', 'whippet', 'wire'
        ]

    return class_names[pos].split('_')