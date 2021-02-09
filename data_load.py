import imgaug.augmenters as iaa
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import os
import numpy as np
import random

def generator(features, batch_size):
    seq = iaa.Sequential([
        iaa.OneOf([
            iaa.AverageBlur(k=((3, 5), (5, 7))),
            iaa.Rotate((-45,45)),
            iaa.Affine(scale=(0.5, 0.95)),    
            iaa.Multiply((0.50, 1.1))
            , iaa.Fliplr(1.0)
            , iaa.Flipud(1.0)
            , iaa.GammaContrast((0.5, 2.0))
            , iaa.WithBrightnessChannels(iaa.Add((-50, 50)), from_colorspace=iaa.CSPACE_BGR)
            ,iaa.WithHueAndSaturation(iaa.WithChannels(0, iaa.Add((0, 50))))
            #,iaa.BlendAlphaRegularGrid(nb_rows=(4, 6), nb_cols=(1, 4),
            #                        foreground=iaa.Multiply(0.0))
            #,iaa.Cartoon()
            ,iaa.Cutout(nb_iterations=(1, 3), size=0.2, squared=False, cval=0)
            ,iaa.Affine(shear=(-48, 48))
            ,iaa.Affine(translate_px={"x": (-42, 42), "y": (-36, 36)})
            ,iaa.KeepSizeByResize(iaa.Resize({"height": (0.70, 0.90), "width": (0.70, 0.90)}))
            ,iaa.CropAndPad(percent=(-0.2, 0.2))
            #,iaa.PiecewiseAffine(scale=(0.01, 0.05))
            ,iaa.PerspectiveTransform(scale=(0.01, 0.1))
            #,iaa.WithPolarWarping(iaa.CropAndPad(percent=(-0.1, 0.1)))
            #,iaa.ElasticTransformation(alpha=(0, 3.0), sigma=0.5)
           ])
        #,iaa.SaveDebugImageEveryNBatches(folder_path, 100)    
    ], random_order=True)

    
    
    while True:
        # Fill arrays of batch size with augmented data taken randomly from full passed arrays
        indexes = random.sample(range(len(features)), batch_size)
      
        # Transform X1 and X2
        x_aug_1 = seq(images =features[indexes])
        x_aug_2 = seq(images =features[indexes])
        yield np.array(x_aug_1), np.array(x_aug_2)

# import image data and combine labels
def load_unlabeled_data(image_size, image_dir, extension = '.JPG'):
    X = []
    for root, folder, files in os.walk(image_dir):
        #print(files)
        for f in files:
            #print(f)
            if f.endswith(extension):
                #print(root, folder, f)
                img = load_img(f'{root}/{f}', target_size=(image_size,image_size,3))
                img_array = img_to_array(img, dtype='uint8')
                X.append(img_array)
    return np.array(X, dtype=np.uint8)