from keras_segmentation.models.unet import unet
import matplotlib as plt
import cv2 as cv
import numpy as np

imagesForTrain = \
    "C:\\Users\\alexa\\OneDrive - ufu.br\\ProjetoWesley\\UNet-for-cercospora-problem\\RecorteCercospora\\Imagem"
maskForTrain = \
    "C:\\Users\\alexa\\OneDrive - ufu.br\\ProjetoWesley\\UNet-for-cercospora-problem\\RecorteCercospora\\Mascara"
imagesForValidation = \
    "C:\\Users\\alexa\\OneDrive - ufu.br\\ProjetoWesley\\UNet-for-cercospora-problem\\RecorteCercospora\\ImagemTest"
maskForValidation = \
    "C:\\Users\\alexa\\OneDrive - ufu.br\\ProjetoWesley\\UNet-for-cercospora-problem\\RecorteCercospora\\MascaraTest"


model = unet(n_classes=2, input_height=512, input_width=512)

model.train(
    train_images=imagesForTrain,
    train_annotations=maskForTrain,
    val_images=imagesForValidation,
    n_classes=2,
    verify_dataset=False,
    input_height=512, input_width=512,
    val_annotations=maskForValidation,
    epochs=2
)

imageInputFroPrediction = \
    "C:\\Users\\alexa\\OneDrive - ufu.br\\ProjetoWesley\\UNet-for-cercospora-problem\\predictResults\\sample_0801.png"


saida = model = model.predict_segmentation(
    inp=imageInputFroPrediction,
    out_fname=
    "C:\\Users\\alexa\\OneDrive - ufu.br\\ProjetoWesley\\UNet-for-cercospora-problem\\predictResults\\predictResults\\out.png"
)

plt.imshow(saida)



###Como abrir imagem com o OPENCV############

# img = cv.imread(maskForTrain+"\\sample_0001.png")
#
#
# print(np.shape(img))
#cv.imshow("mask", img)
#cv.imshow("mask2", imgt)
#print(type(img))

#img = cv.imread("1.png")
#imgt = cv.imread("1t.png")