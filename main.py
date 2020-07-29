from keras_segmentation.models.unet import unet
import matplotlib.pylab  as plt
import matplotlib.pyplot  as pltimg
import cv2 as cv
import numpy as np
import keras
import  tensorflow as tf



####caminho absoluto das imagens de treinamento e teste
imagesForTrain = \
    "C:\\Users\\alexa\\OneDrive - ufu.br\\ProjetoWesley\\UNet-for-cercospora-problem\\RecorteCercospora\\Imagem"
maskForTrain = \
    "C:\\Users\\alexa\\OneDrive - ufu.br\\ProjetoWesley\\UNet-for-cercospora-problem\\RecorteCercospora\\Mascara"
imagesForValidation = \
    "C:\\Users\\alexa\\OneDrive - ufu.br\\ProjetoWesley\\UNet-for-cercospora-problem\\RecorteCercospora\\ImagemTest"
maskForValidation = \
    "C:\\Users\\alexa\\OneDrive - ufu.br\\ProjetoWesley\\UNet-for-cercospora-problem\\RecorteCercospora\\MascaraTest"

#Checkpoint dos pesos
checkpoint_path =  "UnetWeights"

#####Chama o modelo
model = unet(n_classes=2, input_height=512, input_width=512)

#####Função de treinamento do modelo#####
model.train(
    train_images=imagesForTrain,
    train_annotations=maskForTrain,
    #val_images=imagesForValidation,
    n_classes=2,
    verify_dataset=False,
    input_height=512, input_width=512,
    #val_annotations=maskForValidation,
    epochs=1,
    checkpoints_path=checkpoint_path
 )


#Função pra salvar os pesos do modelo, se não setar o formato será em TF se setar em h5
model.save_weights(filepath=checkpoint_path, save_format="h5")
#Função para carregar os pesos para o modelo
model.load_weights(checkpoint_path)


###Fução para salvar o modelo
## model.save("Mymodel.h5")

print("Evaluate da coisa")
print(model.evaluate)


print("quais metricas:")
print(model.metrics_names)
print("Metricas")
print(model.metrics)

##model.summary()

# imageInputFroPrediction = "C:\\Users\\alexa\\OneDrive - ufu.br\\ProjetoWesley\\UNet-for-cercospora-problem\\predictResults\\ImagePredict\\sample_0961.png"
# fileTosaveOutput = "C:\\Users\\alexa\\OneDrive - ufu.br\\ProjetoWesley\\UNet-for-cercospora-problem\\predictResults\\predictResults"

##print(model.evaluate_segmentation( inp_images_dir=imagesForValidation  , annotations_dir=maskForValidation ))


saida = model.predict_segmentation(
    inp=imageInputFroPrediction,
    out_fname=
    "C:\\Users\\alexa\\OneDrive - ufu.br\\ProjetoWesley\\UNet-for-cercospora-problem\\predictResults\\predictResults\\out.png"
)

##model.compile()
##USar a ideia desse git https://github.com/Aiwiscal/ECG_UNet/blob/master/train.py
##keras.callbacks.ModelCheckpoint
print(type(saida))
#cv.imshow("buga", saida)


#### Função vista para pegar do checkpoint  funciona mas precisa ser salvo um arquivo .h5
# import os
# from keras_segmentation.models.all_models import model_from_name
# from keras_segmentation.train import find_latest_checkpoint
# import json
#
#
# def tem_sim(checkpoints_path):
#     assert (os.path.isfile(checkpoints_path)), "Checkpoint not found."
#     model.load_weights(checkpoints_path )
#     return model
#
#
# def model_from_checkpoint_path(checkpoints_path):
#     assert (os.path.isfile(checkpoints_path + "_config.json")), "Checkpoint not found."
#     model_config = json.loads(open(checkpoints_path + "_config.json", "r").read())
#     latest_weights = find_latest_checkpoint(checkpoints_path)
#     assert (not latest_weights is None), "Checkpoint not found."
#     model = model_from_name[model_config['model_class']](model_config['n_classes'],
#                                                          input_height=model_config['input_height'],
#                                                          input_width=model_config['input_width'])
#     print("loaded weights ", latest_weights)
#     model.load_weights(latest_weights)
#     return model
##Exemplo de como funcionaria
#modelFrom = model_from_checkpoint_path(checkpoint_path)



##pltimg.imshow("Bora ver",saida)


###Como abrir imagem com o OPENCV############

## img = cv.imread(maskForTrain+"\\sample_0001.png")
##img = cv.imread(imageInputFroPrediction)
##
##print(type(img))
##cv.imshow("mask", img)
# cv.imshow("mask2", imgt)
# print(type(img))

# img = cv.imread("1.png")
# imgt = cv.imread("1t.png")


