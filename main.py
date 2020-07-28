from keras_segmentation.models.unet import unet

model = unet(n_classes=2, input_height=512, input_width=512)

model.train(
    train_images="RecorteCercospora/Imagem/",
    train_anotations="RecorteCercospora/Mascara/",
    checkpoints_pth="Checkpoints",
    epochs=100
)


