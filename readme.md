# Deep neural networks and Neuroscience

## Papers

- Critical Learning Periods (Achille et al.)
-

## Explicación de código

### Colabs
Hay tres archivos ipynb

***no_blur_all_conv_training.ipynb:*** Se entrena la AllConvNet(3) por 330 épocas con CIFAR100. Cada 10 épocas de entrenamiento, se guarda en drive la red, esto para luego tener distintas versiones del mismo modelo, entrenado hasta cierto número de épocas

***Achille_blurring_experiment.ipynb:*** Busca replicar el experimento realizado en el paper Critical Learning Periods, con la diferencia de que se utiliza BatchNormalization en la arquitectura. Se utilizan dos datasets en el experimento, los cuales son CIFAR100 y una versión borrosa de CIFAR100, generada a través de enpequeñecer a 8x8 y regresar a 32x32 cada imagen. Previamente se tiene la red entrenada con imágenes normales de CIFAR100 hasta 300 épocas, guardada cada 10 épocas generadas en no_blur_all_conv_training.ipynb. Se entrena la red durante 40 épocas con imágenes borrosas a partir de cada una de las redes preentrenadas

***High_frec_achille.ipynb:*** Imita lo realizado en el experimento de Critical Learning Periods, pero en vez de utilizar una versión borrosa de CIFAR100, se generan imágenes de alta frecuencia a partir de 

### Clases:
***NormalizeNegativeImages:***
    Utilizada para garantizar que solo haya valores positivos de pixeles de imagenes y que se encuentren en el rango [0, 1].

### Transformaciones
