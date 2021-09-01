# COVID Chest CTs

The purpose of this project was to recreate results from the following paper:

+ Loey, M., Manogaran, G. & Khalifa, N.E.M. A deep transfer learning model with classical data augmentation and CGAN to detect COVID-19 from chest CT radiography digital images. Neural Comput & Applic (2020). https://doi.org/10.1007/s00521-020-05437-x

As well as to start the process of expanding the artificially generated dataset of COVID Chest CT scans, with use of a variational autoncoder instead of the original CGAN used in their paper. Results seem very promising, but hyperparameter tuning and more training time for the VAE is needed.

## Dataset

https://www.kaggle.com/mloey1/covid19-chest-ct-image-augmentation-gan-dataset


## Notebooks

These pyhon notebooks contain the following:

* Notebook 1) Code for a simple re-implementation of ResNet50 transfer learning on the base dataset.
* Notebook 2) A simple implemenetation of a Variational Autoencoder. 
* Notebook 2b) The same as notebook 2, but this was used for training on Kaggle and includes output cells.
* Notebook 3) Sampling from the VAE model utilizing the testing dataset.

## Authors

Alexander Kahanek -- GitHub @ Alexander-Kahanek

Luke Strgar -- GitHub @ lstrgar
