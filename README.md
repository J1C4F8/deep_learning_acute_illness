# Facial Analysis for classification of acutely ill individuals
### Identifying whether a patient appears sick using Convolutional Neural Networks.

<p align="center">
  <img height="300" src="https://github.com/vandrw/icu_binary/blob/master/documentation/sick_features.jpg">
</p>

<p align="center">
  <b>For in-depth information regarding the project, please refer to the <a href="https://github.com/vandrw/icu_binary/blob/master/documentation/Report.pdf">paper</a></b>.
</p>

## How to run?
To extract the features, first place the images in their corresponding directories (e.g. _data/unparsed/sick_ for images representing sick individuals) and run the command `make create-data` in a terminal.

Once the data is created, `make train-individual` or `make train-stacked` can be used to train all the individual networks (i.e. eyes, nose, mouth and skin) or, respectively, a stacked ensemble that parses each feature at once.

If one desires to remove the created data, `make clean-data` can be used. Furthermore, `make clean-results` will remove any saved models, histories and plots generated.

For python environment details, please check __environment.py__.

### Project Structure
* **augment**: folder containing the code of a neural style transfer network
* **categorization**: folder containing a convolutional neural network that categorizes the images
* **data**: folder containing the collected data set

## Potential Data Sets for Augmentation
* [SoF Dataset](https://sites.google.com/view/sof-dataset)
* [IST-EURECOM Light Field Face Database](http://www.img.lx.it.pt/LFFD/)
* [CVL Face Database](http://www.lrv.fri.uni-lj.si/facedb.html)
* [Chicago Faces Dataset](https://chicagofaces.org/default/download/)
* [YMU and VMU](http://www.antitza.com/makeup-datasets.html)

### References for previous datasets
A. Sepas-Moghaddam, V. Chiesa, P.L. Correia, F. Pereira, J. Dugelay, “The IST-EURECOM Light Field Face Database”, International Workshop on Biometrics and Forensics, IWBF 2017, Coventry, UK, April 2017

Mahmoud Afifi and Abdelrahman Abdelhamed, "AFIF4: Deep gender classification based on an AdaBoost-based fusion of isolated facial features and foggy faces". Journal of Visual Communication and Image Representation, 2019. 

PEER, Peter, EMERŠIČ, Žiga, BULE, Jernej, ŽGANEC GROS, Jerneja, ŠTRUC, Vitomir. Strategies for exploiting independent cloud implementations of biometric experts in multibiometric scenarios. Mathematical problems in engineering, vol. 2014, pp. 1-15, 2014.

