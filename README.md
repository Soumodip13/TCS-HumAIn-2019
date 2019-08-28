# TCS-HumAIn 2019
## Vehicle Number Plate detection ------> Identify the license place in the image and do an OCR to extract the characters from the detected license plate

The dataset was given in a json file named Indian_Number_plates.json(https://github.com/Soumodip13/TCS-HumAIn-2019/blob/master/Indian_Number_plates.json) and I wrote a Jupyter Notebook script to download the data and a script to rename it . Both of the files are in Jupyter Notebooks folder (https://github.com/Soumodip13/TCS-HumAIn-2019/tree/master/Jupyter%20Notebooks).

#### **Licenseplates** folder contains all the images that has been used in this project.

#### **'train'** folder contains 20x20 images for every alphabet except 'I' and 'O' because these two share similarities with '1' and '0' respectively. Images from this folder is used to train the Support Vector Classifier(https://github.com/Soumodip13/TCS-HumAIn-2019/blob/master/models/svc/svc.pkl) that is in the 'models/svc' folder.
