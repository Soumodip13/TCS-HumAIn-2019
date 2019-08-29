# TCS-HumAIn 2019
## Vehicle Number Plate detection ------> Identify the license place in the image and do an OCR to extract the characters from the detected license plate :octocat:

### According to my perception of this problem Number Plate Recognition System has four major steps to follow to find out the vehicle number from the number/license plate.
- **Number Plate Detection:** This is the first and probably themost important stage of the system. It is at this stage that the position of the license plate is determined. The input at this stage is an image of the vehicle and the output is the license plate.
- **Character Segmentation:** It’s at this stage the characters onthe license plate are mapped out and segmented into individual images.
- **Character Recognition:** This is where we wrap things up. The characters earlier segmented are identified here. We’ll be using machine learning for this.
- **Prediction:** This is where the recognised characters will be
predicted on screen or on the image itself.

The dataset was given in a json file named Indian_Number_plates.json(https://github.com/Soumodip13/TCS-HumAIn-2019/blob/master/Indian_Number_plates.json) and I wrote a Jupyter Notebook script to download the data and a script to rename it . Both of the files are in Jupyter Notebooks folder (https://github.com/Soumodip13/TCS-HumAIn-2019/tree/master/Jupyter%20Notebooks).

#### **Licenseplates** folder contains all the images that has been used in this project.

#### **'train'** folder contains 20x20 images for every alphabet except 'I' and 'O' because these two share similarities with '1' and '0' respectively. Images from this folder is used to train the Support Vector Classifier(https://github.com/Soumodip13/TCS-HumAIn-2019/blob/master/models/svc/svc.pkl) that is in the 'models/svc' folder.

## Example Image
## ![Example Image](/Licenseplates/lpr72.jpeg)
#### The detection algorithm detects and isolates the number plate from aregular image of a vehicle, which is used by a machine learning algorithm as uniform inputs to recognise the characters in the number plate.

In **train.py**
```
svc_model = SVC(kernel='linear', probability=True)
cross_validation(svc_model, 4, image_data, target_data)
svc_model.fit(image_data, target_data)
save_directory = os.path.join(current_dir, 'models/svc/')
if not os.path.exists(save_directory):
    os.makedirs(save_directory)
joblib.dump(svc_model, save_directory+'/svc.pkl')
```
**this piece of code declares a support vector classifier of linear kernel and trains the model with the 20x20 images of different characters in the train folder.The trained model gets saved in the models/svc folder to simplify the re-usability.:four: folds are used here meaning e are performing a 4-fold cross validation and it will divide the dataset into 4 and use 1/4 of it for testing and the remaining 3/4 for the training**

- CharRecog.py 
- Chars.py
- LicensePlate.py helps to map the license plate from the possible contours.
- Operations.py carries out image processings that has to be done.
- TracePlate.py
- train.py carries out the svc model training.
- Plate_OCR.py performs recognition of characters in a Number Plate
- Run.py serves as our main():heavy_check_mark:

```    
for count in range(72, 73):
        try:
            print(f"Starting {count}..")
            print("DETECTING PLATE . . .")
 ```
 In **Run.py**  this range implies as the index of the image to be performed Number Plate Recognition on.**range(72, 73)** means that the recognition will be performed on **lpr72.jpeg**.To perform recogniton on every image file do use **range(0,237)** or **range(237)**.
 
 **Images below shows the result**
 
 
 ![Grayscale Image](/steps/1a.JPG)
 ![Binary Image](/steps/1b.JPG)
 ![Canny edges](/steps/2a.JPG)
 ![Char COntours](/steps/2b.JPG)
 ![Refined Image](/steps/3.JPG)
 ![Number Plate like contours](/steps/4a.JPG)
 ![](/steps/4b.JPG)
 ![](/steps/5b1.JPG)
 ![](/steps/5b2.JPG)
![](/steps/5c.JPG) 
![](/steps/5d.JPG)
![](/steps/10.JPG) 
![](/steps/5a.JPG)
![](/steps/5b.JPG)
![](/steps/11.JPG)
![](/steps/12.JPG)
![](/steps/14.JPG)

**Output**
![](/Detectedplates/lpr72.jpeg)

**Model Training Result**
![](/trainingresult.JPG)

**GUI Layout**(code isn't complete)
![](/GUI.jpg)


## File Description
OS : Windows 8.1 64 bit 

Processor:Intel(R) Celeron(R) CPU 1000M   1.80 Ghz

RAM: 6.00 GB

GPU : None

Python : 3.7
## Python Packages used
![](/packagesused.JPG)

:octocat:
