import cv2
import os
import sys

import Plate_OCR
import LicensePlate

SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_BLUE = (255.0, 0.0, 0.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

showSteps = True


def main():
    blnKNNTrainingSuccessful = Plate_OCR.loadKNNDataAndTrainKNN()

    if blnKNNTrainingSuccessful == False:
        print("\nerror: KNN traning was not successful\n")
        return

    for count in range(72, 73):
        try:
            print(f"Starting {count}..")
            print("DETECTING PLATE . . .")

            imgOriginalScene = cv2.imread(f"Licenseplates/lpr{count}.jpeg")

            if imgOriginalScene is None:
                print("\nerror: image not read from file \n\n")
                os.system("pause")
                continue

            listOfPossiblePlates = LicensePlate.detectPlatesInScene(imgOriginalScene)

            listOfPossiblePlates = Plate_OCR.detectCharsInPlates(listOfPossiblePlates)

            # cv2.imshow("imgOriginalScene", imgOriginalScene)

            if len(listOfPossiblePlates) == 0:
                print("\nno license plates were detected\n")
            else:

                listOfPossiblePlates.sort(key=lambda possiblePlate: len(possiblePlate.strChars), reverse=True)

                licPlate = listOfPossiblePlates[0]

                # cv2.imshow("imgPlate", licPlate.imgPlate)
                # cv2.imshow("imgThresh", licPlate.imgThresh)

                if len(licPlate.strChars) == 0:
                    print("\nno characters were detected\n\n")
                    continue

                drawRedRectangleAroundPlate(imgOriginalScene, licPlate)

                print(
                    "\nlicense plate read from image = " + licPlate.strChars + "\n")
                print("----------------------------------------")

                writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)

                # cv2.imshow("imgOriginalScene", imgOriginalScene)

                cv2.imwrite(f"Detectedplates/lpr{count}.jpeg", imgOriginalScene)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)


def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):
    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)

    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)


def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    ptCenterOfTextAreaX = 0
    ptCenterOfTextAreaY = 0

    ptLowerLeftTextOriginX = 0
    ptLowerLeftTextOriginY = 0

    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX
    fltFontScale = float(plateHeight) / 30.0
    intFontThickness = int(round(fltFontScale * 3))
    textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale,
                                         intFontThickness)

    ((intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight),
     fltCorrectionAngleInDeg) = licPlate.rrLocationOfPlateInScene

    intPlateCenterX = int(intPlateCenterX)
    intPlateCenterY = int(intPlateCenterY)

    ptCenterOfTextAreaX = int(intPlateCenterX)

    if intPlateCenterY < (sceneHeight * 0.75):
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(
            round(plateHeight * 1.6))
    else:
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(
            round(plateHeight * 1.6))

    textSizeWidth, textSizeHeight = textSize
    ptLowerLeftTextOriginX = int(
        ptCenterOfTextAreaX - (textSizeWidth / 2))
    ptLowerLeftTextOriginY = int(
        ptCenterOfTextAreaY + (textSizeHeight / 2))

    cv2.putText(imgOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace,
                fltFontScale, SCALAR_BLUE, intFontThickness)


if __name__ == "__main__":
    main()


















