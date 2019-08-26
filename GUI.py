import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tool
import cv2


srcPath = ""
pictureName = ""
tempPicturePath = "./tempPicture/"
outputPicturePath = "./outputPicture/"
baiduAIAccessToken = '24.5d22f3f43aaeee80c6bcaafdb72b093c.2592000.1533119124.282335-11475678'


def resize( w_box, h_box, pil_image):
  w, h = pil_image.size
  f1 = 1.0*w_box/w
  f2 = 1.0*h_box/h
  factor = min([f1, f2])
  width = int(w*factor)
  height = int(h*factor)
  return pil_image.resize((width, height), Image.ANTIALIAS)


window = tk.Tk()
window.title('Number Plate Recognition System')
window.geometry('1000x700')



# chsobin's logo
#canvas = tk.Canvas(window, width=150, height=150)
#image_file = tk.PhotoImage(file='icon.png')
#image = canvas.create_image(0,0, anchor='nw', image=image_file)
#canvas.place(x=20, y=20)


canvas_pic = tk.Canvas(window, width=700, height=600)
canvas_pic.pack(side='right')

def printcoords():
    File = filedialog.askopenfilename(parent=window, title='Please select an image')
    global srcPath
    srcPath=File
    src = Image.open(File)
    resizePic = resize(700, 600, src)
    filename = ImageTk.PhotoImage(resizePic)
    canvas_pic.image = filename  # <--- keep reference of your image
    canvas_pic.create_image(350, 300, image=filename)

btn_choosePic = tk.Button(window, text='Select Image', command=printcoords)
btn_choosePic.place(x=40, y=190)


canvas_plate = tk.Canvas(window, width=200, height=50)
canvas_plate.place(x=10, y=320)

def location():
    global tempPicturePath
    global outputPicturePath
    global pictureName
    pictureName = srcPath.split('/')[-1]
    src = cv2.imread(srcPath)
    afterPreProcess = tool.preProcess(src)
    pic, contours, hierarchy = cv2.findContours(afterPreProcess, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rect = tool.getRect(contours)
    if len(rect) != 1:
        print('The minimum enclosing rectangle has：', len(rect), 'System not recognized')
    else:
        Rotation_M = tool.getRotation_M(rect[0])
        rotatePic = tool.rotateSrc(src, Rotation_M)
        cv2.imwrite(tempPicturePath + pictureName, rotatePic)
        src = cv2.imread(tempPicturePath + pictureName)
        afterPreProcess = tool.preProcess(src)
        pic, contours, hierarchy = cv2.findContours(afterPreProcess, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rect = tool.getRect(contours)
        if len(rect) != 1:
            print('The minimum enclosing rectangle has',len(rect) ,'System not recognized')
        else:
            result = tool.cutSrcByRect(src, rect[0], -5)
            cv2.imwrite(outputPicturePath + pictureName, result)
            print(pictureName, "Successfully positioned the license plate")
            src = Image.open(outputPicturePath + pictureName)
            resizePic = resize(200, 50, src)
            filename = ImageTk.PhotoImage(resizePic)
            canvas_plate.image = filename
            canvas_plate.create_image(100, 25, image=filename)

btn_location = tk.Button(window, text='Start positioning', command=location)
btn_location.place(x=40, y=250)
tk.Label(window, text='License plate location result: ', font='18', width=18).place(x=20, y= 300)




var_returnMsg = tk.StringVar()
var_returnMsg.set(" ")
l = tk.Label(window,
    textvariable=var_returnMsg,
    font=('Arial', 12),
    width=18, height=4
    )
l.place(x=20, y=470)

def recognition():
    global baiduAIAccessToken
    global outputPicturePath
    global pictureName
    result = tool.getCarNumberByBaiDuAI(baiduAIAccessToken, outputPicturePath+pictureName)
    if 'error_code' in result:
        print("--------  " + outputPicturePath + pictureName)
        print("!!!!错误：百度ai识别失败，错误信息为", result)
    else:
        print("--------  " + outputPicturePath + pictureName)
        print("颜色：" + result['words_result']['color'])
        print("号码：" + result['words_result']['number'])
        var_returnMsg.set(result['words_result']['color'] + "  " + result['words_result']['number'])


btn_recognition = tk.Button(window, text='Start identifying', command=recognition)
btn_recognition.place(x=40, y=400)
tk.Label(window, text='License plate recognition result: ', font='18').place(x=20, y= 450)

window.mainloop()

