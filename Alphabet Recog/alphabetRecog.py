import numpy as np
from tensorflow.keras.models import load_model
from PIL import ImageTk, Image, ImageDraw
import PIL
import cv2
import tkinter as tk
from tkinter import *

model=load_model('alpha.h5')

def prediction():
    img=cv2.imread('image.png',0)
    img=cv2.bitwise_not(img)
    img=cv2.resize(img,(28,28))
    comparison = img == all_zeros
    equal_arrays = comparison.all()
    if equal_arrays:
        return "empty"
    else:
        img=img.reshape(1,28,28,1)
        img=img.astype('float32')
        img=img/255

        pred=model.predict(img)
        return pred
	
def paint(event):
    x1, y1 = (event.x - 10), (event.y - 10)
    x2, y2 = (event.x + 10), (event.y + 10)
    canvas.create_oval(x1, y1, x2, y2, fill="black",width=40)
    draw.line([x1, y1, x2, y2],fill="black",width=40)
    
def predictor():
    filename = "image.png"
    image1.save(filename)
    txt.configure(state="normal")
    txt.delete('1.0', END)
    txt.configure(state="disabled")
    pred=testing()
    if pred=="empty":
        txt.configure(state="normal")
        txt.insert(tk.INSERT,"Please Clear and Draw Alphabet")
        txt.configure(state="disabled")
    else:
        print('alphabet',classes[np.argmax(pred[0])])
        txt.configure(state="normal")
        txt.insert(tk.INSERT,"{}\nAccuracy: {}%".format(classes[np.argmax(pred[0])],round(pred[0][np.argmax(pred[0])]*100,3)))
        txt.configure(state="disabled")
    
def clear():
    canvas.delete('all')
    draw.rectangle((0, 0, 400, 400), fill=(255, 255, 255, 0))
    txt.configure(state="normal")
    txt.delete('1.0', END)
    txt.configure(state="disabled")
	
classes={0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',
          14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'}
width=400
height=400
white=(255, 255, 255)
all_zeros=np.zeros((28,28))

root = Tk()

root.resizable(0,0)
canvas = Canvas(root, width=width, height=height, bg='white')
canvas.pack()

image1 = PIL.Image.new("RGB", (width, height), white)
draw = ImageDraw.Draw(image1)

txt=tk.Text(root,bd=3,font='Helvetica',exportselection=0,
            padx=10,pady=10,height=2,width=25)

canvas.pack(expand=YES, fill=BOTH)
canvas.bind("<B1-Motion>", paint)

predBtn=Button(text="Predict Alphabet",command=predictor)
clearBtn=Button(text="Clear Canvas",command=clear)

predBtn.pack(fill=tk.X)
clearBtn.pack(fill=tk.X)
txt.pack(fill=tk.X)

root.title('Upper Case Alphabets')

root.mainloop()
