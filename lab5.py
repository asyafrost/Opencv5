import cv2
import numpy as np
from datetime import timedelta
import os
import sys
from matplotlib import pyplot as plt
from tkinter import *
from tkinter.ttk import Entry
from PIL import *
from tkinter import ttk 
import pytesseract
import pyscreenshot as ImageGrab
import pyautogui
import time
import win32api

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'


def Face():

# создать новый объект камеру
    cap = cv2.VideoCapture(0)
    # инициализировать поиск лица (по умолчанию каскад Хаара)
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    while True:
        # чтение изображения с камеры
        _, image = cap.read()
        # преобразование к оттенкам серого
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # обнаружение лиц на фотографии
        faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)
        # для каждого обнаруженного лица нарисовать синий квадрат
        for x, y, width, height in faces:
            cv2.rectangle(image, (x, y), (x + width, y + height), color=(0, 255, 43), thickness=2)
        cv2.imshow("image", image)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()




def Text():
        
    img = cv2.imread("test1.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU |
                                          cv2.THRESH_BINARY_INV)
    cv2.imwrite('threshold_image.jpg',thresh1)

    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12))

    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 3)
    cv2.imwrite('dilation_image.jpg',dilation)

    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    im2 = img.copy()

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Рисуем ограничительную рамку на текстовой области
        rect=cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Обрезаем область ограничительной рамки
        cropped = im2[y:y + h, x:x + w]
    
        cv2.imshow('rectanglebox.jpg',rect)
        cv2.waitKey(0)



    string = pytesseract.image_to_string(img)
    print(string)
    #data = pytesseract.image_to_data(img)

    
    #cv2.imshow("Image", img)




def Menu():
    window = Tk()

    
    window.title("Menu")

    w = window.winfo_screenwidth()
    h = window.winfo_screenheight()
    w = w//2 # середина экрана
    h = h//2 
    w = w - 200 # смещение от середины
    h = h - 200
    window.geometry('400x300+{}+{}'.format(w, h))
    window.configure(bg='#bb85f3')

    btn = Button(window, text="Нахождение лица ", padx=5, pady=5, command =Face, bg='#eec6ea')  
    btn.pack(anchor="center", padx=20, pady=10)

    btn1 = Button(window, text="Нахождение текста", padx=5, pady=5, command = Text, bg='#eec6ea')  
    btn1.pack(anchor="center", padx=20, pady=10)

    btn2 = Button(window, text="Выход", padx=5, pady=5, command = exit, bg='#eec6ea')  
    btn2.pack(anchor="center", padx=20, pady=10)



    window.mainloop()

Menu()