import datetime
import math
import os
import shutil
import tkinter as tk
import tkinter.filedialog as fd
from datetime import date
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import sys
import random



class TkinterWin(tk.Tk):
    window = 0
    old_x, old_y = 0, 0
    rect_id = -1
    Y = -1;
    rlable = 0
    blable = 0
    glable = 0
    sredotk = 0
    intensitylable = 0
    positionlable = 0
    standotk = 0
    rgb_im = 0
    np_image = 0
    canvas = 0
    var = 0
    var1 = 0
    fig = 0
    temp_mass = [0, 0, 0, 0, 0]
    prev_x = 0
    prev_y = 0
    image = 0
    w = 0
    h = 0
    checkbox_check = 0
    scale = 0
    img = 0
    grafFrame = 0
    window2 = 0
    window3 = 0
    window4 = 0
    canvas2 = 0
    canvas1 = 0
    save_image = 0
    matrix = [[0, 0, 0]] * 3
    matrixText = 0
    nurm_check = 0
    plus128_check = 0
    temp_mass2 = 0
    counter1 = 1
    flag2 = False

    def __init__(self, root):
        self.window = root
        self.window2 = tk.Toplevel(self.window)
        self.window3 = tk.Toplevel(self.window)

        self.fig = plt
        self.window.title("Основное окно")
        self.window2.title("График")
        self.window3.title("Картинка")

        self.matrixText = np.array(
            [[tk.StringVar(), tk.StringVar(), tk.StringVar()], [tk.StringVar(), tk.StringVar(), tk.StringVar()],
             [tk.StringVar(), tk.StringVar(), tk.StringVar()]])
        self.var = tk.IntVar()
        self.var1 = tk.IntVar()
        self.var.set(-1)

        # Основной фрейм для фото
        # Ширину и высоту надо изменить при загрузке изображения
        image1 = Image.open(r'16bit.png')

        self.rgb_im = image1.convert('RGB')
        self.prev_x, self.prev_y = -1, -1
        self.np_image = np.array(image1).astype(np.uint16)
        self.np_image = self.np_image[:, :, :3]
        print(type(self.np_image[0, 0, 2]))
        self.np_image = self.np_image.astype(np.uint8)
        print(type(self.np_image[0, 0, 2]))
        photo = ImageTk.PhotoImage(image1)
        self.w, self.h = image1.size
        imgfarme = tk.Frame(self.window3)
        self.save_image = self.np_image
        self.canvas = tk.Canvas(imgfarme, height=self.h, width=self.w)
        self.image = self.canvas.create_image(0, 0, anchor='nw', image=photo)

        # Основной фрейм для инфы
        infoframe = tk.Frame(self.window, width=400, height=400, highlightbackground="black", background="orange")

        # Фрейм для кнопок и РГБ
        dopframe1 = tk.Frame(infoframe, width=400, height=50)


        # Фрейм для инфы по яркости и РГБ
        rgbfarme = tk.Frame(dopframe1, width=200, height=50, highlightbackground="black")

        # Лейблы для ргб и яркости и позиции
        self.rlable = tk.Label(rgbfarme, text="R: ", anchor="center")
        self.blable = tk.Label(rgbfarme, text="B: ", anchor="center")
        self.glable = tk.Label(rgbfarme, text="G: ", anchor="center")
        self.sredotk = tk.Label(rgbfarme, text="μWp: ", anchor="center")
        self.standotk = tk.Label(rgbfarme, text="sWp: ", anchor="center")
        self.positionlable = tk.Label(rgbfarme, text="Позиция: ", anchor="center")
        self.intensitylable = tk.Label(rgbfarme, text="Интенсивность: ", width=23, anchor="center")

        self.positionlable.pack(fill=tk.X, side=tk.RIGHT)
        self.intensitylable.pack(fill=tk.X, side=tk.RIGHT)
        self.sredotk.pack(fill=tk.X, side=tk.RIGHT)
        self.standotk.pack(fill=tk.X, side=tk.RIGHT)
        self.rlable.pack(fill=tk.X, side=tk.RIGHT)
        self.glable.pack(fill=tk.X, side=tk.RIGHT)
        self.blable.pack(fill=tk.X, side=tk.RIGHT)

        # Фрейм для кнопок
        buttonfarme = tk.Frame(dopframe1, width=200, height=50, highlightbackground="black")
        self.checkbox_check = tk.IntVar()
        savebutton = tk.Button(buttonfarme, text="Сохранить", width=25, command=self.save_click)
        checkbox = tk.Checkbutton(buttonfarme, text="Яркость по строке", variable=self.checkbox_check, onvalue=1,
                                  offvalue=0)

        self.checkbox_check.set(0)
        savebutton.grid(row=0, padx=5, pady=5)
        checkbox.grid(row=1, padx=5, pady=5)

        # Фрейм для изменения характеристик фото
        radbuttonframe = tk.Frame(infoframe, width=400, height=50, highlightbackground="black")
        radiobuttonframe = tk.Frame(radbuttonframe, width=400, height=50)
        peremframe = tk.Frame(radbuttonframe, width=400, height=50)
        self.scale = tk.Scale(peremframe, from_=-100, to=100, orient=tk.HORIZONTAL, resolution=1, length=400,
                              command=self.onScale)

        btrabut = tk.Radiobutton(radiobuttonframe, text="Яркость", variable=self.var, value=0, command=self.setval)
        rrabut = tk.Radiobutton(radiobuttonframe, text="Красный", variable=self.var, value=1, command=self.setval)
        brabut = tk.Radiobutton(radiobuttonframe, text="Синий", variable=self.var, value=3, command=self.setval)
        grabut = tk.Radiobutton(radiobuttonframe, text="Зелёный", variable=self.var, value=2, command=self.setval)
        conrabut = tk.Radiobutton(radiobuttonframe, text="Контрастность", variable=self.var, value=4,
                                  command=self.setval)

        self.scale.pack()
        self.scale.set(0)

        btrabut.grid(padx=5, pady=5, column=0, row=1)
        rrabut.grid(padx=5, pady=5, column=0, row=0)
        brabut.grid(padx=5, pady=5, column=1, row=0, sticky="nw")
        grabut.grid(padx=5, pady=5, column=2, row=0, sticky="nw")

        conrabut.grid(padx=5, pady=5, column=2, row=1)
        radiobuttonframe.grid(padx=5, pady=5, column=0, row=0)
        peremframe.grid(padx=5, pady=5, column=0, row=1)

        # Фрейм для задания 5 "С"
        dopframe2 = tk.Frame(infoframe, width=40, height=50, highlightbackground="black")
        button5Cfarme = tk.Frame(dopframe2, width=40, height=50, highlightbackground="black")
        button5F2farme = tk.Frame(dopframe2, width=40, height=50, highlightbackground="black")
        buttonMatrfarme = tk.Frame(dopframe2, width=40, height=50, highlightbackground="black")

        neglable = tk.Label(button5Cfarme, text="Негатив", anchor="nw")
        btnegbutton = tk.Button(button5Cfarme, text="Яркости", width=13, command=self.neg_br)
        matr = tk.Label(buttonMatrfarme, text="Ввод матрицы ", anchor="nw")
        matrbutton = tk.Button(buttonMatrfarme, text="Матрица", width=13, command=self.openMatrixWindow)

        neglable.grid(column=0, row=0, sticky="nw")
        btnegbutton.grid(padx=5, pady=5, column=0, row=1)
        matr.grid(column=2, row=0, sticky="nw")
        matrbutton.grid(padx=5, pady=5, column=2, row=1)

        # Фрейм для задания 5 "D"
        button5Dfarme = tk.Frame(infoframe, width=400, height=50, highlightbackground="black")
        tr_lable = tk.Label(button5Dfarme, text="Обмен цветов", anchor="nw")
        r_b_tr_button = tk.Button(button5Dfarme, text="R<->B", width=13, command=self.trade_color_R_B)
        r_g_tr_button = tk.Button(button5Dfarme, text="R<->G", width=13, command=self.trade_color_R_G)
        g_b_tr_button = tk.Button(button5Dfarme, text="G<->B", width=13, command=self.trade_color_G_B)
        tr_lable.grid(column=0, row=0, sticky="nw")
        r_b_tr_button.grid(padx=5, pady=5, column=0, row=1)
        r_g_tr_button.grid(padx=5, pady=5, column=0, row=2)
        g_b_tr_button.grid(padx=5, pady=5, column=0, row=3)

        # Фрейм для задания 5 "E"
        button5Efarme = tk.Frame(infoframe, width=400, height=50, highlightbackground="black")
        obm_lable = tk.Label(button5Efarme, text="Симметричное отображение ", anchor="nw")
        hor_obm_button = tk.Button(button5Efarme, text="По горизонтали", width=13, command=self.hor)

        obm_lable.grid(padx=5, pady=5, column=0, row=0)
        hor_obm_button.grid(padx=5, pady=5, column=1, row=0)

        # Фрейм для задания 5 "F"
        button5Ffarme = tk.Frame(infoframe, width=400, height=50, highlightbackground="black")
        del_lable = tk.Label(button5Ffarme, text="Удаление шума", anchor="nw")
        del_4_button = tk.Button(button5Ffarme, text="4-связность", width=13, command=self.del_4)
        del_8_button = tk.Button(button5Ffarme, text="8-связность", width=13, command=self.del_8)

        del_lable.grid(column=2, row=0, sticky="nw")
        del_4_button.grid(padx=5, pady=5, column=2, row=1)
        del_8_button.grid(padx=5, pady=5, column=2, row=2)

        self.canvas.bind('<Motion>', self.motion)
        self.canvas.bind('<Button-1>', self.click_can)
        self.canvas.grid()

        imgfarme.grid(padx=5, pady=5, column=0, row=0)
        infoframe.grid(padx=5, pady=5, column=1, row=0)
        dopframe1.grid(padx=5, pady=5, column=0, row=0, sticky="nw")
        dopframe2.grid(padx=5, pady=5, column=0, row=5, sticky="nw")

        radbuttonframe.grid(padx=5, pady=5, column=0, row=1, sticky="nw")
        button5Cfarme.grid(padx=5, pady=5, column=0, row=0, sticky="nw")
        button5Dfarme.grid(padx=5, pady=5, column=0, row=3, sticky="nw")
        button5Efarme.grid(padx=5, pady=5, column=0, row=4, sticky="nw")
        button5Ffarme.grid(padx=5, pady=5, column=0, row=6, sticky="nw")
        button5F2farme.grid(padx=5, pady=5, column=1, row=0, sticky="nw")
        buttonMatrfarme.grid(padx=5, pady=5, column=2, row=0, sticky="nw")
        rgbfarme.grid(padx=5, pady=5, column=0, row=0)
        buttonfarme.grid(padx=5, pady=5, column=1, row=0)

        self.grafFrame = tk.Frame(self.window2, width=400, height=200,
                                  highlightbackground="black")
        self.grafFrame.grid(padx=5, pady=5, column=0, row=0)
        self.graf()
        self.window.mainloop()

        os.abort()

    def graf(self):
        if self.canvas1:
            self.canvas1.get_tk_widget().destroy()
        colors = ("red", "green", "blue", "black")
        channel_ids = (0, 1, 2, 3)
        self.fig.title("Color Histogram")
        self.fig.xlabel("Color value")
        self.fig.ylabel("Pixel count")
        bb = self.fig.figure()
        ax = bb.add_axes([0.1, 0.1, 0.85, 0.85])
        self.fig.xlim([0, 256])
        for channel_id, c in zip(channel_ids, colors):
            if channel_id != 3:
                histogram, bin_edges = np.histogram(
                    self.np_image[:, :, channel_id], bins=256, range=(0, 256)
                )
            else:
                histogram, bin_edges = np.histogram(
                    self.np_image[:, :, 0] + self.np_image[:, :, 1] + self.np_image[:, :, 2], bins=256, range=(0, 256)
                )
            ax.plot(histogram, label=c, color=c)
        ax.legend()

        self.canvas1 = FigureCanvasTkAgg(bb, master=self.grafFrame)
        self.canvas1.draw()
        self.canvas1.get_tk_widget().grid(padx=5, pady=5, column=0, row=0)

    def openMatrixWindow(self):
        self.window4 = tk.Toplevel(self.window)
        self.window4.title = "Матрица"
        matrixFrame = tk.Frame(self.window4, width=400, height=400, highlightthickness=1, highlightbackground="black")
        buttStandMatrFrame = tk.Frame(self.window4, width=400, height=400, highlightthickness=1,
                                      highlightbackground="black")
        checkFrame = tk.Frame(self.window4, width=400, height=400, highlightthickness=1, highlightbackground="black")
        btFrame = tk.Frame(self.window4, width=400, height=400, highlightthickness=1, highlightbackground="black")
        standMatr1Button = tk.Button(buttStandMatrFrame, text="-1,5,-1", command=self.standMatr1)
        standMatr2Button = tk.Button(buttStandMatrFrame, text="-1,4,-1", command=self.standMatr2)
        self.nurm_check = tk.IntVar()
        self.plus128_check = tk.IntVar()
        norm_checkbox = tk.Checkbutton(checkFrame, text="Нормализация", variable=self.nurm_check, onvalue=1, offvalue=0)
        plus128_checkbox = tk.Checkbutton(checkFrame, text="+128", variable=self.plus128_check, onvalue=1, offvalue=0)

        self.nurm_check.set(0)
        self.plus128_check.set(0)
        exitButton = tk.Button(btFrame, text="OK", command=self.useMatrix)
        High_passButton = tk.Button(btFrame, text="High_pass", command=self.High_pass)
        SmartBlurButton = tk.Button(btFrame, text="SmartBlur", command=self.SmartBlur)
        col, row = 0, 0
        for i in range(3):
            col = 0
            for j in range(3):
                self.matrix[row][col] = tk.Entry(matrixFrame, textvariable=self.matrixText[row][col], width=7)
                self.matrix[row][col].grid(padx=5, pady=5, column=col, row=row);
                col += 1
            row += 1

        matrixFrame.grid(padx=5, pady=5, column=0, row=0)
        buttStandMatrFrame.grid(padx=5, pady=5, column=0, row=1)
        checkFrame.grid(padx=5, pady=5, column=0, row=2)
        btFrame.grid(padx=5, pady=5, column=0, row=3)
        norm_checkbox.grid(padx=5, pady=5, column=0, row=0)
        plus128_checkbox.grid(padx=5, pady=5, column=0, row=1)
        standMatr1Button.grid(padx=5, pady=5, column=0, row=0)
        standMatr2Button.grid(padx=5, pady=5, column=1, row=0)
        exitButton.grid(padx=5, pady=5, column=0, row=0)
        High_passButton.grid(padx=5, pady=5, column=1, row=0)
        SmartBlurButton.grid(padx=5, pady=5, column=3, row=0)

    def standMatr1(self):
        # kernel для повышения резкости изображения
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        col, row = 0, 0
        for i in range(3):
            col = 0
            for j in range(3):
                self.matrixText[row][col].set(kernel[row, col])
                col += 1
            row += 1

    def standMatr2(self):
        # kernel для обнаружения краев
        kernel = np.array([[-1, -1, -1], [-1, 4, -1], [-1, -1, -1]])
        col, row = 0, 0
        for i in range(3):
            col = 0
            for j in range(3):
                self.matrixText[row][col].set(kernel[row, col])
                col += 1
            row += 1

    def High_pass(self):
        kernel = np.array([[-1, -1, -1], [-1, 4, -1], [-1, -1, -1]])
        razm = np.array([[0.0625, 0.125, 0.0625], [0.125, 0.250, 0.125], [0.0625, 0.125, 0.0625]])

        np_image_razm = self.RGB_convolve(self.np_image, razm)
        np_image_razm = self.RGB_convolve(np_image_razm, razm)
        np_image_razm = self.RGB_convolve(np_image_razm, razm)
        self.np_image = self.np_image - np_image_razm

        self.img = ImageTk.PhotoImage(image=Image.fromarray(self.np_image.astype(np.uint8)))
        self.image = self.canvas.create_image(0, 0, anchor="nw", image=self.img)

    def SmartBlur(self):
        razm = np.array([[0.0625, 0.125, 0.0625], [0.125, 0.250, 0.125], [0.0625, 0.125, 0.0625]])
        border_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        border_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        np_image_razm = self.RGB_convolve(self.np_image, razm)
        np_image_razm_gray = np.mean(np_image_razm, -1)
        np_image_border_x = self.convolve2d(np_image_razm_gray, border_x)
        np_image_border_y = self.convolve2d(np_image_razm_gray, border_y)
        np_image_border_x = np.power(np_image_border_x, 2)
        np_image_border_y = np.power(np_image_border_y, 2)
        mask2 = np.sqrt(np_image_border_x + np_image_border_y)
        xpercentile = np.percentile(mask2, 90)
        mask1 = self.color(mask2, xpercentile)
        mask1 = self.delWhite(mask1)
        mask = self.RGB_convolve(mask1, razm)

        a1 = np.multiply(np_image_razm, (255 - mask), dtype=float)
        a2 = np.multiply(self.np_image, mask, dtype=float)
        self.np_image = a1 + a2
        self.np_image = np.array(self.np_image / 255, np.uint8)
        self.img = ImageTk.PhotoImage(image=Image.fromarray(mask1.astype(np.uint8)))
        self.image = self.canvas.create_image(0, 0, anchor="nw", image=self.img)

    col = set()
    newNeb = set()
    a = 0

    def delWhite(self, np_img):
        self.a = np.zeros((self.h, self.w), dtype=int)
        for i in range(self.h):
            for j in range(self.w):
                if np_img[i, j, 0] == 255 and np_img[i, j, 1] == 255 and np_img[i, j, 2] == 255:
                    self.a[i, j] = 1
        cl = 2
        for i in range(self.h):
            for j in range(self.w):
                st, bl = self.nb(i, j)
                if bl:
                    self.col.add((i, j))
                    for k in st:
                        self.col.add(k)
                        self.newNeb.add(k)
                    while len(self.newNeb) != 0:
                        c, cc = self.newNeb.pop()
                        st, bl = self.nb(c, cc)
                        if bl:
                            for k in st:
                                self.col.add(k)
                                self.newNeb.add(k)
                if len(self.col) <= 16:
                    for k in self.col:
                        self.a[k] = 0
                    self.col.clear()
                else:
                    for k in self.col:
                        self.a[k] = cl
                    cl += 1
                    self.col.clear()
        return self.oneInThree(self.a)


    def nb(self,y, x):
        st = set()
        if self.a[y, x] == 1:
            self.col.add((y, x))
            if x + 1 < self.w and self.a[y, x + 1] == 1:
                if (y, x + 1) not in self.col:
                    st.add((y, x + 1))
            if x - 1 > -1 and self.a[y, x - 1] == 1:
                if (y, x - 1) not in self.col:
                    st.add((y, x - 1))
            if y + 1 < self.h and self.a[y + 1, x] == 1:
                if (y + 1, x) not in self.col:
                    st.add((y + 1, x))
            if y - 1 > -1 and self.a[y - 1, x] == 1:
                if (y - 1, x) not in self.col:
                    st.add((y - 1, x))
            return st, True
        return st, False

    def oneInThree(self, oneColorMatrix):
        threeColorMatrix = np.zeros((self.h, self.w, 3), dtype=int)

        for i in range(self.h):
            for j in range(self.w):
                if oneColorMatrix[i, j] == 0:
                    threeColorMatrix[i, j, :] = 0
                else:
                    threeColorMatrix[i, j, :] = 255
        return threeColorMatrix


    def color(self, oneColorImage, perc):
        im2 = np.empty_like(self.np_image)
        for a in range(len(oneColorImage)):
            for b in range(len(oneColorImage[a])):
                if oneColorImage[a][b] > perc:
                    for c in range(3):
                        im2[a][b][c] = 255
                else:
                    for c in range(3):
                        im2[a][b][c] = 0

        return im2

    def convolve2d(self, image, kernel):
        output = np.zeros_like(image)
        xmean = kernel.sum()
        image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
        image_padded[1:-1, 1:-1] = image

        # Loop
        nm = 1
        pl = 0

        if self.nurm_check.get() == 1:
            nm = 1 / xmean
            if nm == math.inf:
                nm = 1
        if self.plus128_check.get() == 1:
            pl = 128
        for x in range(image.shape[1]):
            for y in range(image.shape[0]):
                #
                output[y, x] = (kernel * image_padded[y: y + 3, x: x + 3] * nm).sum() + pl

        return output

    def RGB_convolve(self, im1, kern):
        im2 = np.empty_like(im1)
        for dim in range(im1.shape[2]):  # rgb loop
            im2[:, :, dim] = self.convolve2d(im1[:, :, dim], kern)
        return im2

    def useMatrix(self):
        mx = np.zeros((3, 3))
        col, row = 0, 0
        for i in range(3):
            col = 0
            for j in range(3):
                mx[col][row] = self.matrixText[col][row].get()
                print(self.matrixText[col][row].get())
                col += 1
            row += 1

        print(mx)

        self.np_image = self.RGB_convolve(self.np_image, mx)
        self.img = ImageTk.PhotoImage(image=Image.fromarray(self.np_image.astype(np.uint8)))
        self.image = self.canvas.create_image(0, 0, anchor="nw", image=self.img)

    def motion(self, event):
        try:
            x, y = event.x, event.y
            self.Y = event.y
            if (self.prev_x, self.prev_y) == (x, y): return
            if (self.rect_id < 0):
                self.old_x, self.old_y = x, y
                self.rect_id = self.canvas.create_rectangle(x - 6, y - 6, x + 6, y + 6)
            else:
                self.canvas.delete(self.rect_id)
                self.rect_id = self.canvas.create_rectangle(x - 6, y - 6, x + 6, y + 6)
                shift = x - self.old_x, y - self.old_y
                self.old_x, self.old_y = x, y
                self.canvas.move(self.rect_id, *shift)
                self.prev_x, self.prev_y = x, y

            img = ImageTk.PhotoImage(image=Image.fromarray(self.np_image.astype(np.uint8)))
            self.image = self.canvas.create_image(0, 0, anchor="nw", image=img)
            try:
                r, g, b = int(self.np_image[y][x][0]), int(self.np_image[y][x][1]), int(self.np_image[y][x][2])
                sred = (r + g + b) / 3
                sredotkl = (abs(r - sred) + abs(b - sred) + abs(g - sred) / 3)
                cvadotk = math.sqrt((1 / 3) * (((r - sred) ** 2) + ((g - sred) ** 2) + ((b - sred) ** 2)))
                self.rlable["text"] = f'R: {r}'
                self.blable["text"] = f'B: {b}'
                self.glable["text"] = f'G: {g}'
                self.sredotk["text"] = f'μWp: {sredotkl:.3f}'
                self.intensitylable["text"] = f'Интенсивность: {sred:.3f}'
                self.positionlable["text"] = f'Позиция: {x}, {y}'
                self.standotk["text"] = f'sWp: {cvadotk:.3f}'
            except RuntimeWarning:
                None
            self.prev_x, self.prev_y = x, y
        except IndexError:
            None

    def graf1(self, y):
        if self.canvas1:
            self.canvas1.get_tk_widget().destroy()
        c = "blue"
        self.fig.title("Color Histogram")
        self.fig.xlabel("Color value")
        self.fig.ylabel("Pixel count")
        bb = self.fig.figure()
        ax = bb.add_axes([0.1, 0.1, 0.85, 0.85])
        self.fig.xlim([0, 256])
        histogram, bin_edges = np.histogram(
            self.np_image[y, :, 0] + self.np_image[y, :, 1] + self.np_image[y, :, 2], bins=256, range=(0, 256)
        )
        ax.plot(histogram, label=c, color=c)
        ax.legend()

        self.canvas1 = FigureCanvasTkAgg(bb, master=self.grafFrame)
        self.canvas1.draw()
        self.canvas1.get_tk_widget().grid(padx=5, pady=5, column=0, row=0)

    def save_click(self):
        directory = fd.askdirectory(title="Выберите папку", initialdir="/")
        vrem_name = datetime.datetime.now().time().second.__str__() + "." + datetime.datetime.now().time().minute.__str__() \
                    + "." + datetime.datetime.now().time().hour.__str__() + "_" + date.today().__str__() + ".png"

        Image.fromarray(self.np_image).save(vrem_name)
        shutil.move(vrem_name, directory)

    # Обмен цветов
    def trade_color_R_B(self):
        self.np_image[:, :, [0]], self.np_image[:, :, [2]] = self.np_image[:, :, [2]], self.np_image[:, :, [0]]
        self.img = ImageTk.PhotoImage(image=Image.fromarray(self.np_image.astype(np.uint8)))
        self.save_image = self.np_image
        self.image = self.canvas.create_image(0, 0, anchor="nw", image=self.img)
        self.graf()

    def trade_color_R_G(self):
        self.np_image[:, :, [0]], self.np_image[:, :, [1]] = self.np_image[:, :, [1]], self.np_image[:, :, [0]]
        self.graf()
        self.img = ImageTk.PhotoImage(image=Image.fromarray(self.np_image.astype(np.uint8)))
        self.save_image = self.np_image
        self.image = self.canvas.create_image(0, 0, anchor="nw", image=self.img)

    def trade_color_G_B(self):
        self.np_image[:, :, [2]], self.np_image[:, :, [1]] = self.np_image[:, :, [1]], self.np_image[:, :, [2]]
        self.graf()
        self.img = ImageTk.PhotoImage(image=Image.fromarray(self.np_image.astype(np.uint8)))
        self.save_image = self.np_image
        self.image = self.canvas.create_image(0, 0, anchor="nw", image=self.img)

    def neg_br(self):
        self.np_image = 255 - self.np_image
        self.graf()
        self.img = ImageTk.PhotoImage(image=Image.fromarray(self.np_image.astype(np.uint8)))
        self.save_image = self.np_image
        self.image = self.canvas.create_image(0, 0, anchor="nw", image=self.img)

    # Отражение
    def hor(self):
        self.np_image = np.fliplr(self.np_image)
        self.graf()
        self.img = ImageTk.PhotoImage(image=Image.fromarray(self.np_image.astype(np.uint8)))
        self.save_image = self.np_image
        self.image = self.canvas.create_image(0, 0, anchor="nw", image=self.img)

    # увеличение/уменьшение интенсивности яркости и отдельных цветовых каналов
    def intens(self, val, col):
        vrem = int(val)
        if col == -1:
            self.np_image = self.np_image + vrem
        else:
            if 0 <= col < 3:
                self.np_image[..., col] = self.np_image[..., col] + vrem

        self.img = ImageTk.PhotoImage(image=Image.fromarray(self.np_image.astype(np.uint8)))
        self.image = self.canvas.create_image(0, 0, anchor="nw", image=self.img)

    def contr(self, val):
        self.np_image = self.save_image
        contrast = (100.0 + float(val)) / 100.0
        contrast = contrast * contrast
        a = self.np_image
        a = a / 255.0
        a = a - 0.5
        a = a * contrast
        a = a + 0.5
        a = a * 255

        a = a.astype(int)

        self.np_image = a
        self.img = ImageTk.PhotoImage(image=Image.fromarray(self.np_image.astype(np.uint8)))
        self.image = self.canvas.create_image(0, 0, anchor="nw", image=self.img)


    def onScale(self, val):
        vrem = self.var.get()
        if vrem != -1:
            self.temp_mass[vrem] = val
            if vrem == 0:
                self.intens(val, -1)
            if vrem == 1:
                self.intens(val, vrem - 1)
            if vrem == 2:
                self.intens(val, vrem - 1)
            if vrem == 3:
                self.intens(val, vrem - 1)
            if vrem == 4:
                self.contr(val)

    def setval(self):
        vrem = self.var.get()
        if 0 <= vrem < 4:
            self.scale.config(from_=-255, to=255)
        else:
            self.scale.config(from_=-100, to=100)
        self.scale.set(self.temp_mass[vrem])

    def dop_task(self):
        self.img = np.array(Image.open(r'4.png'))
        self.img = self.img[:, :, :3]
        dst = (self.img * 0.8 + self.np_image * 0.3).astype(np.uint8)
        self.np_image = dst
        self.graf()
        self.img = ImageTk.PhotoImage(image=Image.fromarray(self.np_image))
        self.image = self.canvas.create_image(0, 0, anchor="nw", image=self.img)

    def del_4(self):
        self.rgb_im = Image.fromarray(self.np_image).convert('RGB')
        found_pixels = []
        for i, pixel in enumerate(self.rgb_im.getdata()):
            found_pixels.append(i)

        found_pixels_coords = [divmod(index, self.h) for index in found_pixels]
        members = [(0, 0)] * 5

        for i, j in found_pixels_coords:
            if 0 < i < self.w - 1 and 0 < j < self.h - 1:
                members[0] = self.rgb_im.getpixel((i - 1, j))
                members[1] = self.rgb_im.getpixel((i, j - 1))
                members[2] = self.rgb_im.getpixel((i, j))
                members[3] = self.rgb_im.getpixel((i, j + 1))
                members[4] = self.rgb_im.getpixel((i + 1, j))
                res = [int(sum(ele) / len(members)) for ele in zip(*members)]
                res = tuple(res)
                self.rgb_im.putpixel((i, j), res)
            else:
                continue
        np_image2 = np.array(self.rgb_im)
        self.np_image = np_image2[:, :, :3]
        self.graf()
        self.img = ImageTk.PhotoImage(image=Image.fromarray(self.np_image.astype(np.uint8)))
        self.image = self.canvas.create_image(0, 0, anchor="nw", image=self.img)

    def del_8(self):
        self.rgb_im = Image.fromarray(self.np_image).convert('RGB')
        found_pixels = []
        for i, pixel in enumerate(self.rgb_im.getdata()):
            found_pixels.append(i)

        found_pixels_coords = [divmod(index, self.h) for index in found_pixels]
        members = [(0, 0)] * 9

        for i, j in found_pixels_coords:
            if 0 < i < self.w - 1 and 0 < j < self.h - 1:
                members[0] = self.rgb_im.getpixel((i - 1, j - 1))
                members[1] = self.rgb_im.getpixel((i - 1, j))
                members[2] = self.rgb_im.getpixel((i - 1, j + 1))
                members[3] = self.rgb_im.getpixel((i, j - 1))
                members[4] = self.rgb_im.getpixel((i, j))
                members[5] = self.rgb_im.getpixel((i, j + 1))
                members[6] = self.rgb_im.getpixel((i + 1, j - 1))
                members[7] = self.rgb_im.getpixel((i + 1, j))
                members[8] = self.rgb_im.getpixel((i + 1, j + 1))
                res = [int(sum(ele) / len(members)) for ele in zip(*members)]
                res = tuple(res)
                self.rgb_im.putpixel((i, j), res)
            else:
                continue
        np_image2 = np.array(self.rgb_im)
        self.np_image = np_image2[:, :, :3]
        self.graf()
        self.img = ImageTk.PhotoImage(image=Image.fromarray(self.np_image.astype(np.uint8)))
        self.image = self.canvas.create_image(0, 0, anchor="nw", image=self.img)

    def click_can(self, event):
        if self.checkbox_check.get() == 1:
            self.graf1(self.Y)


sys.setrecursionlimit(10000)
aaa = TkinterWin(tk.Tk())
aaa.window2.destroy()
