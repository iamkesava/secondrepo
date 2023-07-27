# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 20:46:01 2023

@author: kesav
"""

from tkinter import *
import tkinter as tk
from tkinter import ttk, messagebox

import pymysql
from PIL import Image, ImageTk
import random

from signup import Signup
from login import Login


# ------------------------------------------------------------ Main Window -----------------------------------------
def Signupmeth():
    sign = Signup()


def Loginmeth():
    log = Login()


def report():
    wingrid = Tk()
    wingrid.title("View Prediction Report ")
    wingrid.geometry("1300x1500")
    wingrid.maxsize(width=2100, height=2500)
    wingrid.minsize(width=2100, height=2500)

    main_frame = Frame(wingrid)
    main_frame.pack(fill=BOTH, expand=1)

    my_canvas = Canvas(main_frame)
    my_canvas.pack(side=LEFT, fill=BOTH, expand=1)

    my_scrollbar = ttk.Scrollbar(main_frame, orient=VERTICAL, command=my_canvas.yview)
    my_scrollbar.pack(side=RIGHT, fill=Y)

    my_canvas.config(yscrollcommand=my_scrollbar.set)
    my_canvas.bind('<Configure>', lambda e: my_canvas.configure(scrollregion=my_canvas.bbox("all")))

    wingrid = Frame(my_canvas)

    my_canvas.create_window((0, 0), window=wingrid, anchor="nw")

    con = pymysql.connect(host="localhost", port=3306, user="root", password="root", database="realestate")
    cur = con.cursor()

    cur.execute("select * from resultinfo")
    data = cur.fetchall()

    r = 0
    for col in data:
        c = 0
        for row in col:
            label = Label(wingrid, width=23, height=2, text=row, relief=tkinter.RIDGE)
            label.grid(row=r, column=c)
            c += 1
        r += 1
    con.commit()
    con.close()
win = Tk()

# app title
win.title("Real Estate Price Prediction")

# window size
win.maxsize(width=1100, height=1000)
win.minsize(width=1100, height=1000)
bg = PhotoImage(file="Apps5.png")

# Create Canvas
canvas1 = Canvas(win, width=400, height=400)


canvas1.pack(fill="both", expand=True)

# Display image
canvas1.create_image(0, 0, image=bg, anchor="nw")

# heading label
heading = Label(win, text="Real Estate Price Prediction", font='Verdana 20 bold')
heading.place(x=350, y=50)

btn_signup = Button(win, text="Register", font='Verdana 10 bold', width="20", command=Signupmeth)
btn_signup.place(x=600, y=200)
btn_login = Button(win, text="Login", font='Verdana 10 bold', width="20", command=Loginmeth)
btn_login.place(x=600, y=250)

btn_login = Button(win, text="Report", font='Verdana 10 bold', width="20", command=report)
btn_login.place(x=600, y=300)

btn_exit = Button(win, text="Exit", font='Verdana 10 bold', width="20", command=quit)
btn_exit.place(x=600, y=350)

win.mainloop()



from fileinput import filename
from tkinter import *
import tkinter as tk
import tkinter
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import random
import pymysql
import pandas as pd
import csv
from csv import writer
from tkinter import simpledialog
from tkinter.filedialog import askopenfilename
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

class ViewData:
    def _init_(self):
        def regression():
            wingrid1 = Tk()
            wingrid1.title("View Dataset  Window")
            wingrid1.maxsize(width=1400, height=900)
            wingrid1.minsize(width=1400, height=900)
            userinput = simpledialog.askstring(title="Regression Values", prompt="Enter Area Name :")
            print(userinput)

            con = pymysql.connect(host="localhost", user="root", password="root", database="realestate")
            cur = con.cursor()

            cur.execute("select sum(price),avg(price),VARIANCE((price) from Land where Area=%s ", (userinput))
            row = cur.fetchone()
            sumarea, avgarea, vararea = row
            print(sumarea)
            print(avgarea)
            print(vararea)

            con = pymysql.connect(host="localhost", user="root", password="root", database="realestate")
            cur = con.cursor()

            cur.execute("insert into  Regre(area, total,mean,vari) values (%s,%s,%s,)", (userinput, sumarea, avgarea))
            str1 = "Total Value : " + str(sumarea) + "\nMean Value : " + str(avgarea) + "\nVariance Value : " + str(
                vararea) + " saved successfully"

            messagebox.showinfo("Record Saved", str1, parent=wingrid1)

        def feature():
            wingrid1 = Tk()
            wingrid1.title("View Dataset  Window")
            wingrid1.maxsize(width=1400, height=900)
            wingrid1.minsize(width=1400, height=900)

            with open(filename) as file:
                reader = csv.reader(file)
                r = 0
                for row in reader:
                    c = 0
                    for col in row:

                        if (c == 1 or c == 4 or c == 6):
                            label = Label(wingrid1, width=10, height=2, text=col, relief=tkinter.RIDGE)
                            label.grid(row=r, column=c)
                        c = c + 1

                    r += 1

        def upload(winland):
            f_types = [('CSV Files', '.csv'), ('Xlsx Files', '.xlsx')]
            filename = askopenfilename(filetypes=f_types)

            if filename.endswith('.xlsx'):
                file = pd.read_excel(filename)
                file.to_csv(filename.rstrip('.xlsx') + ".csv")
                filename = filename.rstrip('.xlsx') + ".csv"

            con = pymysql.connect(host="localhost", port=3306, user="root", password="root", database="realestate")
            cur = con.cursor()

            count = 0
            with open(filename, newline="") as file:
                reader = csv.reader(file)
                r = 1
                for col in reader:
                    count += 1
                    #print(count)
                    if 'print("null checking")' in col: continue
                    print(col[1], col[2], col[3], col[4], col[5], col[6], col[7])
                    cur.execute(
                        "insert into mynewLand(landid,area,sub,measure,price,approved,year) values (%s,%s,%s,%s,%s,%s,%s)",
                        (
                            col[1], col[2], col[3], col[4], col[5], col[6], col[7]
                        ))


                con.commit()
                con.close()
            messagebox.showinfo("Record Uploaded Successfully", filename)

        def viewdataset():
            wingrid = Tk()
            wingrid.title("View Dataset  Window")
            wingrid.geometry("1400x900")
            # wingrid.maxsize(width=1400 ,  height=2500)
            # wingrid.minsize(width=1400 ,  height=2500)

            main_frame = Frame(wingrid)
            main_frame.pack(fill=BOTH, expand=1)

            my_canvas = Canvas(main_frame)
            my_canvas.pack(side=LEFT, fill=BOTH, expand=1)

            my_scrollbar = ttk.Scrollbar(main_frame, orient=VERTICAL, command=my_canvas.yview)
            my_scrollbar.pack(side=RIGHT, fill=Y)

            my_canvas.config(yscrollcommand=my_scrollbar.set)
            my_canvas.bind('<Configure>', lambda e: my_canvas.configure(scrollregion=my_canvas.bbox("all")))

            wingrid = Frame(my_canvas)

            my_canvas.create_window((0, 0), window=wingrid, anchor="nw")

            con = pymysql.connect(host="localhost", port=3306, user="root", password="root", database="realestate")
            cur = con.cursor()

            cur.execute("select * from mynewLand")
            data = cur.fetchall()

            r = 0
            for col in data:
                c = 0
                for row in col:
                    label = Label(wingrid, width=23, height=2, text=row, relief=tkinter.RIDGE)
                    label.grid(row=r, column=c)
                    c += 1
                r += 1
            con.commit()
            con.close()

        def preprocesssing():
            wingrid = Tk()
            wingrid.title("Preprocessing  Window")
            wingrid.geometry("1400x900")
            # wingrid.maxsize(width=1400 ,  height=2500)
            # wingrid.minsize(width=1400 ,  height=2500)

            main_frame = Frame(wingrid)
            main_frame.pack(fill=BOTH, expand=1)

            my_canvas = Canvas(main_frame)
            my_canvas.pack(side=LEFT, fill=BOTH, expand=1)

            my_scrollbar = ttk.Scrollbar(main_frame, orient=VERTICAL, command=my_canvas.yview)
            my_scrollbar.pack(side=RIGHT, fill=Y)

            my_canvas.config(yscrollcommand=my_scrollbar.set)
            my_canvas.bind('<Configure>', lambda e: my_canvas.configure(scrollregion=my_canvas.bbox("all")))

            wingrid = Frame(my_canvas)

            my_canvas.create_window((0, 0), window=wingrid, anchor="nw")

            con = pymysql.connect(host="localhost", port=3306, user="root", password="root", database="realestate")
            cur = con.cursor()
            query = "SELECT* FROM land WHERE (price IS NOT NULL AND TRIM(price) <> '' AND  measure IS NOT NULL AND TRIM(measure) <> '' AND  year IS NOT NULL AND TRIM(year) <> '')"
            cur.execute(query)
            data = cur.fetchall()
            r = 0
            for col in data:
                c = 0
                for row in col:
                    label = Label(wingrid, width=23, height=2, text=row, relief=tkinter.RIDGE)
                    label.grid(row=r, column=c)
                    c += 1
                r += 1
            con.commit()
            con.close()

        def arima():

            con = pymysql.connect(host="localhost", port=3306, user="root", password="root", database="realestate")
            cur = con.cursor()
            cur1 = con.cursor()
            cur.execute("select area from mynewLand")
            data = cur.fetchall()

            r = 0
            for area in data:
                c = 0
                for row in area:
                    print(row)
                    areaname = row
                    # label = Label(wingrid, width=23, height=2, text=row, relief=tkinter.RIDGE)
                    # label.grid(row=r, column=c)

                    name = "select sum(price),avg(price),VARIANCE(price) from mynewland WHERE area ='" + areaname + "'"
                    cur1.execute(name)

                    row1 = cur1.fetchone()
                    sumarea, avgarea, vararea = row1

                    con1 = pymysql.connect(host="localhost", user="root", password="root", database="realestate")
                    cur = con1.cursor()

                    cur.execute("insert into regre(area,suminfo,mean,vari) values (%s,%s,%s,%s)",
                                (areaname, sumarea, avgarea, vararea))
                    con1.commit()
                    print(sumarea)
                    print(avgarea)
                    print(vararea)

                c += 1
                r += 1
            con.commit()
            con.close()

        def featureextraction():
            wingrid = Tk()
            wingrid.title("Feature Extraction Window")
            wingrid.geometry("1400x900")
            # wingrid.maxsize(width=1400 ,  height=2500)
            # wingrid.minsize(width=1400 ,  height=2500)

            main_frame = Frame(wingrid)
            main_frame.pack(fill=BOTH, expand=1)

            my_canvas = Canvas(main_frame)
            my_canvas.pack(side=LEFT, fill=BOTH, expand=1)

            my_scrollbar = ttk.Scrollbar(main_frame, orient=VERTICAL, command=my_canvas.yview)
            my_scrollbar.pack(side=RIGHT, fill=Y)

            my_canvas.config(yscrollcommand=my_scrollbar.set)
            my_canvas.bind('<Configure>', lambda e: my_canvas.configure(scrollregion=my_canvas.bbox("all")))

            wingrid = Frame(my_canvas)



            my_canvas.create_window((0, 0), window=wingrid, anchor="nw")

            con = pymysql.connect(host="localhost", port=3306, user="root", password="root", database="realestate")
            cur = con.cursor()
            query = "SELECT area, measure, price, approved, year  FROM land WHERE (price IS NOT NULL AND TRIM(price) <> '' AND  measure IS NOT NULL AND TRIM(measure) <> '' AND  year IS NOT NULL AND TRIM(year) <> '')"
            cur.execute(query)
            data = cur.fetchall()

            # feature extraction
            test = SelectKBest(score_func=f_classif, k=5)
            #fit = test.fit(X, Z)
            # summarize scores
            set_printoptions(precision=3)
            r = 0
            for col in data:
                c = 0
                for row in col:
                    label = Label(wingrid, width=23, height=2, text=row, relief=tkinter.RIDGE)
                    label.grid(row=r, column=c)
                    c += 1
                r += 1
            con.commit()
            con.close()

        # close signup function
        def switch():
            winland.destroy()

        # start Signup Window

        winland = Toplevel()
        winland.title("Landdetails Details Window")
        winland.maxsize(width=1200, height=500)
        winland.minsize(width=1200, height=500)
        winland.configure(bg='#f2f28a')
        bg = PhotoImage(file="Apps5.png")

        save_btn = Button(winland, text="Switch To Home", font='Verdana 10 bold', command=switch)
        save_btn.place(x=350, y=100)

        btn_upload = Button(winland, text="Upload Dataset", font='Verdana 10 bold', width="20",
                            command=lambda: upload(winland))
        btn_upload.place(x=100, y=200)

        btn_ds = Button(winland, text="View Dataset", font='Verdana 10 bold', width="20", command=viewdataset)
        btn_ds.place(x=300, y=200)

        btn_ds1 = Button(winland, text="Preprocessing", font='Verdana 10 bold', width="20", command=preprocesssing)
        btn_ds1.place(x=500, y=200)

        btn_ds2 = Button(winland, text="Feature Extraction", font='Verdana 10 bold', width="20",
                         command=featureextraction)
        btn_ds2.place(x=700, y=200)

        btn_ds3 = Button(winland, text="Arima Model", font='Verdana 10 bold', width="20", command=arima)
        btn_ds3.place(x=900, y=200)

        winland.mainloop()