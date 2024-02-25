import tkinter as tk
import tkinter.messagebox
from tkinter import filedialog
from constant import Constant
home=Constant.home


class Chunck():
    def __init__(self, text,root):
        self.text = text
        self.root=root

    def _browse_files(self):
        filename = filedialog.askopenfilename(initialdir=home, title='Select a File')
        self.var.set(filename)

    def _browse_dir(self):
        dir = filedialog.askdirectory(initialdir=home, title='Select a directory')
        self.var.set(dir)

    def select_dir(self, row,column):
        self.var = tk.StringVar()
        self.button = tk.Button(self.root, text=self.text, command=self._browse_dir)
        self.button.grid(row=row,column=column,pady=10)
        self.entry = tk.Entry(self.root, textvariable=self.var, width=50)
        self.entry.grid(row=row,column=column+1,pady=10)

    def select_image(self, row,column):
        self.var = tk.StringVar()
        self.button = tk.Button(self.root, text=self.text, command=self._browse_files)
        self.button.grid(row=row,column=column,pady=10)
        self.entry = tk.Entry(self.root, textvariable=self.var, width=50)
        self.entry.grid(row=row,column=column+1,pady=10)

class CreateAbout:
    def __init__(self, root):
        self.about_frame = tk.Frame(root)
        tk.Label(self.about_frame, text='This software is designed and owned by kkforever').pack()
        tk.Label(self.about_frame,
                 text='For more information, welcome to my github page https://github.com/Ke-Albert').pack()


class CreateRun:
    def __init__(self, root,run):
        self.run_frame = tk.Frame(root)
        # tk.Label(self.run_frame,text='run page').pack()
        self.create_page(self.run_frame,run)

    def create_page(self,root,run):
        button = tk.Button(root, text='run', command=run)
        button.grid(row=5,column=2)


class CreateHyper:
    def __init__(self, root):
        self.hyper_frame = tk.Frame(root)
        # tk.Label(self.hyper_frame,text='hyper page').pack()
        self.create_page(self.hyper_frame)

    def create_page(self,root):
        tk.Label(root,text='train numbers').grid(row=1,column=1)
        self.train_num=tk.Entry(root,width=5)
        self.train_num.grid(row=1,column=2)
        tk.Label(root,text='max_iters').grid(row=2,column=1)
        self.max_iters=tk.Entry(root,width=5)
        self.max_iters.grid(row=2,column=2)
        tk.Label(root,text='lr').grid(row=3,column=1)
        self.lr=tk.Entry(root,width=5)
        self.lr.grid(row=3,column=2)


class CreaterBasic:
    def __init__(self,root):
        self.basic_frame=tk.Frame(root)
        # tk.Label(self.basic_frame,text='basic page').pack()
        self.create_page(self.basic_frame)

    def create_page(self,root):
        self.imageB = Chunck('Select before',root)
        self.imageB.select_image(1,1)
        self.imageA = Chunck('Select after',root)
        self.imageA.select_image(2,1)
        self.splitB = Chunck('Split before',root)
        self.splitB.select_dir(3,1)
        self.splitA = Chunck('Split after',root)
        self.splitA.select_dir(4,1)
        self.splitD = Chunck('Split diff',root)
        self.splitD.select_dir(5,1)
        self.merge = Chunck('Merge',root)
        self.merge.select_dir(6,1)
        self.outPut = Chunck('Output path',root)
        self.outPut.select_dir(7,1)
        self.label = tk.Label(root, text='Output name')
        self.label.grid(row=8,column=1)
        self.entry = tk.Entry(root, width=15)
        self.entry.grid(row=8,column=2)
        self.label_1 = tk.Label(root, text='train or predict')
        self.label_1.grid(row=9,column=1)
        self.entry_1 = tk.Entry(root, width=15)
        self.entry_1.grid(row=9,column=2)



