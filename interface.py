import os, glob
import tkinter as tk
from tkinter import filedialog
import pickle

import main
import utils
import core
from osgeo import gdal
import numpy as np
from sklearn.cluster import KMeans

prefix = 'split'
merge_name = 'merge.tif'

home=os.getcwd()
# print(home)

window = tk.Tk()
window.title('DSFANet Change Detection')
window.geometry('500x500')

class Chunck():
    def __init__(self, text):
        self.text = text

    def _browse_files(self):
        filename = filedialog.askopenfilename(initialdir=home, title='Select a File')
        self.var.set(filename)

    def _browse_dir(self):
        dir = filedialog.askdirectory(initialdir=home, title='Select a directory')
        self.var.set(dir)

    def select_dir(self, bias):
        self.var = tk.StringVar()
        self.button = tk.Button(window, text=self.text, command=self._browse_dir)
        self.button.place(x=15, y=15 + bias * 40)
        self.entry = tk.Entry(window, textvariable=self.var, width=50)
        self.entry.place(x=120, y=20 + bias * 40)

    def select_image(self, bias):
        self.var = tk.StringVar()
        self.button = tk.Button(window, text=self.text, command=self._browse_files)
        self.button.place(x=15, y=15 + bias * 40)
        self.entry = tk.Entry(window, textvariable=self.var, width=50)
        self.entry.place(x=120, y=20 + bias * 40)

def run():
    # split
    img_path_before = imageB.entry.get()
    img_path_after = imageA.entry.get()
    split_path_before = splitB.entry.get()
    split_path_after = splitA.entry.get()
    split_diff = splitD.entry.get()
    merge_path = merge.entry.get()
    out_put_path = outPut.entry.get()
    out_put_name = entry.get()

    data_before = main.GRID.load_img(filename=img_path_before)
    data_after = main.GRID.load_img(filename=img_path_after)
    main.GRID.split_image(fn_out=split_path_before, origin_data=data_before['data'],
                          origin_transform=data_before['transform'], output_size=(2000, 2000),
                          proj=data_before['projection'])
    main.GRID.split_image(fn_out=split_path_after, origin_data=data_after['data'],
                          origin_transform=data_after['transform'], output_size=(2000, 2000),
                          proj=data_after['projection'])

    tifs_before = [i for i in os.listdir(split_path_before) if i.endswith(".tif")]
    tifs_after = [i for i in os.listdir(split_path_after) if i.endswith(".tif")]

    # main loop
    while True:
        train_or_pre = entry_1.get()
        if train_or_pre == '1' or train_or_pre == '2':
            break
        else:
            print('输入错误')
    counter = 1
    for tif_before, tif_after in zip(tifs_before, tifs_after):
        counter += 1
        print('the {} time'.format(counter))
        print('path:', tif_before, tif_after)
        params = utils.load_dataset(split_path_before +'/'+ tif_before, split_path_after +'/'+ tif_after, train_or_pre)
        X, Y, diff, row, column, projection, minx, maxy, xres = params['before'], params['after'], \
            params['diff'], params['row'], params[
            'column'], params['projection'], \
            params['minX'], params['maxY'], params['resolution']

        if train_or_pre == '1':
            core.main(X, Y, row, column, projection, minx, maxy, xres, str(counter), split_diff+'/', diff=diff)

        elif train_or_pre == '2':
            core.main(X, Y, row, column, projection, minx, maxy, xres, str(counter), split_diff+'/', flag='pre')

    del X, Y, diff

    ########
    # MERGE#
    ########
    main.GRID.merge_image(split_diff+'/', merge_path +'/'+ merge_name)

    ###########
    # 阈值分割###
    ###########
    diff = gdal.Open(merge_path +'/'+ merge_name)
    projection = diff.GetProjection()
    minx, xres, xskew, maxy, yskew, yres = diff.GetGeoTransform()
    img_width = diff.RasterXSize  # image width
    img_height = diff.RasterYSize  # image height
    diff = diff.ReadAsArray(0, 0, img_width, img_height)
    diff = diff.flatten()
    diff = np.expand_dims(diff, axis=1)
    bin = KMeans(n_clusters=2).fit(diff).labels_
    out_CD = out_put_path +'/'+ out_put_name+'.tif'
    print('开始输出阈值分割图片...')
    core.outputTif(out_CD, img_width, img_height, minx, maxy, xres, projection, bin.reshape(img_height, img_width))
    print('阈值输出完成!')

    ###################
    # 删除临时分割的影像##
    ###################
    print('清理临时文件中...')
    for file in glob.glob(split_diff+'/' + '*'):
        os.remove(file)
    for file in glob.glob(split_path_before+'/' + '*'):
        os.remove(file)
    for file in glob.glob(split_path_after+'/' + '*'):
        os.remove(file)
    for file in glob.glob(merge_path+'/' + '*'):
        os.remove(file)
    print('清理完成!')
def save_parameters():
    vars={'imgB':imageB.entry.get(),'imgA':imageA.entry.get(),'splitB':splitB.entry.get(),
          'splitA':splitA.entry.get(),'splitD':splitD.entry.get(),'merge':merge.entry.get(),
          'outPut':outPut.entry.get(),'entry':entry.get(),'entry_1':entry_1.get()}
    with open('paraters.pickle','wb') as f:
        pickle.dump(vars,f)

def load_parameters():
    try:
        with open('paraters.pickle','rb') as f:
            vars=pickle.load(f)
        imageB.var.set(vars['imgB'])
        imageA.var.set(vars['imgA'])
        splitB.var.set(vars['splitB'])
        splitA.var.set(vars['splitA'])
        splitD.var.set(vars['splitD'])
        merge.var.set(vars['merge'])
        outPut.var.set(vars['outPut'])
        entry.config(textvariable=vars['entry'])
        entry_1.config(textvariable=vars['entry_1'])
    except FileNotFoundError:
        pass

imageB = Chunck('Select before')
imageB.select_image(0)
imageA = Chunck('Select after')
imageA.select_image(1)
splitB = Chunck('Split before')
splitB.select_dir(2)
splitA = Chunck('Split after')
splitA.select_dir(3)
splitD = Chunck('Split diff')
splitD.select_dir(4)
merge = Chunck('Merge')
merge.select_dir(5)
outPut = Chunck('Output path')
outPut.select_dir(6)
label = tk.Label(window, text='Output name')
label.place(x=15, y=15 + 280)
entry = tk.Entry(window, width=15)
entry.place(x=120, y=20 + 280)
label_1 = tk.Label(window, text='train or predict')
label_1.place(x=15, y=15 + 320)
entry_1 = tk.Entry(window, width=15)
entry_1.place(x=120, y=15 + 320)

button=tk.Button(window,text='run',command=run)
button.place(x=15,y=15+360)

menubar=tk.Menu(window)
filemenu=tk.Menu(menubar,tearoff=0)
menubar.add_cascade(label='File',menu=filemenu)
filemenu.add_command(label='Save parameters',command=save_parameters)
filemenu.add_command(label='Load parameters',command=load_parameters)

window.config(menu=menubar)
window.mainloop()
