import os, glob
import tkinter as tk
import tkinter.messagebox
import pickle
import re
from osgeo import gdal
from sklearn.cluster import KMeans


import main
import utils
import core
import numpy as np
from views import CreateAbout, CreateRun, CreateModel, CreaterBasic
from constant import Constant

prefix = 'split'
merge_name = 'merge.tif'
flag = False


def renew_constant():
    if model_frame.train_num.get():
        Constant.train_num = int(model_frame.train_num.get())
    if model_frame.max_iters.get():
        Constant.max_iters = int(model_frame.max_iters.get())
    if model_frame.lr.get():
        Constant.lr = float(model_frame.lr.get())


def check_tif(entry):
    global flag
    content = entry.get().split('.')[-1]
    if content != 'tif':
        flag = True
        tk.messagebox.showerror(title='Format error', message='The image should be tif format')


def check_blank(entry):
    global flag
    content = entry.get()
    if not content:
        flag = True
        tk.messagebox.showwarning(title='Warning', message='There are some infomation should be completed')


def check_number(entry):
    """
    Example:train_num=200,max_iters=1000,lr=1e-5
    :param entry:
    :return:
    """
    global flag
    content = entry.get()
    if not content or re.match(r'^-?\d*\.?\d*(e-?\d+)?$', content):
        pass
    else:
        flag = True
        tk.messagebox.showerror(title='Format error', message='Your input should be numbers')


def run():
    # error checking
    global flag
    flag = False
    waiting_check = [basic_frame.imageB.entry, basic_frame.imageA.entry, basic_frame.splitB.entry,
                     basic_frame.splitA.entry, basic_frame.splitD.entry,
                     basic_frame.merge.entry, basic_frame.entry, basic_frame.entry_1, model_frame.model]
    for i in waiting_check:
        check_blank(i)
        if flag:
            return
    waiting_check = [basic_frame.imageB.entry, basic_frame.imageA.entry]
    for i in waiting_check:
        check_tif(i)
        if flag:
            return
    waiting_check = [model_frame.train_num, model_frame.max_iters, model_frame.lr]
    for i in waiting_check:
        check_number(i)
        if flag:
            return

    renew_constant()
    train_num, max_iters, lr = Constant.train_num, Constant.max_iters, Constant.lr
    model_path = model_frame.model.get()
    # split
    img_path_before = basic_frame.imageB.entry.get()
    img_path_after = basic_frame.imageA.entry.get()
    split_path_before = basic_frame.splitB.entry.get()
    split_path_after = basic_frame.splitA.entry.get()
    split_diff = basic_frame.splitD.entry.get()
    merge_path = basic_frame.merge.entry.get()
    out_put_path = basic_frame.outPut.entry.get()
    out_put_name = basic_frame.entry.get()

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
        train_or_pre = basic_frame.entry_1.get()
        if train_or_pre == '1' or train_or_pre == '2':
            break
        else:
            print('输入错误')
    counter = 1
    run_frame.progressbar['maximum']=len(tifs_after)
    run_frame.progressbar['value']=0
    for tif_before, tif_after in zip(tifs_before, tifs_after):
        #show progress
        run_frame.progressbar['value']+=1
        run_frame.run_frame.update()

        counter += 1
        print('the {} time'.format(counter))
        print('path:', tif_before, tif_after)
        params = utils.load_dataset(split_path_before + '/' + tif_before, split_path_after + '/' + tif_after,
                                    train_or_pre)
        X, Y, diff, row, column, projection, minx, maxy, xres = params['before'], params['after'], \
            params['diff'], params['row'], params[
            'column'], params['projection'], \
            params['minX'], params['maxY'], params['resolution']

        if train_or_pre == '1':
            core.main(X, Y, row, column, projection, minx, maxy, xres, str(counter), split_diff + '/', model_path,
                      train_num,
                      max_iters, lr, diff=diff)

        elif train_or_pre == '2':
            core.main(X, Y, row, column, projection, minx, maxy, xres, str(counter), split_diff + '/', model_path,
                      train_num,
                      max_iters, lr, flag='pre')

    del X, Y, diff

    ########
    # MERGE#
    ########
    main.GRID.merge_image(split_diff + '/', merge_path + '/' + merge_name)

    ###########
    # 阈值分割###
    ###########
    diff = gdal.Open(merge_path + '/' + merge_name)
    projection = diff.GetProjection()
    minx, xres, xskew, maxy, yskew, yres = diff.GetGeoTransform()
    img_width = diff.RasterXSize  # image width
    img_height = diff.RasterYSize  # image height
    diff = diff.ReadAsArray(0, 0, img_width, img_height)
    diff = diff.flatten()
    diff = np.expand_dims(diff, axis=1)
    bin = KMeans(n_clusters=2).fit(diff).labels_
    out_CD = out_put_path + '/' + out_put_name + '.tif'
    print('开始输出阈值分割图片...')
    core.outputTif(out_CD, img_width, img_height, minx, maxy, xres, projection, bin.reshape(img_height, img_width))
    print('阈值输出完成!')

    ###################
    # 删除临时分割的影像##
    ###################
    print('清理临时文件中...')
    for file in glob.glob(split_diff + '/' + '*'):
        os.remove(file)
    for file in glob.glob(split_path_before + '/' + '*'):
        os.remove(file)
    for file in glob.glob(split_path_after + '/' + '*'):
        os.remove(file)
    for file in glob.glob(merge_path + '/' + '*'):
        os.remove(file)
    print('清理完成!')


def save_parameters():
    vars = {'imgB': basic_frame.imageB.entry.get(), 'imgA': basic_frame.imageA.entry.get(),
            'splitB': basic_frame.splitB.entry.get(),
            'splitA': basic_frame.splitA.entry.get(), 'splitD': basic_frame.splitD.entry.get(),
            'merge': basic_frame.merge.entry.get(),
            'outPut': basic_frame.outPut.entry.get(), 'entry': basic_frame.entry.get(),
            'entry_1': basic_frame.entry_1.get()}
    with open('paraters.pickle', 'wb') as f:
        pickle.dump(vars, f)


def load_parameters():
    try:
        with open('paraters.pickle', 'rb') as f:
            vars = pickle.load(f)
        basic_frame.imageB.var.set(vars['imgB'])
        basic_frame.imageA.var.set(vars['imgA'])
        basic_frame.splitB.var.set(vars['splitB'])
        basic_frame.splitA.var.set(vars['splitA'])
        basic_frame.splitD.var.set(vars['splitD'])
        basic_frame.merge.var.set(vars['merge'])
        basic_frame.outPut.var.set(vars['outPut'])
        basic_frame.entry.config(textvariable=vars['entry'])
        basic_frame.entry_1.config(textvariable=vars['entry_1'])
    except FileNotFoundError:
        pass


def show_about():
    about_frame.about_frame.pack()
    run_frame.run_frame.pack_forget()
    model_frame.model_frame.pack_forget()
    basic_frame.basic_frame.pack_forget()


def show_run():
    run_frame.run_frame.pack()
    about_frame.about_frame.pack_forget()
    model_frame.model_frame.pack_forget()
    basic_frame.basic_frame.pack_forget()


def show_model():
    model_frame.model_frame.pack()
    about_frame.about_frame.pack_forget()
    run_frame.run_frame.pack_forget()
    basic_frame.basic_frame.pack_forget()


def show_basic():
    model_frame.model_frame.pack_forget()
    about_frame.about_frame.pack_forget()
    run_frame.run_frame.pack_forget()
    basic_frame.basic_frame.pack()


window = tk.Tk()
window.title('DSFANet Change Detection')
window.geometry('500x500')
basic_frame = CreaterBasic(window)
about_frame = CreateAbout(window)
run_frame = CreateRun(window, run)
model_frame = CreateModel(window)

menubar = tk.Menu(window)
filemenu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label='File', menu=filemenu)
filemenu.add_command(label='Save parameters', command=save_parameters)
filemenu.add_command(label='Load parameters', command=load_parameters)

menubar.add_command(label='Basic', command=show_basic)
menubar.add_command(label='Model', command=show_model)
menubar.add_command(label='Run', command=show_run)
menubar.add_command(label='About', command=show_about)

window.config(menu=menubar)
window.mainloop()
