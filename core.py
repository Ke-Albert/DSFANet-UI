import numpy as np
import os
import tensorflow as tf
from osgeo import gdal, ogr, osr

from dsfamodel import DSFANet

# os.environ["CUDA_VISIBLE_DEVICES"]="0"
# %%
print(tf.config.list_physical_devices('GPU'))

def main(X, Y, row, column, projection, lon, lat, resolution, number, outputpath, diff=None, flag='train'):
    """
    to train model or predict changed areas, outputting a binary image.
    @params
    2 dims numpy array X and Y;
    row,column projection,resolution,lon/minx and lat/maxy of image;
    number: recording the position of split image
    diff: the difference of two images
    flag: to decide whether to train or predict
    """
    train_num = 10000 # 2000
    # 训练数量
    max_iters = 10000
    # 迭代次数
    lr = 1e-5

    tf.compat.v1.disable_eager_execution()

    ### 用于清除默认图形堆栈并重置全局默认图形####
    tf.compat.v1.reset_default_graph()

    inputX = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, X.shape[-1]])
    inputY = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, Y.shape[-1]])
    model = DSFANet(num=train_num)
    loss = model.forward(X=inputX, Y=inputY)
    tf.compat.v1.disable_eager_execution()
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(lr).minimize(loss)
    init = tf.compat.v1.global_variables_initializer()

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    # create a saver used to save model
    saver = tf.compat.v1.train.Saver()
    sess = tf.compat.v1.Session(config=config)
    sess.run(init)

    if flag == 'train':
        index = np.argsort(diff)

        XData = X[index[0:train_num], :]
        # XData = X;
        # 获取X影像上前2000个会变化的点的位置数据
        YData = Y[index[0:train_num], :]
        # YData = Y;
        # 获取X影像上前2000个会变化的点的位置数据

        # check if having the saved model, then restore its weights and bias
        if os.path.exists("Model/Gfzhengzhou/"):
            print('导入模型权重')
            saver.restore(sess, "Model/Gfzhengzhou/Gfzhengzhou")
            print('模型权重导入成功!')
        train_loss = np.zeros(max_iters)
        for k in range(max_iters):
            _, train_loss[k] = sess.run([optimizer, loss], feed_dict={inputX: XData, inputY: YData})
            if k % 1000 == 0:
                print('iter %4d, loss is %.4f' % (k, train_loss[k]))

        saver.save(sess, "Model/Gfzhengzhou/Gfzhengzhou")  # save trained model

        # return 0
        XTest, YTest = sess.run([model.X_, model.Y_], feed_dict={inputX: X, inputY: Y})
        sess.close()

        diff = XTest - YTest
        diff = diff / np.std(diff, axis=0)

        out_dif = outputpath
        print(row, column)
        diff = (diff ** 2).sum(axis=1).reshape(row, column)
        outputTif(out_dif + number + '.tif', column, row, lon, lat, resolution, projection, diff)
    else:
        try:
            if os.path.exists("Model/Gfzhengzhou/"):
                print('导入模型权重')
                saver.restore(sess, "Model/Gfzhengzhou/Gfzhengzhou")
                print('模型权重导入成功!')
        except:
            raise Exception('模型导入失败!')
        XTest, YTest = sess.run([model.X_, model.Y_], feed_dict={inputX: X, inputY: Y})
        sess.close()

        diff = XTest - YTest
        diff = diff / np.std(diff, axis=0)
        out_dif = outputpath

        print(row,column)
        diff = (diff ** 2).sum(axis=1).reshape(row, column)
        outputTif(out_dif + number + '.tif', column, row, lon, lat, resolution, projection,diff)
def outputTif(out_diff, row, column, lon, lat, resolution, projection, data):
    """tranform the array data into tif image
    row,column is the original image's row and column
    lon,lat,resolution,projection can be obtained using gdal
    """
    driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
    dataset = driver.Create(out_diff, row, column, 1, gdal.GDT_Float32)
    dataset.SetGeoTransform((lon, resolution, 0, lat, 0, -resolution))  # 写入仿射变换参数
    dataset.SetProjection(projection)  # 写入投
    print(data.shape)
    dataset.GetRasterBand(1).WriteArray(data)  # 写入数组数据
    # del dataset