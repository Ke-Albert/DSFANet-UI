import tensorflow as tf
import keras
from keras import layers
class DSFANet(object):
    def __init__(self, num=None,output_num=6,hidden_num=128,layers=2,
                 reg=1e-4,activation=tf.nn.softsign,init=tf.initializers.he_normal):
        self.num = num
        self.output_num = output_num
        self.hidden_num = hidden_num
        self.layers = layers
        self.reg = reg
        self.activation = activation
        self.init = init

    def DSFA(self, X, Y):

        #m, n = tf.shape(X)
        X_hat = X - tf.reduce_mean(X, axis=0)
        Y_hat = Y - tf.reduce_mean(Y, axis=0)

        differ = X_hat - Y_hat

        A = tf.matmul(differ, differ, transpose_a=True)
        A = A/self.num

        Sigma_XX = tf.matmul(X_hat, X_hat, transpose_a=True)
        Sigma_XX = Sigma_XX / self.num + self.reg * tf.eye(self.output_num)
        Sigma_YY = tf.matmul(Y_hat, Y_hat, transpose_a=True)
        Sigma_YY = Sigma_YY / self.num + self.reg * tf.eye(self.output_num)

        B = (Sigma_XX+Sigma_YY)/2

        # For numerical stability.
        D_B, V_B = tf.compat.v1.self_adjoint_eig(B)
        idx = tf.where(D_B > 1e-12)[:, 0]
        D_B = tf.gather(D_B, idx)
        V_B = tf.gather(V_B, idx, axis=1)
        B_inv = tf.matmul(tf.matmul(V_B, tf.compat.v1.diag(tf.compat.v1.reciprocal(D_B))), tf.transpose(V_B))
        ##

        Sigma = tf.matmul(B_inv, A)
        loss = tf.compat.v1.trace(tf.matmul(Sigma, Sigma))

        return loss

    def forward(self, X, Y):
        model_x=keras.Sequential([layers.Dense(self.hidden_num,activation=self.activation,input_shape=[1])])
        model_y=keras.Sequential([layers.Dense(self.hidden_num,activation=self.activation,input_shape=[1])])
        for k in range(self.layers-1):
            model_x.add(layers.Dense(self.hidden_num,activation=self.activation))
            model_y.add(layers.Dense(self.hidden_num, activation=self.activation))
        model_x.add(layers.Dense(self.output_num,activation=self.activation))
        model_y.add(layers.Dense(self.output_num,activation=self.activation))
        self.X_=model_x(X)
        self.Y_=model_y(Y)
