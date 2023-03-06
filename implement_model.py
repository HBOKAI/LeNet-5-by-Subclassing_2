import tensorflow as tf
from tensorflow.keras import Sequential, layers, losses, optimizers, datasets
from PIL import Image
import numpy as np
import cv2
np.set_printoptions(threshold=np.inf)# np.inf = 無窮大的浮點數，若矩陣數量大於threshold部分數值會以...代替
np.set_printoptions(suppress=True)#抑制顯示小數位數

def show_xy(event,x,y,flags,param):
    global dots, draw,img_gray                    # 定義全域變數
    if flags == 1:
        if event == 1:
            dots.append([x,y])            # 如果拖曳滑鼠剛開始，記錄第一點座標
        if event == 4:
            dots = []                     # 如果放開滑鼠，清空串列內容
        if event == 0 or event == 4:
            dots.append([x,y])            # 拖曳滑鼠時，不斷記錄座標
            x1 = dots[len(dots)-2][0]     # 取得倒數第二個點的 x 座標
            y1 = dots[len(dots)-2][1]     # 取得倒數第二個點的 y 座標
            x2 = dots[len(dots)-1][0]     # 取得倒數第一個點的 x 座標
            y2 = dots[len(dots)-1][1]     # 取得倒數第一個點的 y 座標
            cv2.line(draw,(x1,y1),(x2,y2),(255,255,255),20)  # 畫直線
        cv2.imshow('img', draw)#draw

def cross_entropy(inputs):
    mtx = inputs*np.log(inputs)
    return -tf.math.reduce_sum(mtx)

class Con_sp(tf.keras.layers.Layer):

    def __init__(self, filter_num, filter_size, **kwargs):
        super(Con_sp, self).__init__(**kwargs)
        self.filter_num = filter_num
        self.filter_size = filter_size

    def build(self, input_shape): 
        self.weights3 = tf.Variable(initial_value=tf.random.normal([self.filter_size,self.filter_size,3,6]),trainable=True,name='Weight3')
        self.weights4 = tf.Variable(initial_value=tf.random.normal([self.filter_size,self.filter_size,4,6]),trainable=True,name='Weight4')
        self.weights4_4 = tf.Variable(initial_value=tf.random.normal([self.filter_size,self.filter_size,4,3]),trainable=True,name='Weight4_4')
        self.weights6 = tf.Variable(initial_value=tf.random.normal([self.filter_size,self.filter_size,6,1]),trainable=True,name='Weight6')
        self.bias1 = tf.Variable( initial_value=tf.random.normal([self.filter_num]),trainable=True,name='Bias11')
        # self.weights1 = self.add_weight(shape=(self.filter_size,self.filter_size,6,self.filter_num),initializer='random_normal',trainable=True,name='ww11')
        # self.bias1 = self.add_weight(shape=(self.filter_num),initializer='random_normal',trainable=True,name='bb11')
        # self.shape1 = input_shape
        #相当于设置self.built = True
        super(Con_sp,self).build(input_shape)

    def call(self, inputs):
        for i in range(16):
            if i < 1:
                j=i
                basic_out = tf.nn.conv2d(inputs[...,j:j+1],self.weights3[...,0:1,0:1],[1,1,1,1],'VALID')
                j+=1
                if (j/6) == 1: 
                    j=0
                basic_out += tf.nn.conv2d(inputs[...,j:j+1],self.weights3[...,1:2,0:0+1],[1,1,1,1],'VALID')
                j+=1
                if (j/6) == 1: 
                    j=0
                basic_out += tf.nn.conv2d(inputs[...,j:j+1],self.weights3[...,2:3,0:0+1],[1,1,1,1],'VALID',)
                basic_out = tf.nn.bias_add(basic_out, self.bias1[i:i+1])
                # print(basic_out.shape)
            if i>=1 and i < 6:
                j=i
                k=i
                output11 = tf.nn.conv2d(inputs[...,j:j+1],self.weights3[...,0:1,k:k+1],[1,1,1,1],'VALID')
                j+=1
                if (j/6) == 1: 
                    j=0
                output11 += tf.nn.conv2d(inputs[...,j:j+1],self.weights3[...,1:2,k:k+1],[1,1,1,1],'VALID')
                j+=1
                if (j/6) == 1: 
                    j=0
                output11 += tf.nn.conv2d(inputs[...,j:j+1],self.weights3[...,2:3,k:k+1],[1,1,1,1],'VALID')
                output11 = tf.nn.bias_add(output11, self.bias1[i:i+1])
                # print(output11)
                basic_out = tf.concat(axis=3,values=[basic_out,output11])
                # print(i,"和: ",basic_out.shape)
            if i >= 6 and i < 12 :
                j=i-6
                k=i-6
                output11 = tf.nn.conv2d(inputs[...,j:j+1],self.weights4[...,0:1,k:k+1],[1,1,1,1],'VALID')
                j+=1
                if (j/6) == 1:
                    j=0
                output11 += tf.nn.conv2d(inputs[...,j:j+1],self.weights4[...,1:2,k:k+1],[1,1,1,1],'VALID')
                j+=1
                if (j/6) == 1:
                    j=0
                output11 += tf.nn.conv2d(inputs[...,j:j+1],self.weights4[...,2:3,k:k+1],[1,1,1,1],'VALID')
                j+=1
                if (j/6) == 1:
                    j=0
                output11 += tf.nn.conv2d(inputs[...,j:j+1],self.weights4[...,3:4,k:k+1],[1,1,1,1],'VALID')
                output11 = tf.nn.bias_add(output11, self.bias1[i:i+1])
                # print(output11)
                basic_out = tf.concat(axis=3,values=[basic_out,output11])
                # print(i,"和: ",basic_out.shape)
            if i >= 12 and i < 15 :
                j=i-12
                k=i-12
                output11 = tf.nn.conv2d(inputs[...,j:j+1],self.weights4_4[...,0:1,k:k+1],[1,1,1,1],'VALID')
                j+=1
                if (j/6) == 1:
                    j=0
                output11 += tf.nn.conv2d(inputs[...,j:j+1],self.weights4_4[...,1:2,k:k+1],[1,1,1,1],'VALID')
                j+=2
                if (j/6) >= 1:
                    j-=6
                output11 += tf.nn.conv2d(inputs[...,j:j+1],self.weights4_4[...,2:3,k:k+1],[1,1,1,1],'VALID')
                j+=1
                if (j/6) == 1:
                    j=0
                output11 += tf.nn.conv2d(inputs[...,j:j+1],self.weights4_4[...,3:4,k:k+1],[1,1,1,1],'VALID')
                output11 = tf.nn.bias_add(output11, self.bias1[i:i+1])
                # print(output11)
                basic_out = tf.concat(axis=3,values=[basic_out,output11])
                # print(i,"和: ",basic_out.shape)
            if i == 15 :
                output11 = tf.nn.conv2d(inputs,self.weights6,[1,1,1,1],'VALID')
                output11 = tf.nn.bias_add(output11, self.bias1[i:i+1])
                basic_out = tf.concat(axis=3,values=[basic_out,output11])
                # print(i,"和: ",basic_out.shape)
        return basic_out
    # def get_config(self):
    #     # base_config = super(Con_sp, self).get_config
    #     # config1 = {
    #     #     "filter_num":self.filter_num ,
    #     #     "filter_size":self.filter_size
    #     #     }
    #         # dict(list(base_config.items()) + list(config1.items()))
    #     return {"filter_num":self.filter_num , "filter_size":self.filter_size}
    def get_config(self):
        config = super(Con_sp, self).get_config()
        config.update({
            "filter_num":self.filter_num,
            "filter_size":self.filter_size
        })
        return config  
 
class Early_Exit_Model(tf.keras.Model):
    def __init__(self, output_num=10, **kwargs):
        super(Early_Exit_Model,self).__init__(**kwargs)
        self.convolution_1 = layers.Conv2D(6,kernel_size=5,strides=1)
        # self.convolution_2 = Con_sp(filter_num=16,filter_size=5)
        self.convolution_2 = layers.Conv2D(16,kernel_size=5,strides=1)
        self.Dense_1 = layers.Dense(120,activation='relu')
        self.Dense_2 = layers.Dense(84,activation='relu')
        self.output_layer = layers.Dense(10,activation='softmax')
        self.activation = layers.Activation('relu')
        self.averagepooling = layers.AveragePooling2D(pool_size=2,strides=2)
        self.flatten = layers.Flatten()

        #ee_layer1
        self.ee_Dense_outputs1 = layers.Dense(10,activation='softmax')
        self.ee_Dense_outputs2 = layers.Dense(10,activation='softmax')
        self.ee_Dense_outputs3 = layers.Dense(10,activation='softmax')
        self.ee_Dense_outputs4 = layers.Dense(10,activation='softmax')

    def call(self,inputs,ttraining=True,**kwargs):
        stop_predict = False
        x = self.convolution_1(inputs)
        x = self.activation(x)
        ee_layer_1 = self.flatten(x)
        ee_layer_1_output = self.ee_Dense_outputs1(ee_layer_1)
        if(ttraining == False):
            val = cross_entropy(ee_layer_1_output)
            print("No.1 soft cross entropy",val)
            if(val<0.2):
                stop_predict = True
                return ee_layer_1_output
            
        if(stop_predict != True):
            x = self.averagepooling(x)
            x = self.convolution_2(x)
            x = self.activation(x)
            ee_layer_2 = self.flatten(x)
            ee_layer_2_output = self.ee_Dense_outputs2(ee_layer_2)
            if(ttraining == False):
                val = cross_entropy(ee_layer_2_output)
                print("No.2 soft cross entropy",val)
                if(val<0.2):
                    stop_predict = True
                    return ee_layer_2_output
                
        if(stop_predict != True):    
            x = self.averagepooling(x)
            x = self.flatten(x)
            ee_layer_3_output = self.ee_Dense_outputs3(x)
            if(ttraining == False):
                val = cross_entropy(ee_layer_3_output)
                print("No.3 soft cross entropy",val)
                if(val<0.2):
                    stop_predict = True
                    return ee_layer_3_output
                
        if(stop_predict != True):
            x = self.Dense_1(x)
            ee_layer_4_output = self.ee_Dense_outputs4(x)
            if(ttraining == False):
                val = cross_entropy(ee_layer_4_output)
                print("No.4 soft cross entropy",val)
                if(val<0.2):
                    stop_predict = True
                    return ee_layer_4_output
                
        if(stop_predict != True):
            x = self.Dense_2(x)
            output_ = self.output_layer(x)
            # outputs = self.output_layer(x)
            if(ttraining==False):
                return output_
            else:
                return [ee_layer_1_output,ee_layer_2_output,ee_layer_3_output,ee_layer_4_output,output_]
    def build_graph(self):
        x = layers.Input(shape=(32,32,1))
        return tf.keras.Model(inputs=[x],outputs=self.call(x))

model = Early_Exit_Model()
model.build(input_shape=(None,32,32,1))
model.compile(loss="categorical_crossentropy",optimizer=tf.keras.optimizers.Adam(),metrics=["accuracy"]) 
model.load_weights('./Model/SAVE_WEIGHTS/SW')

input_times = 0
correct_times = 0

dots = []   # 建立空陣列記錄座標
w = 320
h = 320
draw = np.zeros((h,w,3), dtype='uint8')   # 建立 420x240 的 RGBA 黑色畫布
while True:
    cv2.imshow('img', draw)
    cv2.setMouseCallback('img', show_xy)
    keyboard = cv2.waitKey(5)                    # 每 5 毫秒偵測一次鍵盤事件
    if keyboard == ord('q'):
        break                                    # 按下 q 就跳出

    if keyboard == ord('n'):
        img_gray = cv2.cvtColor(draw, cv2.COLOR_BGR2GRAY)   # 轉為灰度圖
        img = cv2.resize(img_gray,(32,32))                          # 變更圖片尺寸
        cv2.imwrite(".\images\gray.png",img)
        img = img/255
        img = np.expand_dims(img,0)
        img = np.expand_dims(img,-1)
        np.savetxt("show_data.txt",img[0,...,0],fmt='%.01f')
        predict = model.call(img,ttraining=True)
        print(np.argmax(predict,axis=-1))
        predict = model.call(img,ttraining=False)
        predict_num = np.argmax(predict, axis=-1)
        print('預測結果:\n',predict_num)
        innum = input("輸入正確的數字: ")
        if(int(innum) == predict_num):
            correct_times += 1
        input_times += 1
        print(f"輸入次數: {input_times}, 正確次數: {correct_times}, 正確率: {(correct_times/input_times)*100}%")
        draw = np.zeros((h,w,3), dtype='uint8')
    if keyboard == ord('r'):
        draw = np.zeros((h,w,3), dtype='uint8')  # 按下 r 就變成原本全黑的畫布
        cv2.imshow('img', draw)