import tensorflow as tf
from tensorflow.keras import Sequential, layers, losses, optimizers, datasets
import numpy as np
import matplotlib.pyplot as plt
from Moduless import Read_Dataset
# import pydot
# import os

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
    
devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], True)
print(devices)
train_x, train_y, test_x, test_y  = Read_Dataset.read_dataset()

def cross_entropy(inputs):
    mtx = inputs*np.log(inputs)
    return -tf.math.reduce_sum(mtx)

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

    def call(self,inputs,ttraining=True,**kwargs):
        stop_predict = False
        x = self.convolution_1(inputs)
        x = self.activation(x)
        ee_layer_1 = self.flatten(x)
        ee_layer_1_output = self.output_layer(ee_layer_1)
        info = {"Exit_1_entropy":0.0,"Exit_2_entropy":0.0,"Exit_3_entropy":0.0,"Exit_4_entropy":0.0,"Exit_5_entropy":0.0,"Exit_6_entropy":0.0}
        if(ttraining == False):
            val = cross_entropy(ee_layer_1_output)
            info.update({"Exit_1_entropy":val})
            # print("No.1 soft cross entropy",val)
            if(val<0.2):
                stop_predict = True
                return ee_layer_1_output,info
        
        if(stop_predict != True):
            x = self.averagepooling(x)
            ee_layer_2 = self.flatten(x)
            ee_layer_2_output = self.output_layer(ee_layer_2)
            if(ttraining == False):
                val = cross_entropy(ee_layer_2_output)
                info.update({"Exit_2_entropy":val})
                # print("No.2 soft cross entropy",val)
                if(val<0.2):
                    stop_predict = True
                    return ee_layer_2_output,info

        if(stop_predict != True):
            x = self.convolution_2(x)
            x = self.activation(x)
            ee_layer_3 = self.flatten(x)
            ee_layer_3_output = self.output_layer(ee_layer_3)
            if(ttraining == False):
                val = cross_entropy(ee_layer_3_output)
                info.update({"Exit_3_entropy":val})
                # print("No.3 soft cross entropy",val)
                if(val<0.2):
                    stop_predict = True
                    return ee_layer_3_output,info
                
        if(stop_predict != True):    
            x = self.averagepooling(x)
            x = self.flatten(x)
            ee_layer_4_output = self.output_layer(x)
            if(ttraining == False):
                val = cross_entropy(ee_layer_4_output)
                info.update({"Exit_4_entropy":val})
                # print("No.4 soft cross entropy",val)
                if(val<0.2):
                    stop_predict = True
                    return ee_layer_4_output,info
                
        if(stop_predict != True):
            x = self.Dense_1(x)
            ee_layer_5_output = self.output_layer(x)
            if(ttraining == False):
                val = cross_entropy(ee_layer_5_output)
                info.update({"Exit_5_entropy":val})
                # print("No.5 soft cross entropy",val)
                if(val<0.2):
                    stop_predict = True
                    return ee_layer_5_output,info
                
        if(stop_predict != True):
            x = self.Dense_2(x)
            output_ = self.output_layer(x)
            if(ttraining==False):
                val = cross_entropy(output_)
                info.update({"Exit_6_entropy":val})
                
                return output_,info
            else:
                return [ee_layer_1_output,ee_layer_2_output,ee_layer_3_output,ee_layer_4_output,ee_layer_5_output,output_]
    def build_graph(self):
        x = layers.Input(shape=(32,32,1))
        return tf.keras.Model(inputs=[x],outputs=self.call(x))
model = Early_Exit_Model()

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=40,
    zoom_range=0.3,
    width_shift_range=0.5,
    height_shift_range=0.5)


model.build(input_shape=(None,32,32,1))
model.build_graph()
model.summary() # 看建立的架構
tf.keras.utils.plot_model(model, to_file=".\images\my_first_model_with_shape_info.png", show_shapes=True)

print(tf.config.list_physical_devices('GPU'))
model.compile(loss="categorical_crossentropy",optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),metrics=["accuracy"]) # 定義所要採用的loss funtion, optimizer, metrics
history = model.fit_generator(datagen.flow(train_x,train_y,batch_size=32),steps_per_epoch=len(train_x)/32,epochs=16,validation_data=(test_x,test_y))
print(train_x.shape)
# history = model.fit(x=train_x,y=train_y,batch_size=8,epochs=16,verbose=1,validation_split=0.1) # 設定 batch(批), epochs(跌代), verbose, validation(驗證，功能還不太確定)

# plt.plot(history.history['output_1_accuracy'], label='output_1_accuracy')
# plt.plot(history.history['val_output_1_accuracy'], label = 'val_output_1_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.grid(True)
# # plt.ylim([0.5, 1])
# plt.legend(loc='lower right')
# plt.show()
# plt.plot(history.history['output_2_accuracy'], label='output_2_accuracy')
# plt.plot(history.history['val_output_2_accuracy'], label = 'val_output_2_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.grid(True)
# # plt.ylim([0.5, 1])
# plt.legend(loc='lower right')
# plt.show()
# plt.plot(history.history['output_3_accuracy'], label='output_3_accuracy')
# plt.plot(history.history['val_output_3_accuracy'], label = 'val_output_3_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.grid(True)
# # plt.ylim([0.5, 1])
# plt.legend(loc='lower right')
# plt.show()
# plt.plot(history.history['output_4_accuracy'], label='output_4_accuracy')
# plt.plot(history.history['val_output_4_accuracy'], label = 'val_output_4_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.grid(True)
# # plt.ylim([0.5, 1])
# plt.legend(loc='lower right')
# plt.show()
# plt.plot(history.history['output_5_accuracy'], label='output_5_accuracy')
# plt.plot(history.history['val_output_5_accuracy'], label = 'val_output_5_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.grid(True)
# # plt.ylim([0.5, 1])
# plt.legend(loc='lower right')
# plt.show()


import time
# result = model.call(test_x[90:99],ttraining=True)
# print(np.argmax(result,axis=-1))
start = time.time()
result,info = model.call(test_x[90:99],ttraining=False)
end = time.time()
# result = model.predict(test_x[0:9])
print("use_time: ",end-start)
print('前9筆預測結果: ',np.argmax(result, axis=-1),'\n')
print('前9筆實際值: ',np.argmax(test_y[90:99],axis=-1),'\n')
print("比對結果",np.argmax(result, axis=-1)-np.argmax(test_y[90:99],axis=-1))


import cv2
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


import csv
csvfile = open('info.csv','a+')
r = csv.writer(csvfile)
r.writerow(['',"Exit_1_entropy","Exit_2_entropy","Exit_3_entropy","Exit_4_entropy","Exit_5_entropy","predict_number"])
test_times = 0

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
        # img = np.array(img)
        # img = np.pad(img,pad_width=((2,2),(2,2)),mode='constant',constant_values=0)
        img = img/255
        img = np.expand_dims(img,0)
        img = np.expand_dims(img,-1)
        # np.savetxt("show_data.txt",img[0,...,0],fmt='%.01f')
        # print(img)
        print(img.shape)
        # model.summary()
        # predict = model(img)
        # predict = model.predict(img)
        # print(predict)
        
        # predict = model.call(img,ttraining=True)
        # print(np.argmax(predict,axis=-1))
        start = time.time()
        predict,info = model.call(img,ttraining=False)
        end = time.time()
        print("use_time :",end-start)
        predict_number = np.argmax(predict,axis=-1)
        print('預測結果:\n',predict_number)
        # print(info)
        test_times+=1
        r.writerow([test_times,np.float32(info["Exit_1_entropy"]),np.float32(info["Exit_2_entropy"]),np.float32(info["Exit_3_entropy"]),np.float32(info["Exit_4_entropy"]),np.float32(info["Exit_4_entropy"]),predict_number])
        draw = np.zeros((h,w,3), dtype='uint8')
    if keyboard == ord('r'):
        draw = np.zeros((h,w,3), dtype='uint8')  # 按下 r 就變成原本全黑的畫布
        cv2.imshow('img', draw)


model.save('./Model/CNN_MODEL')
model.save_weights('./Model/SAVE_WEIGHTS/SW')
# LeNet-5第三層自訂_subclassing實作early exit\CNN_WEIGHTS