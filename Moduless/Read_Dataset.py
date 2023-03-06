import tensorflow as tf
import numpy as np
import cv2
def read_dataset():
    (train_x, train_y),(test_x, test_y) = tf.keras.datasets.mnist.load_data()
    train_x = 2*tf.convert_to_tensor(train_x,dtype=tf.float32)
    train_x = tf.pad(train_x,[[0,0],[2,2],[2,2]],"CONSTANT",0) # 外圍填充
    train_x = train_x / 255 # 圖像歸一化 0~1
    train_x = tf.expand_dims(train_x,-1)
    train_y = tf.one_hot(train_y, depth=10)

    test_x = 2*tf.convert_to_tensor(test_x,dtype=tf.float32)
    test_x = tf.pad(test_x,[[0,0],[2,2],[2,2]],"CONSTANT",0) # 外圍填充
    test_x = test_x / 255 # 圖像歸一化 0~1
    test_x = tf.expand_dims(test_x,-1)
    test_y = tf.one_hot(test_y, depth=10)
    return train_x,train_y,test_x,test_y

if __name__ == "__main__":
    train_x,train_y,test_x,test_y = read_dataset()
    # images = np.reshape(train_x,(60000,32,32))
    print(train_x.shape)
    show_img = np.array(train_x[1,...,0])
    show_img = cv2.resize(show_img,(320,320))
    cv2.imshow('img',show_img) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()
