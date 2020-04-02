import numpy as np
import  warnings
import os
import tensorflow as tf
warnings.filterwarnings('ignore')

def load_and_preprocess_image(path):
    # 读取图片
    image = tf.io.read_file(path)
    # 将jpg格式的图片解码，得到一个张量（三维的矩阵）
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image , tf.float32)
    image = tf.image.resize(image, [100, 100 ])
    image /= 255.0
    # 转换维度
    image = tf.transpose( image , perm= (2, 0,1))
    return image


base = os.path.abspath("../Dataset/")

basedirlist = sorted(os.path.join(base, path)
                     for path in os.listdir(base) if path != ".DS_Store")
all_images_paths = []
length = []
for dir_file in basedirlist:
    for file in os.listdir(dir_file):
        imgae_file_path = dir_file + '\\' + file
        all_images_paths.append(imgae_file_path)
    length.append(len(os.listdir(dir_file)))

all_image_labels =[]
for  i in range(10):
    all_image_labels.extend( [i]* length[i] )


path_ds = tf.data.Dataset.from_tensor_slices(np.array(all_images_paths)) #路径字符串集合
image_ds = path_ds.map(load_and_preprocess_image)

# 构建类标数据的“dataset”
label_ds = tf.data.Dataset.from_tensor_slices(
    tf.cast(all_image_labels, tf.int64))
# 将图片和类标压缩为（图片，类标）对
image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))


# shape 查看
image_label_ds = image_label_ds.shuffle(10000).repeat(10).batch(128)
for i , (x ,y ) in enumerate( image_label_ds.take(1)):
    print (  x.numpy().shape , y.numpy().shape)


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(
        input_shape=(  3, 100, 100 ) ,
        filters = 64 ,
        kernel_size = (3, 3 ),
        padding='SAME' ,
        activation='relu'),
     tf.keras.layers.Conv2D(
        filters = 128 ,
        kernel_size = (3, 3 ),
        padding='SAME' ,
        activation='relu'),
    tf.keras.layers.MaxPool2D( (2,2 ) ),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense( 1024 , activation='relu') ,
    tf.keras.layers.Dense(10)
])


model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'] ,
 )


history  = model.fit(image_label_ds  ,
                     epochs  = 10 )
