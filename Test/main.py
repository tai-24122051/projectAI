# import os
# import cv2
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# import pytesseract as pt
# import plotly.express as px
# import matplotlib.pyplot as plt
# import xml.etree.ElementTree as xet
#
# from glob import glob
# from skimage import io
# from shutil import copy
# from tensorflow.keras.models import Model
# from tensorflow.keras.callbacks import TensorBoard
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.applications import InceptionResNetV2
# from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
#
# pt.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# glob('Automatic-License-Plate-Detection/images/*.xml')
#
# path = glob('Automatic-License-Plate-Detection/images/*.xml')
# labels_dict = dict(filepath=[],xmin=[],xmax=[],ymin=[],ymax=[])
# for i in path:
#     info = xet.parse(i)
#     root = info.getroot()
#     member_object = root.find('object')
#     labels_info = member_object.find('bndbox')
#     xmin = int(labels_info.find('xmin').text)
#     xmax = int(labels_info.find('xmax').text)
#     ymin = int(labels_info.find('ymin').text)
#     ymax = int(labels_info.find('ymax').text)
#
#     labels_dict['filepath'].append(i)
#     labels_dict['xmin'].append(xmin)
#     labels_dict['xmax'].append(xmax)
#     labels_dict['ymin'].append(ymin)
#     labels_dict['ymax'].append(ymax)
#
# df = pd.DataFrame(labels_dict)
# df.to_csv('labels.csv',index=False)
# df.head()
#
# filename = df['filepath'][0]
# def getFilename(filename):
#     filename_image = xet.parse(filename).getroot().find('filename').text
#     filepath_image = os.path.join('Automatic-License-Plate-Detection/images',filename_image)
#     return filepath_image
# getFilename(filename)
#
# image_path = list(df['filepath'].apply(getFilename))
# image_path[:10]#random check
#
# file_path = image_path[0]
# img = cv2.imread(file_path)
# img = io.imread(file_path)
# fig = px.imshow(img)
# fig.update_layout(width=600, height=500, margin=dict(l=10, r=10, b=10, t=10),xaxis_title='Figure 8')
# fig.add_shape(type='rect',x0=795, x1=1095, y0=751, y1=840, xref='x', yref='y',line_color='cyan')
#
# # Convert the labels_dict into a Pandas DataFrame
# labels_df = pd.DataFrame(labels_dict)
#
# # Now you can use .iloc on the DataFrame
# labels_df.iloc[:, 1:].values
#
# #Targeting all our values in array selecting all columns
# labels = df.iloc[ : , 1 : ].values
# data = []
# output = []
# for ind in range(len(image_path)):
#     image = image_path[ind]
#     img_arr = cv2.imread(image)
#     h,w,d = img_arr.shape
#     # Prepprocesing
#     load_image = load_img(image,target_size=(224,224))
#     load_image_arr = img_to_array(load_image)
#     norm_load_image_arr = load_image_arr/255.0 # Normalization
#     # Normalization to labels
#     xmin,xmax,ymin,ymax = labels[ind]
#     nxmin,nxmax = xmin/w,xmax/w
#     nymin,nymax = ymin/h,ymax/h
#     label_norm = (nxmin,nxmax,nymin,nymax) # Normalized output
#     # Append
#     data.append(norm_load_image_arr)
#     output.append(label_norm)
#
# (1, (0.7997159090909091, 0.9588068181818182, 0.6272727272727273, 0.7212121212121212))
# # Convert data to array
# X = np.array(data,dtype=np.float32)
# y = np.array(output,dtype=np.float32)
#
# # Split the data into training and testing set using sklearn.
# x_train,x_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=0)
# x_train.shape,x_test.shape,y_train.shape,y_test.shape
#
# inception_resnet = InceptionResNetV2(weights="imagenet",include_top=False, input_tensor=Input(shape=(224,224,3)))
# # ---------------------
# headmodel = inception_resnet.output
# headmodel = Flatten()(headmodel)
# headmodel = Dense(500,activation="relu")(headmodel)
# headmodel = Dense(250,activation="relu")(headmodel)
# headmodel = Dense(4,activation='sigmoid')(headmodel)
#
#
# # ---------- model
# model = Model(inputs=inception_resnet.input,outputs=headmodel)
#
# # Complie model
# model.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
# model.summary()
#
# # tfb = TensorBoard('object_detection')
# # history = model.fit(x=x_train,y=y_train,batch_size=10,epochs=100,validation_data=(x_test,y_test),callbacks=[tfb])
# # model.save('model.keras')
#
# # Load model
# model = tf.keras.models.load_model('model.keras')
# print('Model loaded Sucessfully')
#
# path = 'Automatic-License-Plate-Detection/images/N3.jpeg'
# image = load_img(path) # PIL object
# image = np.array(image,dtype=np.uint8) # 8 bit array (0,255)
# image1 = load_img(path,target_size=(224,224))
# image_arr_224 = img_to_array(image1)/255.0  # Convert into array and get the normalized output
#
# # Size of the orginal image
# h,w,d = image.shape
# print('Height of the image =',h)
# print('Width of the image =',w)
#
# fig = px.imshow(image)
# fig.update_layout(width=700, height=500,  margin=dict(l=10, r=10, b=10, t=10), xaxis_title='Figure 13 - TEST Image')
#
# image_arr_224.shape
#
# test_arr = image_arr_224.reshape(1,224,224,3)
# test_arr.shape
#
# # Make predictions
# coords = model.predict(test_arr)
# coords
#
# # Denormalize the values
# denorm = np.array([w,w,h,h])
# coords = coords * denorm
# coords
#
# coords = coords.astype(np.int32)
# coords
#
# # Draw bounding on top the image
# xmin, xmax,ymin,ymax = coords[0]
# pt1 =(xmin,ymin)
# pt2 =(xmax,ymax)
# print(pt1, pt2)
#
# cv2.rectangle(image,pt1,pt2,(0,255,0),3)
# fig = px.imshow(image)
# fig.update_layout(width=700, height=500, margin=dict(l=10, r=10, b=10, t=10))
#
# # Create pipeline
# path = 'Automatic-License-Plate-Detection/images/N5.jpeg'
# def object_detection(path):
#
#     # Read image
#     image = load_img(path) # PIL object
#     image = np.array(image,dtype=np.uint8) # 8 bit array (0,255)
#     image1 = load_img(path,target_size=(224,224))
#
#     # Data preprocessing
#     image_arr_224 = img_to_array(image1)/255.0 # Convert to array & normalized
#     h,w,d = image.shape
#     test_arr = image_arr_224.reshape(1,224,224,3)
#
#     # Make predictions
#     coords = model.predict(test_arr)
#
#     # Denormalize the values
#     denorm = np.array([w,w,h,h])
#     coords = coords * denorm
#     coords = coords.astype(np.int32)
#
#     # Draw bounding on top the image
#     xmin, xmax,ymin,ymax = coords[0]
#     pt1 =(xmin,ymin)
#     pt2 =(xmax,ymax)
#     print(pt1, pt2)
#     cv2.rectangle(image,pt1,pt2,(0,255,0),3)
#     return image, coords
#
# image, cods = object_detection(path)
#
# fig = px.imshow(image)
# fig.update_layout(width=700, height=500, margin=dict(l=10, r=10, b=10, t=10),xaxis_title='Figure 14')
#
# img = np.array(load_img(path))
# xmin ,xmax,ymin,ymax = cods[0]
# roi = img[ymin:ymax,xmin:xmax]
# fig = px.imshow(roi)
# fig.update_layout(width=350, height=250, margin=dict(l=10, r=10, b=10, t=10),xaxis_title='Figure 15 Cropped image')
#
# # extract text from image
# text = pt.image_to_string(roi)
# print(text)
#
# # parsing
# def parsing(path):
#     parser = xet.parse(path).getroot()
#     name = parser.find('filename').text
#     filename = f'yolov5/data_images/train/{name}'
#
#     # width and height
#     parser_size = parser.find('size')
#     width = int(parser_size.find('width').text)
#     height = int(parser_size.find('height').text)
#
#     return filename, width, height
# df[['filename','width','height']] = df['filepath'].apply(parsing).apply(pd.Series)
# df.head()
#
# # center_x, center_y, width , height
# df['center_x'] = (df['xmax'] + df['xmin'])/(2*df['width'])
# df['center_y'] = (df['ymax'] + df['ymin'])/(2*df['height'])
#
# df['bb_width'] = (df['xmax'] - df['xmin'])/df['width']
# df['bb_height'] = (df['ymax'] - df['ymin'])/df['height']
# df.head()
#
# import shutil
#
#
# ### split the data into train and test
# df_train = df.iloc[:200]
# df_test = df.iloc[200:]
# train_folder = 'yolov5/data_images/train'
#
# values = df_train[['filename','center_x','center_y','bb_width','bb_height']].values
# for fname, x,y, w, h in values:
#     image_name = os.path.split(fname)[-1]
#     txt_name = os.path.splitext(image_name)[0]
#
#     dst_image_path = os.path.join(train_folder,image_name)
#     dst_label_file = os.path.join(train_folder,txt_name+'.txt')
#     print(fname)
#     print(dst_image_path)
#     if os.path.exists(fname):
#       shutil.copy(fname, dst_image_path)
#     else:
#       print(f"The file {fname} does not exist.")
#
# train_folder = 'yolov5/data_images/train'
#
# values = df_train[['filename','center_x','center_y','bb_width','bb_height']].values
# for fname, x,y, w, h in values:
#     image_name = os.path.split(fname)[-1]
#     txt_name = os.path.splitext(image_name)[0]
#
#     dst_image_path = os.path.join(train_folder,image_name)
#     dst_label_file = os.path.join(train_folder,txt_name+'.txt')
#     print(fname)
#     print(dst_image_path)
#     # copy each image into the folder
#     shutil.copy(fname,dst_image_path)
#
#     # generate .txt which has label info
#     label_txt = f'0 {x} {y} {w} {h}'
#     with open(dst_label_file,mode='w') as f:
#         f.write(label_txt)
#
#         f.close()
#
# test_folder = 'yolov5/data_images/test'
#
# values = df_test[['filename','center_x','center_y','bb_width','bb_height']].values
# for fname, x,y, w, h in values:
#     image_name = os.path.split(fname)[-1]
#     txt_name = os.path.splitext(image_name)[0]
#
#     dst_image_path = os.path.join(test_folder,image_name)
#     dst_label_file = os.path.join(test_folder,txt_name+'.txt')
#
#     # copy each image into the folder
#     copy(fname,dst_image_path)
#
#     # generate .txt which has label info
#     label_txt = f'0 {x} {y} {w} {h}'
#     with open(dst_label_file,mode='w') as f:
#         f.write(label_txt)
#
#         f.close()

import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import pytesseract as pt
import plotly.express as px
import matplotlib.pyplot as plt
import xml.etree.ElementTree as xet

from glob import glob
from skimage import io
from shutil import copy
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

path = glob('Automatic-License-Plate-Detection/images/*.xml')
labels_dict = dict(filepath=[],xmin=[],xmax=[],ymin=[],ymax=[])
for filename in path:

    info = xet.parse(filename)
    root = info.getroot()
    member_object = root.find('object')
    labels_info = member_object.find('bndbox')
    xmin = int(labels_info.find('xmin').text)
    xmax = int(labels_info.find('xmax').text)
    ymin = int(labels_info.find('ymin').text)
    ymax = int(labels_info.find('ymax').text)

    labels_dict['filepath'].append(filename)
    labels_dict['xmin'].append(xmin)
    labels_dict['xmax'].append(xmax)
    labels_dict['ymin'].append(ymin)
    labels_dict['ymax'].append(ymax)

df = pd.DataFrame(labels_dict)
df.to_csv('labels.csv',index=False)
df.head()

filename = df['filepath'][0]
def getFilename(filename):
    filename_image = xet.parse(filename).getroot().find('filename').text
    filepath_image = os.path.join('Automatic-License-Plate-Detection/images',filename_image)
    return filepath_image
getFilename(filename)

image_path = list(df['filepath'].apply(getFilename))
image_path[:10]#random check

file_path = image_path[87] #path of our image N2.jpeg
img = cv2.imread(file_path) #read the image
# xmin-1804/ymin-1734/xmax-2493/ymax-1882
img = io.imread(file_path) #Read the image
fig = px.imshow(img)
fig.update_layout(width=600, height=500, margin=dict(l=10, r=10, b=10, t=10),xaxis_title='Figure 8 - N2.jpeg with bounding box')
fig.add_shape(type='rect',x0=1804, x1=2493, y0=1734, y1=1882, xref='x', yref='y',line_color='cyan')

#Targeting all our values in array selecting all columns
labels = df.iloc[:,1:].values
data = []
output = []
for ind in range(len(image_path)):
    image = image_path[ind]
    img_arr = cv2.imread(image)
    h,w,d = img_arr.shape
    # Prepprocesing
    load_image = load_img(image,target_size=(224,224))
    load_image_arr = img_to_array(load_image)
    norm_load_image_arr = load_image_arr/255.0 # Normalization
    # Normalization to labels
    xmin,xmax,ymin,ymax = labels[ind]
    nxmin,nxmax = xmin/w,xmax/w
    nymin,nymax = ymin/h,ymax/h
    label_norm = (nxmin,nxmax,nymin,nymax) # Normalized output
    # Append
    data.append(norm_load_image_arr)
    output.append(label_norm)

# Convert data to array
X = np.array(data,dtype=np.float32)
y = np.array(output,dtype=np.float32)

# Split the data into training and testing set using sklearn.
x_train,x_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=0)
x_train.shape,x_test.shape,y_train.shape,y_test.shape

inception_resnet = InceptionResNetV2(weights="imagenet",include_top=False, input_tensor=Input(shape=(224,224,3)))
# ---------------------
headmodel = inception_resnet.output
headmodel = Flatten()(headmodel)
headmodel = Dense(500,activation="relu")(headmodel)
headmodel = Dense(250,activation="relu")(headmodel)
headmodel = Dense(4,activation='sigmoid')(headmodel)


# ---------- model
model = Model(inputs=inception_resnet.input,outputs=headmodel)

# Complie model
model.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
model.summary()

tfb = TensorBoard('object_detection')
history = model.fit(x=x_train,y=y_train,batch_size=10,epochs=180,validation_data=(x_test,y_test),callbacks=[tfb])

model.save('model.keras')

# Load model
model = tf.keras.models.load_model('model.keras')
print('Model loaded Sucessfully')

path = 'Automatic-License-Plate-Detection/images/N3.jpeg'
image = load_img(path) # PIL object
image = np.array(image,dtype=np.uint8) # 8 bit array (0,255)
image1 = load_img(path,target_size=(224,224))
image_arr_224 = img_to_array(image1)/255.0  # Convert into array and get the normalized output

# Size of the orginal image
h,w,d = image.shape
print('Height of the image =',h)
print('Width of the image =',w)

fig = px.imshow(image)
fig.update_layout(width=700, height=500,  margin=dict(l=10, r=10, b=10, t=10), xaxis_title='Figure 13 - TEST Image')

image_arr_224.shape

test_arr = image_arr_224.reshape(1,224,224,3)
test_arr.shape

# Make predictions
coords = model.predict(test_arr)
coords

# Denormalize the values
denorm = np.array([w,w,h,h])
coords = coords * denorm
coords

coords = coords.astype(np.int32)
coords

# Draw bounding on top the image
xmin, xmax,ymin,ymax = coords[0]
pt1 =(xmin,ymin)
pt2 =(xmax,ymax)
print(pt1, pt2)

cv2.rectangle(image,pt1,pt2,(0,255,0),3)
fig = px.imshow(image)
fig.update_layout(width=700, height=500, margin=dict(l=10, r=10, b=10, t=10))

# Create pipeline
path = 'Automatic-License-Plate-Detection/images/N3.jpeg'


def object_detection(path):
    # Read image
    image = load_img(path)  # PIL object
    image = np.array(image, dtype=np.uint8)  # 8 bit array (0,255)
    image1 = load_img(path, target_size=(224, 224))

    # Data preprocessing
    image_arr_224 = img_to_array(image1) / 255.0  # Convert to array & normalized
    h, w, d = image.shape
    test_arr = image_arr_224.reshape(1, 224, 224, 3)

    # Make predictions
    coords = model.predict(test_arr)

    # Denormalize the values
    denorm = np.array([w, w, h, h])
    coords = coords * denorm
    coords = coords.astype(np.int32)

    # Draw bounding on top the image
    xmin, xmax, ymin, ymax = coords[0]
    pt1 = (xmin, ymin)
    pt2 = (xmax, ymax)
    print(pt1, pt2)
    cv2.rectangle(image, pt1, pt2, (0, 255, 0), 3)
    return image, coords


image, cods = object_detection(path)

fig = px.imshow(image)
fig.update_layout(width=700, height=500, margin=dict(l=10, r=10, b=10, t=10), xaxis_title='Figure 14')
img = np.array(load_img(path))
xmin ,xmax,ymin,ymax = cods[0]
roi = img[ymin:ymax,xmin:xmax]
fig = px.imshow(roi)
fig.update_layout(width=350, height=250, margin=dict(l=10, r=10, b=10, t=10),xaxis_title='Figure 15 Cropped image')

# extract text from image
text = pt.image_to_string(roi)
print(text)