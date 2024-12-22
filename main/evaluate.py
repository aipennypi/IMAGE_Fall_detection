#set up the path
val_img_files = os.listdir('./falldetection/fall_dataset/images/val')
val_img_files.sort()
r3 = './falldetection/fall_dataset/images/val/'
val_label_files = os.listdir('./falldetection/fall_dataset/labels/val')
val_label_files.sort()
r4 = './falldetection/fall_dataset/labels/val/'

complete_imagesV = []
complete_classV = []
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, add, GlobalAveragePooling2D, Dense, Dropout,MaxPool2D
from tensorflow.keras import Model
import tensorflow as tf
from tensorflow import keras

for i in range(len(val_img_files)):
    img = plt.imread(r3+val_img_files[i])
    with open(r4+val_label_files[i],'r') as file:
        r = file.readlines()
    bounding_boxes = []
    for j in r:
        j = j.split()
        bounding_boxes.append([int(j[0]),float(j[1]),float(j[2]),float(j[3]),float(j[4])])
    for box in bounding_boxes:
        image_height, image_width, _ = img.shape
        xmin, ymin, width, height = box[1:]
        xmin = int(xmin * image_width)
        ymin = int(ymin * image_height)
        width = int(width * image_width)
        height = int(height * image_height)
        complete_classV.append(box[0])
        complete_imagesV.append(img[ymin-height//2:ymin+height//2, xmin-width//2:xmin+width//2])

pref_size = (224,224)
for i in range(len(complete_imagesV)):
    complete_imagesV[i] = cv2.resize(complete_imagesV[i],pref_size)

df_V = pd.DataFrame()
df_V['Images'] = complete_imagesV
df_V['Class'] = complete_classV
df_V['Images']/=255

X_test = np.array(df_V['Images'].tolist())
y_test = np.array(df_V['Class'])
#load the model
model =  tf.keras.models.load_model(('my_model.h5') # your model path
model.summary()
#evaluate the model
model.evaluate(X_test,y_test)
