from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
Model_car=load_model('ml\cnn_car_model\model_resnet50.h5')

def predict(image_path,model):

    img=image.load_img(image_path,target_size=(224,224))
    
    x=image.img_to_array(img) #----- convert into array
    x=x/255                   #----- rescale {0,1}
    x=np.expand_dims(x,axis=0) #---- adding dims for batch size
    img_data=preprocess_input(x)
    # img_data.shape
    res=np.argmax(model.predict(img_data), axis=1)
    type = {
        0 : "audi",
        1: "Lambergini",
        2: "Mercedes",
    }
    result = type[res[0]]
    return result

# print(predict(r"D:\Himanshu_data_science\Deep-Learning-Car-Brand-master\Deep-Learning-Car-Brand-master\Datasets\Test\mercedes\43.jpg",Model_car))