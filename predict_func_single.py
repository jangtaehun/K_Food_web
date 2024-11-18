from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess_input
from tensorflow.keras.models import load_model
from dash import html
import numpy as np
from PIL import Image
import cv2
import base64
from io import BytesIO

IMAGE_SIZE = 300 
BATCH_SIZE = 32
preprocessing_func = eff_preprocess_input

unique_labels = np.load("label_mapping.npy", allow_pickle=True)
model = load_model('effi_batch_fix_best.keras')


def preprocess_images(contents):
    # Base64 디코딩
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    # OpenCV로 이미지 읽기
    np_array = np.frombuffer(decoded, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # 이미지 크기 조정 및 색상 변환
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE)) 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

    # Preprocessing 적용
    image = preprocessing_func(image)
    return np.expand_dims(image, axis=0)



# def preprocess_images(contents):
#     content_type, content_string = contents.split(',')
#     decoded = base64.b64decode(content_string)
#     image = Image.open(BytesIO(decoded)).convert('RGB')
#     image = image.resize((300, 300))
#     image_array = np.array(image)
#     image_array = preprocessing_func(image_array)
#     return np.expand_dims(image_array, axis=0)