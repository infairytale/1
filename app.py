from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np


tf.config.set_visible_devices([], 'GPU')

app = Flask(__name__)

# Загрузка модели Keras
model = load_model("best_model.h5")

# Функция для предсказания класса рентгенограммы
def predict_xray(img):
    # Изменение размера изображения до 299x299 (размер, ожидаемый моделью)
    img = img.resize((299, 299))
    
    # Преобразование изображения в массив NumPy
    img = np.array(img)
    
    # Убедитесь, что изображение имеет 3 канала (RGB)
    if img.shape[-1] != 3:
        img = np.stack((img,) * 3, axis=-1)
    
    # Нормализация данных
    img = img.reshape(1, 299, 299, 3).astype('float32') / 255.0

    # Предсказание класса рентгенограммы
    prediction = model.predict(img)
    
    # Возвращаем индекс класса с наибольшей вероятностью
    return np.argmax(prediction)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Получение изображения из запроса
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file uploaded"}), 400
        
        img = Image.open(file.stream)
        class_names = ['COVID19', 'NORMAL', 'PNEUMONIA', 'PNEUMOTRACS', 'TURBERCULOSIS']
        # Предсказание класса рентгенограммы
        xray_class = class_names[predict_xray(img)]

        
        return jsonify({"class": xray_class})
    
    return render_template("index1.html")

if __name__ == "__main__":
    app.run(debug=True)


