# -*- coding: utf-8 -*-
"""
Carregar Modelo e Testar API
"""

import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image

# Configurações
MODEL_PATH = "animal_classifier_model.h5"
NUM_CLASSES = 6
TARGET_SIZE = (96, 96)

# Mapear os índices das classes para os seus rótulos reais
cifar_labels = [
     "passaro", "gato", "veado",
    "cao", "sapo", "cavalo"
]

# Inicializar Flask
app = Flask(__name__)

# Carregar o modelo salvo
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Modelo carregado com sucesso de {MODEL_PATH}")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    model = None


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Modelo não carregado"}), 500

    if 'file' not in request.files:
        return jsonify({"error": "Nenhum arquivo foi fornecido"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Nenhum arquivo selecionado"}), 400

    try:
        # Abrir a imagem enviada e pré-processá-la
        image = Image.open(file).resize(TARGET_SIZE)
        image = np.array(image) / 255.0  # Normalizar
        image = image.reshape(1, *TARGET_SIZE, 3)  # Adicionar dimensão do batch

        # Fazer a previsão
        prediction = model.predict(image)
        class_id = np.argmax(prediction)  # Obter o índice da classe com maior probabilidade
        confidence = float(np.max(prediction))  # Confiança da previsão
        class_label = cifar_labels[class_id] if class_id < len(cifar_labels) else "Desconhecida"

        return jsonify({
            "class_id": int(class_id),
            "class_label": class_label,
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=5000, debug=True)
