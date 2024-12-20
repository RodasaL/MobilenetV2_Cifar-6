# -*- coding: utf-8 -*-
"""
Projeto Fase III - Aprendizagem por Transferência e Deployment
"""

import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_recall_fscore_support
from flask import Flask, request, jsonify
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from scikeras.wrappers import KerasClassifier

# Configurações do modelo e hiperparâmetros
NUM_CLASSES = 6
BATCH_SIZE = 64
EPOCHS = 15
DATA_DIR = r'D:\OneDriveIsec\OneDrive - ISEC\IC\files'

# 1. Funções de pré-processamento e carregamento de dados
def load_cifar_batch(file):
    with open(file, 'rb') as f:
        batch = pickle.load(f, encoding='latin1')
        images = batch['data']
        labels = batch['labels']
        images = images.reshape(-1, 3, 32, 32)
        images = np.transpose(images, (0, 2, 3, 1))
        return images, labels


def load_cifar10(data_dir):
    x_train, y_train = [], []
    for i in range(1, 6):
        batch_file = os.path.join(data_dir, f'data_batch_{i}')
        images, labels = load_cifar_batch(batch_file)
        x_train.append(images)
        y_train += labels
    x_train = np.concatenate(x_train)
    y_train = np.array(y_train)
    test_file = os.path.join(data_dir, 'test_batch')
    x_test, y_test = load_cifar_batch(test_file)
    return x_train, y_train, x_test, y_test


def normalize_data(x):
    return x.astype('float32') / 255.0


def filter_classes(x, y, classes):
    x = np.array(x)
    y = np.array(y)
    mask = np.isin(y, classes)
    return x[mask], y[mask]


def resize_images(images, target_size=(96, 96)):
    return np.array([tf.image.resize(image, target_size).numpy() for image in images])

# 2. Modelo pré-treinado
def build_model(learning_rate=0.001, dense_units=128, dropout_rate=0.3):
    base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(96, 96, 3))
    base_model.trainable = False
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(dense_units, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(dropout_rate),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def random_search_hyperparameters(x_train, y_train):
    model = KerasClassifier(model=build_model, verbose=0)

    param_distributions = {
        "model__learning_rate": [1e-4, 1e-3, 1e-2],
        "model__dense_units": [64, 128, 256,512,1024],
        "model__dropout_rate": [0.2, 0.3, 0.4,0.5,0.6],
        "batch_size": [32, 64,128],
    }

    random_search = RandomizedSearchCV(model, param_distributions, n_iter=3, cv=2, verbose=2, n_jobs=-1)
    random_search.fit(x_train, y_train)
    print(f"Melhores parâmetros encontrados: {random_search.best_params_}")
    return random_search.best_params_

# 3. Treinamento e avaliação
def train_model(model, x_train, y_train, x_val, y_val, epochs, batch_size):
    datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    train_data_generator = datagen.flow(x_train, y_train, batch_size=batch_size)
    return model.fit(train_data_generator, epochs=epochs, validation_data=(x_val, y_val), verbose=1)


def save_results_to_excel(history, params, y_true, y_pred, filename, class_labels):
    results = pd.DataFrame({
        "epoch": list(range(1, len(history.history['loss']) + 1)),
        "train_loss": history.history['loss'],
        "val_loss": history.history['val_loss'],
        "train_accuracy": history.history['accuracy'],
        "val_accuracy": history.history['val_accuracy']
    })

    # Calcular métricas finais
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    auc = roc_auc_score(pd.get_dummies(y_true), pd.get_dummies(y_pred), multi_class='ovr')

    results_test = {
        "class_label": class_labels,
        "precision": precision,
        "recall (sensibilidade)": recall,
        "f1_score": f1,
        "AUC": auc,
        "accuracy": np.mean(y_true == y_pred),
        "learning_rate": [params["learning_rate"]] * len(class_labels),
        "dropout_rate": [params["dropout_rate"]] * len(class_labels),
        "dense_units": [params["dense_units"]] * len(class_labels),
        "batch_size": [params["batch_size"]] * len(class_labels),
    }

    results_test_df = pd.DataFrame(results_test)

    with pd.ExcelWriter(filename) as writer:
        results.to_excel(writer, sheet_name="Train_Validation_Results", index=False)
        results_test_df.to_excel(writer, sheet_name="Test_Results", index=False)

def plot_confusion_matrix(y_true, y_pred, class_labels):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Matriz de Confusão')
    plt.show()

def plot_training_validation_performance(history):
    epochs = range(1, len(history.history['accuracy']) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history.history['accuracy'], label='Training Accuracy', color='blue')
    plt.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    plt.plot(epochs, history.history['loss'], label='Training Loss', linestyle='--', color='blue')
    plt.plot(epochs, history.history['val_loss'], label='Validation Loss', linestyle='--', color='orange')

    plt.title('Training and Validation Performance')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

# 4. Fluxo principal
if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_cifar10(DATA_DIR)
    classes = [2, 3, 4, 5, 6, 7]
    x_train, y_train = filter_classes(x_train, y_train, classes)
    x_test, y_test = filter_classes(x_test, y_test, classes)

    class_mapping = {original: new for new, original in enumerate(classes)}
    y_train = np.array([class_mapping[label] for label in y_train])
    y_test = np.array([class_mapping[label] for label in y_test])
    class_labels = ["passaro", "gato", "veado", "cao", "sapo", "cavalo"]

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train, random_state=42)
    x_train = normalize_data(resize_images(x_train))
    x_val = normalize_data(resize_images(x_val))
    x_test = normalize_data(resize_images(x_test))

    best_params = random_search_hyperparameters(x_train, y_train)
    clean_params = {key.split('__')[-1]: value for key, value in best_params.items()}

    model = build_model(
        learning_rate=clean_params["learning_rate"],
        dense_units=clean_params["dense_units"],
        dropout_rate=clean_params["dropout_rate"]
    )

    history = train_model(model, x_train, y_train, x_val, y_val, epochs=EPOCHS, batch_size=best_params["batch_size"])

    plot_training_validation_performance(history)
    y_pred = np.argmax(model.predict(x_test), axis=1)
    plot_confusion_matrix(y_test, y_pred, class_labels)

    save_results_to_excel(history, clean_params, y_test, y_pred, "results.xlsx", class_labels)
    model.save("animal_classifier_model.h5")
