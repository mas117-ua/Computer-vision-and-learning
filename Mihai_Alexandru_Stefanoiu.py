import os

import pandas as pd

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, LeakyReLU
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns





# ----------------------------------------------
# Función para cargar y preprocesar CIFAR-10
def cargar_y_preprocesar_cifar10():
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.cifar10.load_data()
    X_train = X_train.reshape(X_train.shape[0], -1).astype('float32') / 255
    X_test = X_test.reshape(X_test.shape[0], -1).astype('float32') / 255
    y_train = keras.utils.to_categorical(Y_train, 10).astype('float32')
    y_test = keras.utils.to_categorical(Y_test, 10).astype('float32')
    return X_train, y_train, X_test, y_test


# ----------------------------------------------
# Función para crear el modelo MLP
def crear_MLP(input_shape, ocultas, activ):
    modelo = Sequential()
    modelo.add(Input(shape=input_shape))  # Entrada
    for neuronas, activacion in zip(ocultas, activ):
        modelo.add(Dense(neuronas, activation=activacion))  # Capas ocultas
    modelo.add(Dense(10, activation='softmax'))  # Capa de salida
    modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return modelo


# ----------------------------------------------
# Función para graficar curvas
def plot_curves(history):
    epochs = range(1, len(history.history['accuracy']) + 1)

    # Accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history['accuracy'], label='Train Accuracy')
    plt.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['loss'], label='Train Loss')
    plt.plot(epochs, history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


# ----------------------------------------------
# Tarea A: Entrenar un MLP básico
def tarea_A(X_train, y_train, X_test, y_test):
    print("\nTAREA A: Entrenamiento básico con un MLP")
    modelo = crear_MLP(X_train[0].shape, ocultas=[32], activ=["sigmoid"])
    history = modelo.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)
    plot_curves(history)
    loss, accuracy = modelo.evaluate(X_test, y_test, verbose=1)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy * 100:.2f}%")
    return modelo


# ----------------------------------------------
# Tarea B: Ajustar el número de épocas
def tarea_B(X_train, y_train, X_test, y_test):
    print("\nTAREA B: Ajustar el número de épocas con promedios")
    repeticiones = 5  # Número de entrenamientos independientes
    epochs = 32       # Número de épocas para Tarea B
    histories = []

    # Entrenamientos independientes
    for i in range(repeticiones):
        print(f"Entrenamiento {i + 1}/{repeticiones}...")
        modelo = crear_MLP(X_train[0].shape, ocultas=[32], activ=["sigmoid"])
        history = modelo.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_split=0.1,
            verbose=1
        )
        histories.append(history.history)

    # Promediar los resultados
    def promediar_histories(histories):
        metricas = histories[0].keys()
        promedios = {metrica: [] for metrica in metricas}
        for metrica in metricas:
            valores = np.array([h[metrica] for h in histories])
            promedios[metrica] = np.mean(valores, axis=0)
        return promedios

    promedios = promediar_histories(histories)

    # Graficar resultados promedio
    def plot_curves_promedio(promedios):
        epochs = range(1, len(promedios['accuracy']) + 1)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, promedios['accuracy'], label='Train Accuracy (Promedio)')
        plt.plot(epochs, promedios['val_accuracy'], label='Validation Accuracy (Promedio)')
        plt.title('Accuracy Promedio')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, promedios['loss'], label='Train Loss (Promedio)')
        plt.plot(epochs, promedios['val_loss'], label='Validation Loss (Promedio)')
        plt.title('Loss Promedio')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()

    plot_curves_promedio(promedios)

    # Evaluar un modelo final en el conjunto de test
    modelo_final = crear_MLP(X_train[0].shape, ocultas=[32], activ=["sigmoid"])
    modelo_final.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.1, verbose=0)
    loss, accuracy = modelo_final.evaluate(X_test, y_test, verbose=1)

    print(f"Test Loss (Modelo Final): {loss:.4f}, Test Accuracy: {accuracy * 100:.2f}%")
    return modelo_final

# ----------------------------------------------
# Tarea C: Evaluar el impacto de batch_size
def tarea_C(X_train, y_train, X_test, y_test):
    print("\nTAREA C: Evaluar el impacto de batch_size")

    # Valores de batch_size a probar
    batch_sizes = [8, 16, 32, 64, 128, 256]
    epochs = 5  # Usamos el mismo número de épocas para todas las pruebas
    resultados = []

    # Entrenar y evaluar para cada batch_size
    for batch_size in batch_sizes:
        print(f"Entrenando con batch_size={batch_size}...")
        modelo = crear_MLP(X_train[0].shape, ocultas=[32], activ=["sigmoid"])

        # Medir tiempo de entrenamiento
        inicio = time.time()
        modelo.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=0
        )
        fin = time.time()

        # Evaluar en el conjunto de prueba
        loss, accuracy = modelo.evaluate(X_test, y_test, verbose=0)

        # Guardar resultados
        resultados.append({
            "batch_size": batch_size,
            "test_accuracy": accuracy * 100,
            "training_time": fin - inicio
        })

        print(f"Batch Size: {batch_size}, Training Time: {fin - inicio:.2f}s, Test Accuracy: {accuracy * 100:.2f}%")

    # Datos para los gráficos
    batch_sizes = [r["batch_size"] for r in resultados]
    training_times = [r["training_time"] for r in resultados]
    test_accuracies = [r["test_accuracy"] for r in resultados]

    # Gráfico 1: Batch Size vs Training Time (líneas)
    plt.figure(figsize=(8, 6))
    plt.plot(batch_sizes, training_times, color='r', marker='o', label='Training Time')
    plt.title('Batch Size vs Training Time')
    plt.xlabel('Batch Size')
    plt.ylabel('Training Time (s)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Gráfico 2: Batch Size vs Test Accuracy (barras)
    bar_width = 4  # Grosor de las barras
    plt.figure(figsize=(8, 6))
    plt.bar(batch_sizes, test_accuracies, width=bar_width, color='b', alpha=0.7)
    plt.title('Batch Size vs Test Accuracy')
    plt.xlabel('Batch Size')
    plt.ylabel('Test Accuracy (%)')
    plt.xticks(batch_sizes)  # Ajustar las etiquetas del eje X para que coincidan con los batch sizes
    plt.tight_layout()
    plt.show()


def tarea_D(X_train, y_train, X_test, y_test):
    print("\nTAREA D: Evaluar diferentes funciones de activación")

    # Diccionario para mapear nombres de funciones más legibles
    activation_labels = {
        "sigmoid": "Sigmoid",
        "relu": "ReLU",
        "leaky_relu": "Leaky ReLU",
        "tanh": "Tanh",
        "elu": "ELU",
        "selu": "SELU"
    }

    activations = list(activation_labels.keys())
    epochs = 20
    batch_size = 128
    repeticiones = 5
    resultados = []

    # Entrenar y evaluar para cada función de activación
    for activacion in activations:
        print(f"\nEntrenando con activación: {activacion}")

        # Almacenar resultados de múltiples ejecuciones
        iteracion_resultados = {
            "test_accuracies": [],
            "val_accuracies": [],
            "training_times": []
        }

        for rep in range(repeticiones):
            print(f"  Repetición {rep + 1}/{repeticiones}")

            modelo = Sequential()
            modelo.add(Input(shape=(3072,)))  # Entrada

            if activacion == "leaky_relu":
                modelo.add(Dense(32))
                modelo.add(LeakyReLU(negative_slope=0.1))
            elif activacion == "elu":
                modelo.add(Dense(32, activation='elu'))
            elif activacion == "selu":
                modelo.add(Dense(32, activation='selu'))
            else:
                modelo.add(Dense(32, activation=activacion))

            modelo.add(Dense(10, activation='softmax'))  # Capa de salida

            modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            inicio = time.time()
            history = modelo.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.1,
                verbose=0
            )
            fin = time.time()

            loss, accuracy = modelo.evaluate(X_test, y_test, verbose=0)

            iteracion_resultados["test_accuracies"].append(accuracy * 100)
            iteracion_resultados["val_accuracies"].append(max(history.history['val_accuracy']) * 100)
            iteracion_resultados["training_times"].append(fin - inicio)

        # Calcular promedios para esta función de activación
        resultados.append({
            "activation": activacion,
            "test_accuracy": np.mean(iteracion_resultados["test_accuracies"]),
            "test_accuracy_std": np.std(iteracion_resultados["test_accuracies"]),
            "val_accuracy": np.mean(iteracion_resultados["val_accuracies"]),
            "val_accuracy_std": np.std(iteracion_resultados["val_accuracies"]),
            "training_time": np.mean(iteracion_resultados["training_times"]),
            "training_time_std": np.std(iteracion_resultados["training_times"])
        })

    # Preparar datos con etiquetas legibles
    activation_labels_list = [activation_labels[act] for act in activations]
    training_times = [r["training_time"] for r in resultados]
    training_times_std = [r["training_time_std"] for r in resultados]
    test_accuracies = [r["test_accuracy"] for r in resultados]
    test_accuracies_std = [r["test_accuracy_std"] for r in resultados]
    val_accuracies = [r["val_accuracy"] for r in resultados]
    val_accuracies_std = [r["val_accuracy_std"] for r in resultados]

    plt.figure(figsize=(15, 5))

    # Gráfico 1: Tiempo de entrenamiento
    plt.subplot(1, 3, 1)
    bars1 = plt.bar(activation_labels_list, training_times, color='r', alpha=0.7, yerr=training_times_std, capsize=5)
    plt.title('Tiempo de Entrenamiento Promedio')
    plt.xlabel('Función de Activación')
    plt.ylabel('Tiempo (s)')
    plt.xticks(rotation=45, ha='right')
    for bar in bars1:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{bar.get_height():.2f}',
                 ha='center', va='bottom')

    # Gráfico 2: Test Accuracy
    plt.subplot(1, 3, 2)
    bars2 = plt.bar(activation_labels_list, test_accuracies, color='b', alpha=0.7, yerr=test_accuracies_std, capsize=5)
    plt.title('Accuracy de Test Promedio')
    plt.xlabel('Función de Activación')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45, ha='right')
    for bar in bars2:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{bar.get_height():.2f}',
                 ha='center', va='bottom')

    # Gráfico 3: Validation Accuracy
    plt.subplot(1, 3, 3)
    bars3 = plt.bar(activation_labels_list, val_accuracies, color='g', alpha=0.7, yerr=val_accuracies_std, capsize=5)
    plt.title('Accuracy de Validación Promedio')
    plt.xlabel('Función de Activación')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45, ha='right')
    for bar in bars3:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{bar.get_height():.2f}',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    print("\nResumen de Resultados:")
    for resultado in resultados:
        act_label = activation_labels[resultado['activation']]
        print(f"{act_label}: "
              f"Test Accuracy = {resultado['test_accuracy']:.2f}% ± {resultado['test_accuracy_std']:.2f}%, "
              f"Validation Accuracy = {resultado['val_accuracy']:.2f}% ± {resultado['val_accuracy_std']:.2f}%, "
              f"Training Time = {resultado['training_time']:.2f}s ± {resultado['training_time_std']:.2f}s")
def tarea_E(X_train, y_train, X_test, y_test):
    print("\nTAREA E: Ajustar el número de neuronas")

    # Funciones de activación a probar
    activations = ['elu']

    # Número de neuronas a probar
    neuron_counts = [16, 32, 64]
    epochs = 20
    batch_size = 128
    repeticiones = 5
    resultados = []

    # Entrenar y evaluar para cada función de activación y número de neuronas
    for activacion in activations:
        for neuronas in neuron_counts:
            print(f"\nEntrenando con {activacion}, {neuronas} neuronas")

            # Almacenar resultados de múltiples ejecuciones
            iteracion_resultados = {
                "test_accuracies": [],
                "val_accuracies": [],
                "training_times": []
            }

            for rep in range(repeticiones):
                print(f"  Repetición {rep + 1}/{repeticiones}")

                modelo = Sequential()
                modelo.add(Input(shape=(3072,)))  # Entrada

                # Manejar diferentes funciones de activación
                if activacion == "leaky_relu":
                    modelo.add(Dense(neuronas))
                    modelo.add(LeakyReLU(negative_slope=0.1))
                else:
                    modelo.add(Dense(neuronas, activation=activacion))

                modelo.add(Dense(10, activation='softmax'))  # Capa de salida

                modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

                inicio = time.time()
                history = modelo.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.1,
                    verbose=0
                )
                fin = time.time()

                loss, accuracy = modelo.evaluate(X_test, y_test, verbose=0)

                iteracion_resultados["test_accuracies"].append(accuracy * 100)
                iteracion_resultados["val_accuracies"].append(max(history.history['val_accuracy']) * 100)
                iteracion_resultados["training_times"].append(fin - inicio)

            # Calcular promedios para esta configuración
            resultados.append({
                "activation": activacion,
                "neurons": neuronas,
                "test_accuracy": np.mean(iteracion_resultados["test_accuracies"]),
                "test_accuracy_std": np.std(iteracion_resultados["test_accuracies"]),
                "val_accuracy": np.mean(iteracion_resultados["val_accuracies"]),
                "val_accuracy_std": np.std(iteracion_resultados["val_accuracies"]),
                "training_time": np.mean(iteracion_resultados["training_times"]),
                "training_time_std": np.std(iteracion_resultados["training_times"])
            })

    # Encontrar los mejores modelos por función de activación
    mejores_modelos = {}
    for activacion in activations:
        modelos_activacion = [r for r in resultados if r['activation'] == activacion]
        mejor_modelo = max(modelos_activacion, key=lambda x: x['test_accuracy'])
        mejores_modelos[activacion] = mejor_modelo

    # Imprimir resumen de mejores modelos
    print("\nMejores modelos por función de activación:")
    for activacion, modelo in mejores_modelos.items():
        print(f"{activacion.upper()}: {modelo['neurons']} neuronas, "
              f"Test Accuracy = {modelo['test_accuracy']:.2f}% ± {modelo['test_accuracy_std']:.2f}%")

    return resultados
def tarea_F(X_train, y_train, X_test, y_test):
    print("\nTAREA F: Optimizar un MLP de múltiples capas")

    # Basado en los resultados previos: ELU con 64 neuronas
    activations = ['elu']
    neuron_configs = [
        [64, 32],     # Dos capas con reducción gradual
        [64, 64, 32], # Tres capas
        [128, 64, 32] # Más neuronas en primera capa
    ]
    epochs = 20
    batch_size = 128
    repeticiones = 5
    resultados = []

    # Entrenar y evaluar para cada configuración de capas
    for activacion in activations:
        for capas in neuron_configs:
            print(f"\nEntrenando con {activacion}, capas: {capas}")

            # Almacenar resultados de múltiples ejecuciones
            iteracion_resultados = {
                "test_accuracies": [],
                "val_accuracies": [],
                "training_times": []
            }

            for rep in range(repeticiones):
                print(f"  Repetición {rep + 1}/{repeticiones}")

                modelo = Sequential()
                modelo.add(Input(shape=(3072,)))  # Entrada

                # Añadir capas dinámicamente
                for neuronas in capas[:-1]:
                    if activacion == "leaky_relu":
                        modelo.add(Dense(neuronas))
                        modelo.add(LeakyReLU(negative_slope=0.1))
                    else:
                        modelo.add(Dense(neuronas, activation=activacion))

                # Última capa puede ser diferente
                if activacion == "leaky_relu":
                    modelo.add(Dense(capas[-1]))
                    modelo.add(LeakyReLU(negative_slope=0.1))
                else:
                    modelo.add(Dense(capas[-1], activation=activacion))

                modelo.add(Dense(10, activation='softmax'))  # Capa de salida

                modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

                inicio = time.time()
                history = modelo.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.1,
                    verbose=0
                )
                fin = time.time()

                loss, accuracy = modelo.evaluate(X_test, y_test, verbose=0)

                iteracion_resultados["test_accuracies"].append(accuracy * 100)
                iteracion_resultados["val_accuracies"].append(max(history.history['val_accuracy']) * 100)
                iteracion_resultados["training_times"].append(fin - inicio)

            # Calcular promedios para esta configuración
            resultados.append({
                "activation": activacion,
                "layers": capas,
                "test_accuracy": np.mean(iteracion_resultados["test_accuracies"]),
                "test_accuracy_std": np.std(iteracion_resultados["test_accuracies"]),
                "val_accuracy": np.mean(iteracion_resultados["val_accuracies"]),
                "val_accuracy_std": np.std(iteracion_resultados["val_accuracies"]),
                "training_time": np.mean(iteracion_resultados["training_times"]),
                "training_time_std": np.std(iteracion_resultados["training_times"])
            })

    # Imprimir resumen de resultados
    print("\nResumen de Resultados:")
    for resultado in resultados:
        print(f"Capas {resultado['layers']}: "
              f"Test Accuracy = {resultado['test_accuracy']:.2f}% ± {resultado['test_accuracy_std']:.2f}%, "
              f"Validation Accuracy = {resultado['val_accuracy']:.2f}% ± {resultado['val_accuracy_std']:.2f}%, "
              f"Training Time = {resultado['training_time']:.2f}s ± {resultado['training_time_std']:.2f}s")

    return resultados
#---------------------------------------------------------------------------------------------------------------------


def cargar_y_preprocesar_cifar10_cnn():
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.cifar10.load_data()

    # Normalizar las imágenes sin aplanarlas (mantener la forma 32x32x3)
    X_train = X_train.astype('float32') / 255  # Imágenes: 32x32x3
    X_test = X_test.astype('float32') / 255    # Imágenes: 32x32x3

    # Convertir las etiquetas a one-hot encoding
    y_train = keras.utils.to_categorical(Y_train, 10)
    y_test = keras.utils.to_categorical(Y_test, 10)

    return X_train, y_train, X_test, y_test

def crear_CNN(input_shape, con_maxpool=True):
    modelo = Sequential()

    # Definir la forma de entrada utilizando la capa Input
    modelo.add(Input(shape=input_shape))  # Forma de entrada: (32, 32, 3)

    # Capa Conv2D (16 filtros, 3x3)
    modelo.add(Conv2D(16, (3, 3), activation='relu'))

    if con_maxpool:
        # MaxPooling2D después de la primera capa Conv2D
        modelo.add(MaxPooling2D(pool_size=(2, 2)))

    # Capa Conv2D (32 filtros, 3x3)
    modelo.add(Conv2D(32, (3, 3), activation='relu'))

    if con_maxpool:
        # MaxPooling2D después de la segunda capa Conv2D
        modelo.add(MaxPooling2D(pool_size=(2, 2)))

    # Aplanar para la capa densa
    modelo.add(Flatten())

    # Capa de salida (softmax)
    modelo.add(Dense(10, activation='softmax'))

    # Compilación del modelo
    modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return modelo
# Tarea G: Definir, entrenar y evaluar un CNN sencillo
def tarea_G(X_train, y_train, X_test, y_test):
    print("\nTAREA G: Definir, entrenar y evaluar un CNN sencillo")

    # Preprocesamiento de los datos para CNN (sin aplanar las imágenes)
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    input_shape = X_train.shape[1:]  # (32, 32, 3)

    # Entrenar el modelo con MaxPooling2D
    print("\nEntrenando CNN con MaxPooling2D...")
    modelo_maxpool = crear_CNN(input_shape, con_maxpool=True)
    history_maxpool = modelo_maxpool.fit(
        X_train, y_train,
        epochs=50,
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )

    # Entrenar el modelo sin MaxPooling2D
    print("\nEntrenando CNN sin MaxPooling2D...")
    modelo_sin_maxpool = crear_CNN(input_shape, con_maxpool=False)
    history_sin_maxpool = modelo_sin_maxpool.fit(
        X_train, y_train,
        epochs=50   ,
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )

    # Evaluar ambos modelos
    print("\nEvaluando CNN con MaxPooling2D...")
    loss_maxpool, accuracy_maxpool = modelo_maxpool.evaluate(X_test, y_test, verbose=1)
    print(f"Test Loss (MaxPooling2D): {loss_maxpool:.4f}, Test Accuracy: {accuracy_maxpool * 100:.2f}%")

    print("\nEvaluando CNN sin MaxPooling2D...")
    loss_sin_maxpool, accuracy_sin_maxpool = modelo_sin_maxpool.evaluate(X_test, y_test, verbose=1)
    print(f"Test Loss (sin MaxPooling2D): {loss_sin_maxpool:.4f}, Test Accuracy: {accuracy_sin_maxpool * 100:.2f}%")

    # Graficar las curvas de ambos modelos
    print("\nGraficando resultados de CNN con MaxPooling2D...")
    plot_curves(history_maxpool)

    print("\nGraficando resultados de CNN sin MaxPooling2D...")
    plot_curves(history_sin_maxpool)



def crear_CNN_con_kernel(input_shape, kernel_size=(3, 3)):
    modelo = Sequential()

    # Usar Input() para especificar el tamaño de la entrada
    modelo.add(Input(shape=input_shape))  # Primero definimos la entrada

    # Capa Conv2D con el tamaño de filtro ajustado
    modelo.add(Conv2D(16, kernel_size=kernel_size, activation='relu'))
    modelo.add(MaxPooling2D(pool_size=(2, 2)))

    # Segunda capa Conv2D
    modelo.add(Conv2D(32, kernel_size=kernel_size, activation='relu'))
    modelo.add(MaxPooling2D(pool_size=(2, 2)))

    # Capa densa final
    modelo.add(Flatten())
    modelo.add(Dense(10, activation='softmax'))

    modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return modelo

def tarea_H(X_train, y_train, X_test, y_test):
    print("\nTAREA H: Ajustar el parámetro kernel_size")

    # Diferentes tamaños de filtros a probar
    kernel_sizes = [(3, 3), (5, 5), (7, 7)]
    epochs = 20
    batch_size = 128
    resultados = []

    for kernel_size in kernel_sizes:
        print(f"\nEntrenando con kernel_size = {kernel_size}...")

        # Crear el modelo con el tamaño de filtro específico
        modelo = crear_CNN_con_kernel(X_train[0].shape, kernel_size)

        # Medir tiempo de entrenamiento
        inicio = time.time()
        history = modelo.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=0
        )
        fin = time.time()

        # Evaluar en el conjunto de prueba
        loss, accuracy = modelo.evaluate(X_test, y_test, verbose=0)

        # Guardar resultados
        resultados.append({
            "kernel_size": kernel_size,
            "test_accuracy": accuracy * 100,
            "training_time": fin - inicio
        })

        print(f"kernel_size = {kernel_size}, Training Time: {fin - inicio:.2f}s, Test Accuracy: {accuracy * 100:.2f}%")

    # Datos para los gráficos
    kernel_sizes = [r["kernel_size"] for r in resultados]
    training_times = [r["training_time"] for r in resultados]
    test_accuracies = [r["test_accuracy"] for r in resultados]

    # Gráfico 1: Kernel Size vs Training Time (líneas)
    plt.figure(figsize=(8, 6))
    plt.plot([str(k) for k in kernel_sizes], training_times, color='r', marker='o', label='Training Time')
    plt.title('Kernel Size vs Training Time')
    plt.xlabel('Kernel Size')
    plt.ylabel('Training Time (s)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Gráfico 2: Kernel Size vs Test Accuracy (barras)
    bar_width = 0.4  # Grosor de las barras
    plt.figure(figsize=(8, 6))
    plt.bar([str(k) for k in kernel_sizes], test_accuracies, width=bar_width, color='b', alpha=0.7)
    plt.title('Kernel Size vs Test Accuracy')
    plt.xlabel('Kernel Size')
    plt.ylabel('Test Accuracy (%)')
    plt.tight_layout()
    plt.show()

    # Resultados finales
    print("\nResumen de Resultados:")
    for resultado in resultados:
        print(f"kernel_size {resultado['kernel_size']}: "
              f"Test Accuracy = {resultado['test_accuracy']:.2f}%, "
              f"Training Time = {resultado['training_time']:.2f}s")

    # Seleccionar el mejor kernel_size basado en la precisión y el tiempo de entrenamiento
    mejor_modelo = max(resultados, key=lambda x: x['test_accuracy'])
    print(f"\nMejor kernel_size: {mejor_modelo['kernel_size']} con una precisión de {mejor_modelo['test_accuracy']:.2f}%")


def crear_CNN_optimal(input_shape, capas_conv=[16, 32, 64], capas_dense=[64]):
    modelo = Sequential()

    # Capa Conv2D inicial
    modelo.add(Input(shape=input_shape))  # Especificamos el tamaño de entrada
    for filtros in capas_conv:
        modelo.add(Conv2D(filtros, kernel_size=(3, 3), activation='relu'))
        modelo.add(MaxPooling2D(pool_size=(2, 2)))

    # Capa densa final
    modelo.add(Flatten())
    for neuronas in capas_dense:
        modelo.add(Dense(neuronas, activation='relu'))

    modelo.add(Dense(10, activation='softmax'))  # Capa de salida

    modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return modelo


def tarea_I(X_train, y_train, X_test, y_test):
    print("\nTAREA I: Optimizar la arquitectura del modelo")

    # Configuraciones de capas y filtros a probar
    configuraciones = [
        ([16, 32], [64]),  # 2 capas convolucionales y 1 capa densa
        ([32, 64, 128], [64]),  # 3 capas convolucionales y 1 capa densa
        ([16, 32, 64], [128, 64])  # 3 capas convolucionales y 2 capas densas
    ]

    epochs = 50
    batch_size = 128
    resultados = []

    for capas_conv, capas_dense in configuraciones:
        print(f"\nEntrenando modelo con Conv2D {capas_conv} y Dense {capas_dense}...")

        # Crear el modelo con la configuración actual
        modelo = crear_CNN_optimal(X_train[0].shape, capas_conv, capas_dense)

        # Medir tiempo de entrenamiento
        inicio = time.time()
        history = modelo.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1
        )
        fin = time.time()

        # Evaluar en el conjunto de prueba
        loss, accuracy = modelo.evaluate(X_test, y_test, verbose=1)

        # Guardar resultados
        resultados.append({
            "capas_conv": capas_conv,
            "capas_dense": capas_dense,
            "test_accuracy": accuracy * 100,
            "training_time": fin - inicio
        })

        print(f"Configuración Conv2D {capas_conv} y Dense {capas_dense}: "
              f"Test Accuracy = {accuracy * 100:.2f}%, "
              f"Training Time = {fin - inicio:.2f}s")

    # Datos para los gráficos
    configuraciones_str = [f"Conv2D {r['capas_conv']} / Dense {r['capas_dense']}" for r in resultados]
    training_times = [r["training_time"] for r in resultados]
    test_accuracies = [r["test_accuracy"] for r in resultados]

    # Gráfico 1: Arquitectura vs Time de Entrenamiento
    plt.figure(figsize=(10, 6))
    plt.bar(configuraciones_str, training_times, color='r', alpha=0.7)
    plt.title('Arquitectura vs Time de Entrenamiento')
    plt.xlabel('Arquitectura')
    plt.ylabel('Training Time (s)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Gráfico 2: Arquitectura vs Test Accuracy
    plt.figure(figsize=(10, 6))
    plt.bar(configuraciones_str, test_accuracies, color='b', alpha=0.7)
    plt.title('Arquitectura vs Test Accuracy')
    plt.xlabel('Arquitectura')
    plt.ylabel('Test Accuracy (%)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Resultados finales
    print("\nResumen de Resultados:")
    for resultado in resultados:
        print(f"Conv2D {resultado['capas_conv']} y Dense {resultado['capas_dense']}: "
              f"Test Accuracy = {resultado['test_accuracy']:.2f}%, "
              f"Training Time = {resultado['training_time']:.2f}s")

    # Seleccionar el mejor modelo basado en la precisión y el tiempo de entrenamiento
    mejor_modelo = max(resultados, key=lambda x: x['test_accuracy'])
    print(f"\nMejor arquitectura: Conv2D {mejor_modelo['capas_conv']} y Dense {mejor_modelo['capas_dense']} "
          f"con una precisión de {mejor_modelo['test_accuracy']:.2f}%")



def tarea_J(base_dir="./imagenes"):

    print("\nTAREA J: Creación de conjunto de prueba y evaluación de la generalización")

    # Mapeo de etiquetas de CIFAR-10
    CATEGORIES = {
        "avión": 0, "automóvil": 1, "pájaro": 2, "gato": 3,
        "ciervo": 4, "perro": 5, "rana": 6, "caballo": 7,
        "barco": 8, "camión": 9
    }

    # Dimensiones de las imágenes
    IMG_HEIGHT, IMG_WIDTH = 32, 32

    # Listas para almacenar imágenes y etiquetas
    imagenes = []
    etiquetas = []

    # Recorrer las categorías y procesar las imágenes
    for categoria, etiqueta in CATEGORIES.items():
        path = os.path.join(base_dir, categoria)
        print(f"Procesando categoría: {categoria} (Etiqueta: {etiqueta})")
        count = 0  # Contador para limitar a las primeras 15 imágenes
        for archivo in os.listdir(path):
            if archivo.endswith(('.png', '.jpg', '.jpeg')):  # Validar extensiones
                if count >= 15:  # Limitar a 15 imágenes
                    break
                img_path = os.path.join(path, archivo)
                try:
                    # Abrir, redimensionar y convertir a RGB
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
                    imagenes.append(np.array(img))
                    etiquetas.append(etiqueta)
                    count += 1
                except Exception as e:
                    print(f"Error procesando {img_path}: {e}")

    # Verificar contenido antes de continuar
    if not etiquetas:
        raise ValueError("No se encontraron etiquetas. Verifica las carpetas y los archivos.")

    # Convertir listas a arreglos NumPy
    imagenes = np.array(imagenes, dtype='float32') / 255.0  # Normalizar
    etiquetas = np.array(etiquetas)

    # Convertir etiquetas a one-hot encoding
    etiquetas = keras.utils.to_categorical(etiquetas, num_classes=10)

    # Verificar el conjunto de datos
    print(f"Imágenes procesadas: {imagenes.shape}")
    print(f"Etiquetas procesadas: {etiquetas.shape}")

    return imagenes, etiquetas


def tarea_K(X_train, y_train, X_test, y_test):

    print("\nTAREA K: Evaluando las 3 mejores configuraciones de CNN")

    # Load custom dataset
    custom_images, custom_labels = tarea_J("./imagenes")

    # Class names for visualization
    class_names = ['avión', 'automóvil', 'pájaro', 'gato', 'ciervo',
                   'perro', 'rana', 'caballo', 'barco', 'camión']

    # Top 3 configuraciones esperadas
    configurations = [
        {
            'name': 'CNN1 - Medium con Adam',
            'filters': [32, 64, 128],
            'activation': 'relu',
            'optimizer': ('adam', 0.001),
            'kernel_size': (3, 3)
        },
        {
            'name': 'CNN2 - Medium con ELU',
            'filters': [32, 64, 128],
            'activation': 'elu',
            'optimizer': ('adam', 0.001),
            'kernel_size': (3, 3)
        },
        {
            'name': 'CNN3 - Deep con ReLU',
            'filters': [32, 64, 128, 256],
            'activation': 'relu',
            'optimizer': ('adam', 0.001),
            'kernel_size': (3, 3)
        }
    ]

    batch_size = 128
    epochs = 10
    results = []

    def create_model(filters, activation, kernel_size=(3, 3)):
        model = Sequential()
        model.add(Input(shape=(32, 32, 3)))

        for f in filters:
            model.add(Conv2D(f, kernel_size, activation=activation, padding='same'))
            model.add(MaxPooling2D())

        model.add(Flatten())
        model.add(Dense(128, activation=activation))
        model.add(Dense(10, activation='softmax'))
        return model

    # Test configurations
    for config in configurations:
        print(f"\nProbando configuración: {config['name']}")

        model = create_model(config['filters'], config['activation'], config['kernel_size'])
        opt_name, lr = config['optimizer']
        optimizer = eval(f"tf.keras.optimizers.{opt_name.capitalize()}(learning_rate={lr})")
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        start_time = time.time()
        history = model.fit(X_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_split=0.2,
                            verbose=1)
        training_time = time.time() - start_time

        loss, accuracy = model.evaluate(custom_images, custom_labels, verbose=1)
        y_pred = model.predict(custom_images, verbose=1)
        cm = confusion_matrix(np.argmax(custom_labels, axis=1),
                              np.argmax(y_pred, axis=1))

        results.append({
            'name': config['name'],
            'accuracy': accuracy * 100,
            'loss': loss,
            'training_time': training_time,
            'confusion_matrix': cm,
            'val_accuracy': max(history.history['val_accuracy'])
        })

    # Plot comparison
    plt.figure(figsize=(10, 6))
    accuracies = [r['accuracy'] for r in results]
    names = [r['name'] for r in results]
    plt.bar(names, accuracies)
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Plot confusion matrices
    for result in results:
        plt.figure(figsize=(10, 8))
        sns.heatmap(result['confusion_matrix'],
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title(f'Confusion Matrix - {result["name"]}\nAccuracy: {result["accuracy"]:.2f}%')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # Print summary
    print("\nResumen de Resultados:")
    print("=" * 80)
    for result in results:
        print(f"\nModelo: {result['name']}")
        print(f"Accuracy en dataset personalizado: {result['accuracy']:.2f}%")
        print(f"CIFAR-10 Validation Accuracy: {result['val_accuracy'] * 100:.2f}%")
        print(f"Tiempo de entrenamiento: {result['training_time']:.2f}s")

    best_result = max(results, key=lambda x: x['accuracy'])
    print("\nMejor Configuración:")
    print(f"Modelo: {best_result['name']}")
    print(f"Accuracy: {best_result['accuracy']:.2f}%")
    print(f"Training Time: {best_result['training_time']:.2f}s")

    return results

from tensorflow.keras.layers import (
    Dropout,
    BatchNormalization,
    Activation,
    GlobalAveragePooling2D,
    RandomFlip,
    RandomRotation,
    RandomTranslation,
    Rescaling
)
from tensorflow.keras.regularizers import l2


def tarea_L(X_train, y_train, X_test, y_test):

    print("\nTAREA L: Entrenando CNN mejorada (versión básica)")

    # Load custom dataset
    custom_images, custom_labels = tarea_J("./imagenes")

    # Class names for visualization
    class_names = ['avión', 'automóvil', 'pájaro', 'gato', 'ciervo',
                   'perro', 'rana', 'caballo', 'barco', 'camión']

    def create_improved_cnn():
        model = Sequential([
            # Input Layer
            Input(shape=(32, 32, 3)),

            # First Convolutional Block
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            MaxPooling2D(),

            # Second Convolutional Block
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D(),

            # Flatten + Dense Layers
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),  # Only one dropout layer before final classification
            Dense(10, activation='softmax')
        ])
        return model

    # Create and compile model
    model = create_improved_cnn()

    # Using standard Adam settings
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Print model summary
    print("\nModel Architecture:")
    model.summary()

    # Train model
    print("\nTraining model...")
    epochs = 50
    batch_size = 128

    start_time = time.time()
    history = model.fit(
        X_train / 255.0,  # Normalize data here
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        verbose=1
    )
    training_time = time.time() - start_time

    # Evaluate on custom dataset
    print("\nEvaluating on custom dataset...")
    loss, accuracy = model.evaluate(custom_images / 255.0, custom_labels, verbose=1)
    y_pred = model.predict(custom_images / 255.0, verbose=1)

    # Calculate confusion matrix
    cm = confusion_matrix(np.argmax(custom_labels, axis=1),
                          np.argmax(y_pred, axis=1))

    # Plot training curves
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix\nAccuracy: {:.2f}%'.format(accuracy * 100))
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Print results summary
    print("\nResults Summary:")
    print("=" * 80)
    print(f"Custom Dataset Accuracy: {accuracy * 100:.2f}%")
    print(f"Custom Dataset Loss: {loss:.4f}")
    print(f"Training Time: {training_time:.2f}s")
    print(f"Best Validation Accuracy: {max(history.history['val_accuracy']) * 100:.2f}%")

    return model, history

if __name__ == "__main__":
    # Cargar y preprocesar datos
    X_train, y_train, X_test, y_test = cargar_y_preprocesar_cifar10()
    #
    # # Ejecutar las tareas
    # tarea_A(X_train, y_train, X_test, y_test)  # Tarea A
    # tarea_B(X_train, y_train, X_test, y_test)  # Tarea B
    # tarea_C(X_train, y_train, X_test, y_test)  # Tarea C
    # tarea_D(X_train, y_train, X_test, y_test)
    # tarea_E(X_train, y_train, X_test, y_test)
    # tarea_F(X_train, y_train, X_test, y_test)
    #
    X_train, y_train, X_test, y_test = cargar_y_preprocesar_cifar10_cnn()
    #
    # tarea_G(X_train, y_train, X_test, y_test)
    # tarea_H(X_train, y_train, X_test, y_test)
    # tarea_I(X_train, y_train, X_test, y_test)

    #tarea_K(X_train, y_train, X_test, y_test)
    tarea_L(X_train, y_train, X_test, y_test)
