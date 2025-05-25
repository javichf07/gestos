import cv2
import os
from tqdm import tqdm # Para visualización de progreso
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def extract_frames(video_path, output_folder, frame_interval=1):
    """
    Extrae fotogramas de un video y los guarda en una carpeta.

    Args:
        video_path (str): Ruta completa al archivo de video.
        output_folder (str): Carpeta donde se guardarán los fotogramas.
        frame_interval (int): Cada cuántos fotogramas se extrae uno.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: No se pudo abrir el video {video_path}")
        return False

    frame_count = 0
    success, image = video.read()

    # Obtener el número total de fotogramas para la barra de progreso
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Usar tqdm para una barra de progreso visual
    with tqdm(total=total_frames // frame_interval, desc=f"Extrayendo frames de {os.path.bsename(video_path)}") as pbar:
        while success:
            if frame_count % frame_interval == 0:
                frame_filename = f"frame_{frame_count:06d}.jpg" # Formato para ordenar fácilmente
                cv2.imwrite(os.path.join(output_folder, frame_filename), image)
                pbar.update(1) # Actualiza la barraa de progreso

            frame_count += 1
            success, image = video.read()

    video.release()
    return True

# Ejemplo de uso (ejecutado en un script principal)
# video_file_path = "data/videos/uno.mp4"
# output_frames_dir = "data/frames/uno" # Se creará una subcarpeta para cada video
# extract_frames(video_file_path, output_frames_dir, frame_interval=2)



def normalize_image(image, target_size=(224, 224)):
    """
    Redimensiona una imagen y escala sus valores de píxeles a [0, 1].

    Args:
        image (numpy.ndarray): La imagen de entrada (OpenCV lee en BGR).
        target_size (tuple): La resolución deseada (ancho, alto).

    Returns:
        numpy.ndarray: La imagen redimensionada y normalizada.
    """
    # Redimensionar la imagen
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    # Escalar los píxeles a [0, 1]
    normalized_image = resized_image.astype(np.float32) / 255.0

    return normalized_image

# Ejemplo de uso
# loaded_image = cv2.imread("data/frames/uno/frame_000000.jpg")
# normalized_img = normalize_image(loaded_image, target_size=(224, 224))

def segment_hand_hsv(image):
    """
    Segmenta la mano de la imagen usando un rango de colores HSV para la piel.

    Args:
        image (numpy.ndarray): La imagen BGR de entrada.

    Returns:
        numpy.ndarray: La imagen con la mano segmentada (fondo negro).
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Rangos de color para la piel en HSV (ejemplo, ajustar según tus datos)
    # H (Hue): Tono (0-179 en OpenCV)
    # S (Saturation): Saturación (0-255)
    # V (Value): Valor/Brillo (0-255)
    lower_skin = np.array([0, 40, 80], dtype=np.uint8)  # Ajustar!
    upper_skin = np.array([25, 255, 255], dtype=np.uint8) # Ajustar!

    # Crear una máscara de la piel
    skin_mask = cv2.inRange(hsv_image, lower_skin, upper_skin)

    # Opcional: Aplicar operaciones morfológicas para limpiar la máscara
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel) # Eliminar pequeños ruidos
    # skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel) # Cerrar pequeños huecos

    # Aplicar la máscara a la imagen original
    segmented_image = cv2.bitwise_and(image, image, mask=skin_mask)

    return segmented_image

# Ejemplo de uso
# loaded_image = cv2.imread("data/frames/uno/frame_000000.jpg")
# segmented_img = segment_hand_hsv(loaded_image)
# cv2.imwrite("data/preprocessed/uno/segmented_frame_000000.jpg", segmented_img) # Guardar para revisar



def create_annotations_file(preprocessed_base_folder, output_csv_path="data/annotations/annotations.csv"):
    """
    Crea un archivo CSV de anotaciones a partir de los fotogramas preprocesados.
    Asume que las subcarpetas dentro de preprocessed_base_folder son las etiquetas.

    Args:
        preprocessed_base_folder (str): Ruta a la carpeta base de fotogramas preprocesados.
        output_csv_path (str): Ruta donde se guardará el archivo CSV de anotaciones.
    """
    data = []
    # Itera sobre cada subcarpeta (que representa una seña)
    for sign_label in tqdm(os.listdir(preprocessed_base_folder), desc="Creando archivo de anotaciones"):
        sign_folder_path = os.path.join(preprocessed_base_folder, sign_label)
        if os.path.isdir(sign_folder_path): # Asegurarse de que es una carpeta
            for frame_filename in os.listdir(sign_folder_path):
                if frame_filename.endswith(('.jpg', '.png')):
                    # Construir la ruta relativa al fotograma
                    frame_relative_path = os.path.join(sign_label, frame_filename)
                    data.append({'image_path': frame_relative_path, 'label': sign_label})

    # Crear un DataFrame de pandas y guardarlo como CSV
    df = pd.DataFrame(data)
    df.to_csv(output_csv_path, index=False)
    print(f"Archivo de anotaciones creado en: {output_csv_path}")
    return df

# Ejemplo de uso (dentro del script principal)
# preprocessed_base_dir = "data/preprocessed"
# annotations_df = create_annotations_file(preprocessed_base_dir)
# print(annotations_df.head())


import shutil # Para copiar archivos

# Función para visualizar la distribución (para análisis)
def analyze_dataset_distribution(annotations_df):
    print("\nDistribución inicial de muestras por seña:")
    print(annotations_df['label'].value_counts())
    print("-" * 40)

# Esta función debería ser parte del pipeline de entrenamiento,
# ya que el aumento de datos generalmente se aplica al vuelo durante el entrenamiento
# o para generar nuevas imágenes y luego re-crear el dataset.
# Aquí un ejemplo conceptual de cómo podrías generar más datos para una clase específica.
def apply_data_augmentation_for_balancing(image_path, target_folder, num_augmentations_per_image=5):
    """
    Aplica aumento de datos a una imagen y guarda las nuevas imágenes.
    (Ejemplo simplificado, idealmente esto se integra en el pipeline de entrenamiento)
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    image = cv2.imread(image_path)
    if image is None:
        print(f"Advertencia: No se pudo cargar la imagen {image_path}")
        return

    # Keras ImageDataGenerator espera un batch (1, H, W, C)
    image_batch = np.expand_dims(image, axis=0)

    # Define un generador de aumento de datos
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True, # Cuidado si los gestos tienen simetría asimétrica
        fill_mode='nearest'
    )

    count = 0
    for batch in datagen.flow(image_batch, batch_size=1,
                             save_to_dir=target_folder,
                             save_prefix=f"aug_{os.path.basename(image_path).split('.')[0]}",
                             save_format='jpg'):
        count += 1
        if count >= num_augmentations_per_image:
            break

# Integración en el script principal (main.py):
# Después de crear el CSV de anotaciones:
# annotations_df = create_annotations_file(preprocessed_base_dir)
# analyze_dataset_distribution(annotations_df)

# Para balancear el dataset (ejemplo conceptual para oversampling):
# Si detectas que 'seña_X' tiene pocas muestras, puedes:
# minority_class_samples = annotations_df[annotations_df['label'] == 'seña_X']
# target_count = annotations_df['label'].value_counts().max() # O un número deseado
#
# for index, row in minority_class_samples.iterrows():
#    full_image_path = os.path.join(preprocessed_base_dir, row['image_path'])
#    target_aug_folder = os.path.join(preprocessed_base_dir, row['label']) # Guardar en la misma carpeta de la seña
#    # Calcula cuántas aumentaciones necesitas para alcanzar el objetivo
#    num_needed = target_count - len(minority_class_samples)
#    if num_needed > 0:
#        apply_data_augmentation_for_balancing(full_image_path, target_aug_folder, num_needed // len(minority_class_samples) + 1)
#
# Después de las aumentaciones, recrea el archivo de anotaciones para incluir las nuevas imágenes.
# annotations_df_balanced = create_annotations_file(preprocessed_base_dir)
# analyze_dataset_distribution(annotations_df_balanced)

# División del dataset (después de tener el dataset balanceado en el CSV)
def split_dataset(annotations_df, test_size=0.15, val_size=0.15, random_state=42):
    """
    Divide el dataset en conjuntos de entrenamiento, validación y prueba.

    Args:
        annotations_df (pd.DataFrame): DataFrame con 'image_path' y 'label'.
        test_size (float): Proporción del dataset para el conjunto de prueba.
        val_size (float): Proporción del dataset para el conjunto de validación.
        random_state (int): Semilla para reproducibilidad.

    Returns:
        tuple: DataFrames para entrenamiento, validación y prueba.
    """
    # Primero, dividir en entrenamiento + validación y prueba
    train_val_df, test_df = train_test_split(
        annotations_df, test_size=test_size, stratify=annotations_df['label'], random_state=random_state
    )

    # Luego, dividir el conjunto de entrenamiento + validación en entrenamiento y validación
    # Ajustamos el tamaño de validación en proporción al nuevo conjunto (train_val_df)
    val_relative_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_relative_size, stratify=train_val_df['label'], random_state=random_state
    )

    print(f"\nNúmero de muestras en el conjunto de entrenamiento: {len(train_df)}")
    print(f"Número de muestras en el conjunto de validación: {len(val_df)}")
    print(f"Número de muestras en el conjunto de prueba: {len(test_df)}")

    return train_df, val_df, test_df

# Uso en el script principal:
# train_df, val_df, test_df = split_dataset(annotations_df_balanced)
# # Guarda estos DataFrames o úsalos para cargar los datos durante el entrenamiento
# train_df.to_csv("data/annotations/train_annotations.csv", index=False)
# val_df.to_csv("data/annotations/val_annotations.csv", index=False)
# test_df.to_csv("data/annotations/test_annotations.csv", index=False)
