import os
import pandas as pd
from src.preproceso import extract_frames, normalize_image, segment_hand_hsv, create_annotations_file, \
    analyze_dataset_distribution, split_dataset
import cv2  # Necesario para imwrite en el bucle principal


def run_preprocessing_pipeline():
    """
    Orquesta el pipeline completo de preprocesamiento de datos.
    """
    videos_folder = "C:/Users/fylco/Desktop/Test/data/videos"
    output_frames_base_folder = "C:/Users/fylco/Desktop/Test/data/frames"
    output_preprocessed_base_folder = "C:/Users/fylco/Desktop/Test/data/preprocessed"
    annotations_csv_path = "C:/Users/fylco/Desktop/Test/data/annotations/annotations.csv"

    frame_interval = 2  # Extraer cada 2do fotograma
    target_image_size = (224, 224)  # Tamaño de imagen para normalización

    print("--- 2.1.1. Extracción de Fotogramas ---")
    for video_file in os.listdir(videos_folder):
        if video_file.endswith(('.mp4', '.avi', '.mov')):  # Asegura que procesas solo archivos de video
            video_path = os.path.join(videos_folder, video_file)
            video_name_without_ext = os.path.splitext(video_file)[0]

            # Subcarpeta para frames específicos de este video
            video_frames_folder = os.path.join(output_frames_base_folder, video_name_without_ext)
            if not os.path.exists(video_frames_folder):
                os.makedirs(video_frames_folder)

            print(f"Procesando video: {video_file}")
            extract_frames(video_path, video_frames_folder, frame_interval)

    print("\n--- 2.1.2. & 2.1.3. Normalización y Segmentación ---")
    # Itera sobre las carpetas de fotogramas extraídos (que son las etiquetas de las señas)
    for sign_label_folder in os.listdir(output_frames_base_folder):
        sign_frames_path = os.path.join(output_frames_base_folder, sign_label_folder)

        if os.path.isdir(sign_frames_path):  # Asegurarse de que es una carpeta
            # Crear la carpeta de salida preprocesada para esta seña
            preprocessed_sign_folder = os.path.join(output_preprocessed_base_folder, sign_label_folder)
            if not os.path.exists(preprocessed_sign_folder):
                os.makedirs(preprocessed_sign_folder)

            frame_files = sorted([f for f in os.listdir(sign_frames_path) if f.endswith(('.jpg', '.png'))])

            print(f"Preprocesando fotogramas para la seña: {sign_label_folder}")
            for frame_file in tqdm(frame_files, desc=f"Procesando {sign_label_folder}"):
                frame_path = os.path.join(sign_frames_path, frame_file)
                image = cv2.imread(frame_path)

                if image is not None:
                    # Aplicar segmentación
                    segmented_image = segment_hand_hsv(image)

                    # Aplicar normalización (redimensionamiento y escala)
                    # Es importante que la normalización sea después de la segmentación si la segmentación produce imágenes de tamaño diferente
                    # o si se quiere normalizar el fondo negro también.
                    preprocessed_image = normalize_image(segmented_image, target_image_size)

                    # Convertir de float32 a uint8 antes de guardar, y escalar de 0-1 a 0-255
                    # Esto es necesario porque cv2.imwrite espera uint8 y valores 0-255
                    output_image_display = (preprocessed_image * 255).astype(np.uint8)

                    output_path = os.path.join(preprocessed_sign_folder, frame_file)
                    cv2.imwrite(output_path, output_image_display)
                else:
                    print(f"Advertencia: No se pudo cargar el fotograma {frame_path}")

    print("\n--- 2.1.4. Anotación de Datos ---")
    annotations_df = create_annotations_file(output_preprocessed_base_folder, annotations_csv_path)

    print("\n--- 2.1.5. Construcción de un Dataset Balanceado (Análisis y División) ---")
    analyze_dataset_distribution(annotations_df)  # Muestra la distribución actual

    # Nota: El aumento de datos para balanceo se haría aquí si se generan imágenes nuevas
    # o de forma "on-the-fly" durante el entrenamiento. Para esta etapa, solo se analiza la distribución
    # y se divide el dataset.

    train_df, val_df, test_df = split_dataset(annotations_df, test_size=0.15, val_size=0.15, random_state=42)

    # Guarda los DataFrames divididos para uso posterior en el entrenamiento
    train_df.to_csv("C:/Users/fylco/Desktop/Test/data/annotations/train_annotations.csv", index=False)
    val_df.to_csv("C:/Users/fylco/Desktop/Test/data/annotations/val_annotations.csv", index=False)
    test_df.to_csv("C:/Users/fylco/Desktop/Test/data/annotations/test_annotations.csv", index=False)
    print("\nPipeline de preprocesamiento completado exitosamente.")


if __name__ == '__main__':
    run_preprocessing_pipeline()