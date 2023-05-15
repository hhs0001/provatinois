import os
import numpy as np
import pydicom
import cv2
from skimage import filters, morphology

def load_dicom_images(path):
    images = []
    for file in os.listdir(path):
        if file.endswith(".dcm"):
            ds = pydicom.dcmread(os.path.join(path, file))
            images.append(ds.pixel_array)
    return images


def segment_tissues(image):
    # Normalizar a imagem
    normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Filtro de mediana para reduzir ruído
    filtered_image = cv2.medianBlur(normalized_image, 5)

    # Aqui você precisa implementar uma segmentação mais avançada para distinguir os diferentes tecidos
    # Este é apenas um exemplo básico
    _, binary_image_fibroglandular = cv2.threshold(filtered_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, binary_image_fatty = cv2.threshold(filtered_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Supondo que a prótese mamária seja o tecido restante
    binary_image_prosthesis = cv2.bitwise_and(cv2.bitwise_not(binary_image_fibroglandular), cv2.bitwise_not(binary_image_fatty))

    # Colorindo os tecidos
    color_image = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2BGR)
    color_image[binary_image_fibroglandular == 255] = [255, 0, 0]  # Fibroglandular em vermelho
    color_image[binary_image_fatty == 255] = [0, 255, 0]  # Gorduroso em verde
    color_image[binary_image_prosthesis == 255] = [0, 0, 255]  # Prótese mamária em azul

    return color_image, binary_image_fibroglandular, binary_image_fatty, binary_image_prosthesis

def quantify_and_save(images, output_path):
    tissue_percentages = []
    for idx, img in enumerate(images):
        segmented_img, img_fibroglandular, img_fatty, img_prosthesis = segment_tissues(img)

        total_area = np.size(segmented_img) / 3
        fibroglandular_area = np.count_nonzero(img_fibroglandular)
        fatty_area = np.count_nonzero(img_fatty)
        prosthesis_area = np.count_nonzero(img_prosthesis)

        percentages = {
            "fibroglandular": (fibroglandular_area / total_area) * 100,
            "fatty": (fatty_area / total_area) * 100,
            "prosthesis": (prosthesis_area / total_area) * 100
        }
        tissue_percentages.append(percentages)

        cv2.imwrite(os.path.join(output_path, f'segmented_image_{idx + 1}.png'), segmented_img)

    return tissue_percentages

def main(input_path, output_path):
    images = load_dicom_images(input_path)
    tissue_percentages = quantify_and_save(images, output_path)

    # Imprimir porcentagens de tecido fibroglandular, gorduroso e prótese mamária
    for idx, perc in enumerate(tissue_percentages):
        print(f'Imagem {idx + 1}:')
        print(f'\t{perc["fibroglandular"]:.2f}% de tecido fibroglandular')
        print(f'\t{perc["fatty"]:.2f}% de tecido gorduroso')
        print(f'\t{perc["prosthesis"]:.2f}% de prótese mamária')

if __name__ == "__main__":
    input_path = "./dicon"
    output_path = "./out"
    main(input_path, output_path)

