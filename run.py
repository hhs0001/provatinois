import os
import numpy as np
import pydicom
import cv2
from skimage import filters

def load_dicom_images(path):
    images = []
    for file in os.listdir(path):
        if file.endswith(".dcm"):
            ds = pydicom.dcmread(os.path.join(path, file))
            images.append(ds.pixel_array)
    return images

from skimage.morphology import closing, square

from skimage.measure import label, regionprops

def segment_tissues(image):
    # Normalizar a imagem
    normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Cortar 30% da parte inferior da imagem
    cut_image = normalized_image[:int(normalized_image.shape[0] * 0.7), :]

    # Filtro de mediana para reduzir ruído
    filtered_image = cv2.bilateralFilter(cut_image, 5, 75, 75)

    # Máscara do fundo (pixels com valor 0)
    background_mask = (filtered_image == 0)

    # Máscara da mama (todos os pixels que não são de fundo)
    breast_mask = ~background_mask

    # Operação de fechamento para preencher pequenos buracos na máscara
    breast_mask = closing(breast_mask, square(5))

    # Rotulagem das regiões da máscara
    label_image = label(breast_mask)
    regions = regionprops(label_image)

    # Encontrar a maior região (presumivelmente, a mama)
    largest_region = max(regions, key=lambda region: region.area)

    # Criar uma nova máscara contendo apenas a maior região
    new_breast_mask = label_image == largest_region.label

    # Limiarização de Otsu para segmentar os tecidos
    thresholds = filters.threshold_multiotsu(filtered_image[new_breast_mask], classes=3)
    if isinstance(thresholds, np.ndarray):
        binary_image_fatty = np.zeros_like(filtered_image)
        binary_image_fatty[new_breast_mask] = (filtered_image[new_breast_mask] <= thresholds[0]).astype(np.uint8) * 255

        binary_image_fibroglandular = np.zeros_like(filtered_image)
        binary_image_fibroglandular[new_breast_mask] = ((filtered_image[new_breast_mask] > thresholds[0]) & (filtered_image[new_breast_mask] <= thresholds[1])).astype(np.uint8) * 255

        binary_image_prosthesis = np.zeros_like(filtered_image)
        binary_image_prosthesis[new_breast_mask] = (filtered_image[new_breast_mask] > thresholds[1]).astype(np.uint8) * 255
    else:  # se thresholds é um valor escalar
        # nesse caso, consideramos que a imagem inteira é um único tecido
        binary_image_fatty = (filtered_image <= thresholds).astype(np.uint8) * 255
        binary_image_fibroglandular = np.zeros_like(binary_image_fatty)
        binary_image_prosthesis = np.zeros_like(binary_image_fatty)

    # Colorindo os tecidos
    color_image = np.zeros((*filtered_image.shape, 3), dtype=np.uint8)  # Cria uma nova imagem colorida com o mesmo tamanho da imagem original
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
