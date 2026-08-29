"""
Módulo de utilitários para o projeto EPI Finder.

Contém funções para manipulação de coordenadas, cálculo de métricas de visão
computacional (IoU), operações matriciais com NumPy e renderização visual
com OpenCV.
"""

from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


# Mapeamento padrão de classes do EPI Finder (Novo Dataset Roboflow)
# 0: head -> Pessoa sem capacete / cabeça desprotegida (Infração / Alerta)
# 1: helmet -> Pessoa com capacete de proteção (Seguro / Conforme)
DEFAULT_CLASSES: Dict[int, str] = {
    0: "head",
    1: "helmet"
}

# Paleta de cores padrão no padrão BGR (OpenCV)
# 0: head (Sem capacete) -> Vermelho (Alerta/Infração)
# 1: helmet (Com capacete) -> Verde (Seguro/Conforme)
CLASS_COLORS_BGR: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 255),    # Vermelho (Alerta)
    1: (0, 255, 0)     # Verde (Seguro)
}


def yolo_to_xyxy(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    img_width: int,
    img_height: int
) -> Tuple[int, int, int, int]:
    """
    Converte coordenadas de caixa delimitadora no formato YOLO normalizado [0.0, 1.0]
    para coordenadas absolutas de pixel em formato [x1, y1, x2, y2].

    Args:
        x_center: Coordenada X central normalizada (0 a 1).
        y_center: Coordenada Y central normalizada (0 a 1).
        width: Largura normalizada da caixa (0 a 1).
        height: Altura normalizada da caixa (0 a 1).
        img_width: Largura total da imagem em pixels.
        img_height: Altura total da imagem em pixels.

    Returns:
        Tupla de inteiros (x1, y1, x2, y2) representando o canto superior esquerdo
        e inferior direito em pixels.
    """
    x1 = int(round((x_center - width / 2.0) * img_width))
    y1 = int(round((y_center - height / 2.0) * img_height))
    x2 = int(round((x_center + width / 2.0) * img_width))
    y2 = int(round((y_center + height / 2.0) * img_height))

    # Garante que as coordenadas permaneçam dentro dos limites da imagem
    x1 = max(0, min(x1, img_width - 1))
    y1 = max(0, min(y1, img_height - 1))
    x2 = max(0, min(x2, img_width))
    y2 = max(0, min(y2, img_height))

    return x1, y1, x2, y2


def xyxy_to_yolo(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    img_width: int,
    img_height: int
) -> Tuple[float, float, float, float]:
    """
    Converte coordenadas em pixels [x1, y1, x2, y2] para o formato YOLO normalizado.

    Args:
        x1, y1: Canto superior esquerdo em pixels.
        x2, y2: Canto inferior direito em pixels.
        img_width: Largura total da imagem em pixels.
        img_height: Altura total da imagem em pixels.

    Returns:
        Tupla (x_center, y_center, width, height) normalizada entre 0.0 e 1.0.
    """
    box_w = x2 - x1
    box_h = y2 - y1
    x_center = (x1 + box_w / 2.0) / img_width
    y_center = (y1 + box_h / 2.0) / img_height
    norm_w = box_w / img_width
    norm_h = box_h / img_height

    return float(x_center), float(y_center), float(norm_w), float(norm_h)


def compute_iou(
    box1: Union[List[float], Tuple[float, ...], np.ndarray],
    box2: Union[List[float], Tuple[float, ...], np.ndarray]
) -> float:
    """
    Calcula a Intersection over Union (IoU) entre duas caixas no formato [x1, y1, x2, y2].
    Implementação matemática do zero utilizando operações NumPy (np.maximum e np.minimum).

    A fórmula matemática:
        IoU = Area_Intersecao / Area_Uniao
        Area_Uniao = Area(box1) + Area(box2) - Area_Intersecao

    Args:
        box1: Coordenadas da primeira caixa [x1, y1, x2, y2].
        box2: Coordenadas da segunda caixa [x1, y1, x2, y2].

    Returns:
        Valor de IoU entre 0.0 (sem sobreposição) e 1.0 (sobreposição idêntica).
    """
    b1 = np.asarray(box1, dtype=np.float32)
    b2 = np.asarray(box2, dtype=np.float32)

    # Coordenadas do retângulo de interseção
    inter_x1 = np.maximum(b1[0], b2[0])
    inter_y1 = np.maximum(b1[1], b2[1])
    inter_x2 = np.minimum(b1[2], b2[2])
    inter_y2 = np.minimum(b1[3], b2[3])

    # Largura e altura da interseção (se não houver sobreposição, max garante 0.0)
    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    intersection_area = inter_w * inter_h

    # Áreas de cada caixa individual
    area_b1 = np.maximum(0.0, b1[2] - b1[0]) * np.maximum(0.0, b1[3] - b1[1])
    area_b2 = np.maximum(0.0, b2[2] - b2[0]) * np.maximum(0.0, b2[3] - b2[1])

    # Área da União
    union_area = area_b1 + area_b2 - intersection_area

    if union_area <= 0.0:
        return 0.0

    return float(intersection_area / union_area)


def compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Calcula a matriz de IoU entre N caixas em boxes1 e M caixas em boxes2.
    Vetorização completa com NumPy para alta performance (broadcasting).

    Args:
        boxes1: Array de formato (N, 4) com caixas [x1, y1, x2, y2].
        boxes2: Array de formato (M, 4) com caixas [x1, y1, x2, y2].

    Returns:
        Matriz de formato (N, M) contendo o IoU entre cada par de caixas.
    """
    boxes1 = np.asarray(boxes1, dtype=np.float32)
    boxes2 = np.asarray(boxes2, dtype=np.float32)

    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    # Interseção com broadcasting: (N, 1, 2) e (1, M, 2)
    inter_top_left = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    inter_bottom_right = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])

    inter_wh = np.maximum(0.0, inter_bottom_right - inter_top_left)
    inter_area = inter_wh[:, :, 0] * inter_wh[:, :, 1]

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    union_area = area1[:, None] + area2[None, :] - inter_area
    union_area = np.maximum(union_area, 1e-8)

    return inter_area / union_area


def extract_roi(image: np.ndarray, box_xyxy: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Recorta a Região de Interesse (Region of Interest - ROI) da imagem
    utilizando fatiamento (slicing) direto de matriz NumPy: img[y1:y2, x1:x2].

    Args:
        image: Array NumPy representando a imagem (H, W, C).
        box_xyxy: Coordenadas em pixels (x1, y1, x2, y2).

    Returns:
        Array NumPy contendo a região recortada.
    """
    x1, y1, x2, y2 = map(int, box_xyxy)
    return image[y1:y2, x1:x2].copy()


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Converte uma imagem de BGR (formato padrão OpenCV) para RGB (Matplotlib/PyTorch)
    utilizando fatiamento reverso de matriz NumPy: img[:, :, ::-1].

    Args:
        image: Array NumPy de imagem em padrão BGR.

    Returns:
        Array NumPy de imagem em padrão RGB.
    """
    return image[:, :, ::-1]


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normaliza os valores de intensidade dos pixels de [0, 255] (uint8)
    para [0.0, 1.0] (float32).

    Args:
        image: Imagem em uint8 com valores entre 0 e 255.

    Returns:
        Array float32 normalizado no intervalo [0.0, 1.0].
    """
    return (image.astype(np.float32)) / 255.0


def load_yolo_annotation(label_path: str) -> np.ndarray:
    """
    Lê um arquivo de texto de anotação no formato YOLO.

    Cada linha do arquivo contém:
    <class_id> <x_center> <y_center> <width> <height>

    Args:
        label_path: Caminho para o arquivo .txt de anotação.

    Returns:
        Array NumPy 2D de formato (N, 5) com as anotações. Retorna array vazio (0, 5)
        se o arquivo não existir ou estiver vazio.
    """
    try:
        data = []
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    coords = [float(p) for p in parts[1:5]]
                    data.append([cls_id] + coords)

        if not data:
            return np.empty((0, 5), dtype=np.float32)

        return np.asarray(data, dtype=np.float32)
    except FileNotFoundError:
        return np.empty((0, 5), dtype=np.float32)


def draw_bounding_boxes(
    image: np.ndarray,
    boxes_xyxy: Union[List, np.ndarray],
    class_ids: Union[List[int], np.ndarray],
    confidences: Optional[Union[List[float], np.ndarray]] = None,
    class_names: Optional[Dict[int, str]] = None,
    thickness: int = 2
) -> np.ndarray:
    """
    Desenha caixas delimitadoras e etiquetas na imagem utilizando OpenCV.
    - head (0): Caixa e texto em Vermelho (Infração / Alerta).
    - helmet (1): Caixa e texto em Verde (Seguro / Conforme).

    Args:
        image: Imagem de entrada em formato BGR.
        boxes_xyxy: Lista ou array de caixas [x1, y1, x2, y2].
        class_ids: Lista ou array com os IDs de classe.
        confidences: Opcional, lista/array com as pontuações de confiança [0.0 - 1.0].
        class_names: Mapeamento {id: nome}. Padrão usa DEFAULT_CLASSES.
        thickness: Espessura da borda da caixa.

    Returns:
        Cópia da imagem com as anotações desenhadas.
    """
    canvas = image.copy()
    names = class_names or DEFAULT_CLASSES

    for idx, (box, cls_id) in enumerate(zip(boxes_xyxy, class_ids)):
        x1, y1, x2, y2 = map(int, box)
        cls_id = int(cls_id)

        # Escolhe cor com base na classe
        color = CLASS_COLORS_BGR.get(cls_id, (255, 255, 255))
        class_name = names.get(cls_id, f"Class {cls_id}")

        # Monta o rótulo de texto
        if confidences is not None and idx < len(confidences):
            conf = float(confidences[idx])
            label = f"{class_name} {conf:.2f}"
        else:
            label = class_name

        # Desenha o retângulo da caixa delimitadora
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

        # Desenha a barra de fundo para o texto ficar legível
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

        text_y1 = max(0, y1 - text_h - baseline - 4)
        text_y2 = y1
        text_x2 = min(canvas.shape[1], x1 + text_w + 4)

        cv2.rectangle(canvas, (x1, text_y1), (text_x2, text_y2), color, -1)
        cv2.putText(
            canvas,
            label,
            (x1 + 2, text_y2 - baseline),
            font,
            font_scale,
            (255, 255, 255) if cls_id == 0 else (0, 0, 0),
            font_thickness,
            cv2.LINE_AA
        )

    return canvas
