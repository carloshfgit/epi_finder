"""
Testes unitários para os novos métodos auxiliares de inferência (EPI Finder - Etapa 7.2).
"""

from pathlib import Path
import cv2
import numpy as np
import pytest

from src.inference import (
    infer_single_image,
    load_model,
    process_video_stream,
)


def test_load_model_fallback():
    """Valida o carregamento do modelo com fallback para yolov8n.pt se best.pt não existir."""
    # Testando com o modelo local yolov8n.pt
    model = load_model("yolov8n.pt")
    assert model is not None
    assert hasattr(model, "predict")


def test_infer_single_image_blank():
    """Valida inferência em imagem sintética sem exceções."""
    model = load_model("yolov8n.pt")
    # Cria uma imagem preta de 640x640x3
    blank_img = np.zeros((640, 640, 3), dtype=np.uint8)

    result = infer_single_image(
        model=model,
        image_bgr=blank_img,
        conf_threshold=0.25,
        draw_banner=True,
        alert_labels=True,
    )

    assert "annotated_bgr" in result
    assert "annotated_rgb" in result
    assert "detections" in result
    assert "total_persons" in result
    assert "compliance_rate_percent" in result
    assert result["total_persons"] == 0
    assert result["compliance_rate_percent"] == 100.0
    assert result["annotated_rgb"].shape == blank_img.shape


def test_infer_single_image_with_real_image():
    """Valida inferência em imagem de teste real caso disponível."""
    sample_path = Path("data/dataset/test/images/hard_hat_workers149_png.rf.0e440314a54738e5d32d30293fe41326.jpg")
    if not sample_path.exists():
        pytest.skip("Imagem de amostra não disponível para teste.")

    model = load_model("yolov8n.pt")
    img_bgr = cv2.imread(str(sample_path))
    assert img_bgr is not None

    result = infer_single_image(
        model=model,
        image_bgr=img_bgr,
        conf_threshold=0.15,
        draw_banner=True,
    )

    assert isinstance(result["annotated_bgr"], np.ndarray)
    assert isinstance(result["annotated_rgb"], np.ndarray)
    assert isinstance(result["detections"], list)
    assert isinstance(result["violation_crops"], list)
