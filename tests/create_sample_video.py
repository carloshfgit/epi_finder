"""
Gera um clipe curto de vídeo sintético (MP4) a partir de imagens reais de teste,
simulando movimento de trabalhadores para validar o pipeline de tracking e desduplicação.
"""

from pathlib import Path
import cv2
import numpy as np

def generate_synthetic_video(output_path: str = "data/sample_cctv.mp4", num_frames: int = 40):
    images_dir = Path("data/dataset/test/images")
    img_violation = images_dir / "hard_hat_workers149_png.rf.0e440314a54738e5d32d30293fe41326.jpg"
    img_helmet = images_dir / "hard_hat_workers1_png.rf.eb448f1482a8ae423d6d2df59509596d.jpg"

    if not img_violation.exists():
        print("Imagem de teste não encontrada.")
        return

    frame_v = cv2.imread(str(img_violation))
    h, w, _ = frame_v.shape

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_p), fourcc, 10.0, (w, h))

    for i in range(num_frames):
        # Translação suave de 1 pixel por frame para testar o tracking em movimento
        dx = int((i % 10) * 0.5)
        dy = int((i % 6) * 0.5)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        frame = cv2.warpAffine(frame_v, M, (w, h))
        writer.write(frame)

    writer.release()
    print(f"✅ Vídeo de teste com infrações gerado com sucesso: {out_p} ({w}x{h}, {num_frames} quadros)")

if __name__ == "__main__":
    generate_synthetic_video()
