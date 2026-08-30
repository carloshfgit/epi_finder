"""
Módulo de inferência operacional e relatórios de conformidade (EPI Finder).

Permite a execução do detector de capacetes (head vs helmet) em múltiplas fontes:
- Imagens isoladas (.jpg, .png, .jpeg, etc.)
- Diretórios contendo lotes de imagens
- Arquivos de vídeo (.mp4, .avi, .mkv, etc.)
- Câmeras e streams ao vivo (webcam / RTSP)

Gera alertas visuais dinâmicos (OpenCV), recortes de evidências de infrações (NumPy)
e relatórios tabulares consolidados de conformidade para Segurança do Trabalho (Pandas).
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, Generator, List, Optional, Set, Tuple, Union

# Garante que a raiz do projeto esteja no sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

try:
    from src.utils import (
        ALERT_CLASSES,
        CLASS_COLORS_BGR,
        DEFAULT_CLASSES,
        draw_bounding_boxes,
        draw_telemetry_banner,
        extract_roi,
    )
except ImportError:
    from utils import (
        ALERT_CLASSES,
        CLASS_COLORS_BGR,
        DEFAULT_CLASSES,
        draw_bounding_boxes,
        draw_telemetry_banner,
        extract_roi,
    )


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VALID_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}


def get_default_device() -> str:
    """
    Identifica dinamicamente o dispositivo mais adequado (CUDA GPU ou CPU).

    Returns:
        '0' se GPU com CUDA estiver disponível, senão 'cpu'.
    """
    return "0" if torch.cuda.is_available() else "cpu"


class TemporalTrackerFilter:
    """
    Filtro de estabilização temporal para Multi-Object Tracking (MOT - Etapa 7.1).

    Minimiza ruídos e falsos alertas transitórios (1 a 2 frames) causados por
    movimentações bruscas da cabeça, sombras repentinas ou oclusões parciais,
    exigindo consistência de detecção por N quadros consecutivos antes de
    confirmar uma mudança definitiva de status.
    """

    def __init__(self, min_consecutive_frames: int = 3, max_history: int = 30) -> None:
        """
        Args:
            min_consecutive_frames: Quantidade mínima de quadros consecutivos para confirmação.
            max_history: Tamanho máximo da janela de histórico por indivíduo (evita vazamento de memória).
        """
        self.min_consecutive_frames = max(1, min_consecutive_frames)
        self.max_history = max_history
        self.history: Dict[int, List[int]] = {}
        self.stabilized_classes: Dict[int, int] = {}
        self.confirmed_violations: Set[int] = set()

    def update(self, track_id: int, detected_class_id: int) -> int:
        """
        Atualiza o buffer de detecções do track_id e retorna a classe estabilizada.

        Args:
            track_id: Identificador numérico do objeto rastreado.
            detected_class_id: ID da classe detectada no frame atual (0=head, 1=helmet).

        Returns:
            ID da classe estabilizada após filtragem temporal.
        """
        if track_id not in self.history:
            self.history[track_id] = []

        self.history[track_id].append(detected_class_id)
        if len(self.history[track_id]) > self.max_history:
            self.history[track_id].pop(0)

        # Se ainda não possui observações suficientes, assume a detecção atual
        if len(self.history[track_id]) < self.min_consecutive_frames:
            if track_id not in self.stabilized_classes:
                self.stabilized_classes[track_id] = detected_class_id
            return self.stabilized_classes[track_id]

        recent = self.history[track_id][-self.min_consecutive_frames:]
        # Confirma mudança somente se todos os últimos N frames forem idênticos
        if all(c == detected_class_id for c in recent):
            self.stabilized_classes[track_id] = detected_class_id
            if detected_class_id == 0:
                self.confirmed_violations.add(track_id)
        elif track_id not in self.stabilized_classes:
            self.stabilized_classes[track_id] = detected_class_id

        return self.stabilized_classes[track_id]

    def is_violation_confirmed(self, track_id: int) -> bool:
        """Verifica se a infração (classe 0 - head) já atingiu o critério de confirmação temporal."""
        return track_id in self.confirmed_violations


def resolve_tracker_config(
    tracker_name_or_path: str,
    conf_threshold: float,
    cache_dir: Path
) -> str:
    """
    Garante que os limiares de associação do tracker (ByteTrack/BoT-SORT)
    sejam compatíveis com o limiar de confiança solicitado pelo usuário.
    Se conf_threshold for menor que o padrão (0.25), gera dinamicamente
    uma configuração YAML calibrada para garantir que os objetos recebam Track IDs.
    """
    tracker_p = Path(tracker_name_or_path)
    if tracker_p.exists() and tracker_p.is_file():
        return str(tracker_p.resolve())

    # Se for bytetrack e conf_threshold for inferior ao padrão 0.25:
    if "bytetrack" in tracker_name_or_path.lower() and conf_threshold < 0.25:
        custom_yaml = cache_dir / "calibrated_bytetrack.yaml"
        custom_yaml.parent.mkdir(parents=True, exist_ok=True)
        content = (
            "tracker_type: bytetrack\n"
            f"track_high_thresh: {float(conf_threshold)}\n"
            f"track_low_thresh: {float(max(0.0001, conf_threshold * 0.5))}\n"
            f"new_track_thresh: {float(conf_threshold)}\n"
            "track_buffer: 30\n"
            "match_thresh: 0.8\n"
            "fuse_score: True\n"
        )
        custom_yaml.write_text(content, encoding="utf-8")
        return str(custom_yaml)

    return tracker_name_or_path


class ComplianceAuditor:
    """
    Agregador de auditoria e telemetria de Segurança e Saúde no Trabalho (SST).

    Armazena eventos de detecção, calcula métricas estatísticas e consolida
    relatórios de conformidade utilizando Pandas.
    Suporta Multi-Object Tracking (MOT), filtragem temporal e desduplicação por Track ID
    (Opção B: grava no CSV apenas a primeira detecção de cada indivíduo rastreado).
    """

    def __init__(self, camera_id: str = "Camera-01", dedup_by_track: bool = False) -> None:
        """
        Args:
            camera_id: Identificador do posto ou câmera de monitoramento.
            dedup_by_track: Se True, filtra o relatório CSV para conter somente
                           a primeira detecção de cada track_id (Opção B).
        """
        self.camera_id = camera_id
        self.dedup_by_track = dedup_by_track
        self.records: List[Dict[str, Any]] = []
        self.frame_summaries: List[Dict[str, Any]] = []
        self.start_time = datetime.now()

        # Estruturas de conjunto para rastreamento de indivíduos únicos
        self.seen_track_ids: Set[int] = set()
        self.violation_track_ids: Set[int] = set()
        self.conformant_track_ids: Set[int] = set()
        self.logged_track_ids: Set[int] = set()

    def record_detection(
        self,
        source: str,
        frame_id: int,
        detection_id: int,
        class_id: int,
        confidence: float,
        bbox_xyxy: Tuple[int, int, int, int],
        track_id: Optional[int] = None,
        timestamp: Optional[datetime] = None,
        crop_path: Optional[str] = None,
    ) -> bool:
        """
        Registra uma detecção individual (objeto head ou helmet).

        Args:
            source: Nome do arquivo ou fluxo de entrada.
            frame_id: Índice do frame processado.
            detection_id: Identificador da caixa no frame atual.
            class_id: Classe predita (0=head, 1=helmet).
            confidence: Probabilidade / confiança da detecção.
            bbox_xyxy: Coordenadas em pixels (x1, y1, x2, y2).
            track_id: Identificador único persistente do indivíduo (MOT).
            timestamp: Data/hora da captura.
            crop_path: Caminho da foto de evidência recortada (se houver).

        Returns:
            True se o registro foi salvo no relatório; False se foi descartado por desduplicação.
        """
        ts = timestamp or datetime.now()
        x1, y1, x2, y2 = bbox_xyxy
        is_violation = (class_id == 0)

        # Atualiza métricas de instâncias únicas se houver tracking
        if track_id is not None:
            self.seen_track_ids.add(track_id)
            if is_violation:
                self.violation_track_ids.add(track_id)
            else:
                self.conformant_track_ids.add(track_id)

            # Opção B: filtrar diretamente o compliance_report.csv para conter
            # somente a primeira detecção de cada track_id
            if self.dedup_by_track and track_id in self.logged_track_ids:
                return False
            self.logged_track_ids.add(track_id)
        elif self.dedup_by_track:
            # Protege contra poluição repetitiva de caixas espúrias sem track_id
            if -1 in self.logged_track_ids:
                return False
            self.logged_track_ids.add(-1)

        record = {
            "timestamp": ts.isoformat(),
            "camera_id": self.camera_id,
            "source": str(source),
            "frame_id": frame_id,
            "detection_id": detection_id,
            "track_id": track_id if track_id is not None else -1,
            "class_id": int(class_id),
            "class_name": DEFAULT_CLASSES.get(class_id, f"class_{class_id}"),
            "status_label": ALERT_CLASSES.get(class_id, f"Class {class_id}"),
            "confidence": float(round(confidence, 4)),
            "is_violation": is_violation,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "box_width": x2 - x1,
            "box_height": y2 - y1,
            "box_area": (x2 - x1) * (y2 - y1),
            "crop_path": crop_path or ""
        }
        self.records.append(record)
        return True

    def record_frame_summary(
        self,
        source: str,
        frame_id: int,
        total_persons: int,
        conformant_count: int,
        violation_count: int,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Registra o resumo agregado de um frame ou imagem processada.
        """
        ts = timestamp or datetime.now()
        compliance_rate = (
            (conformant_count / total_persons * 100.0) if total_persons > 0 else 100.0
        )

        summary = {
            "timestamp": ts.isoformat(),
            "camera_id": self.camera_id,
            "source": str(source),
            "frame_id": frame_id,
            "total_persons": total_persons,
            "conformant_helmets": conformant_count,
            "violations_head": violation_count,
            "compliance_rate": float(round(compliance_rate, 2)),
        }
        self.frame_summaries.append(summary)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Retorna o histórico completo de detecções como um pandas.DataFrame.
        """
        if not self.records:
            return pd.DataFrame(columns=[
                "timestamp", "camera_id", "source", "frame_id", "detection_id",
                "track_id", "class_id", "class_name", "status_label", "confidence",
                "is_violation", "x1", "y1", "x2", "y2", "box_width", "box_height",
                "box_area", "crop_path"
            ])
        return pd.DataFrame(self.records)

    def to_frame_summary_dataframe(self) -> pd.DataFrame:
        """
        Retorna o histórico resumido por frame como um pandas.DataFrame.
        """
        if not self.frame_summaries:
            return pd.DataFrame(columns=[
                "timestamp", "camera_id", "source", "frame_id", "total_persons",
                "conformant_helmets", "violations_head", "compliance_rate"
            ])
        return pd.DataFrame(self.frame_summaries)

    def get_audit_summary(self) -> Dict[str, Any]:
        """
        Calcula indicadores consolidados de conformidade para o relatório final,
        incluindo métricas tradicionais por detecção e métricas de indivíduos únicos via MOT.
        """
        total_records = len(self.records)
        df_det = self.to_dataframe()

        if total_records > 0:
            helmets = int((df_det["class_id"] == 1).sum())
            violations = int((df_det["class_id"] == 0).sum())
            overall_rate = float(round((helmets / total_records) * 100.0, 2))
            mean_conf = float(round(df_det["confidence"].mean(), 4))
            mean_conf_violation = float(round(df_det[df_det["class_id"] == 0]["confidence"].mean(), 4)) if violations > 0 else 0.0
            mean_conf_helmet = float(round(df_det[df_det["class_id"] == 1]["confidence"].mean(), 4)) if helmets > 0 else 0.0
        else:
            helmets = 0
            violations = 0
            overall_rate = 100.0
            mean_conf = 0.0
            mean_conf_violation = 0.0
            mean_conf_helmet = 0.0

        total_frames = len(self.frame_summaries)
        frames_with_violations = 0
        if self.frame_summaries:
            df_frames = self.to_frame_summary_dataframe()
            frames_with_violations = int((df_frames["violations_head"] > 0).sum())

        # Métricas de Indivíduos Únicos (MOT - Etapa 7.1)
        total_tracks = len(self.seen_track_ids)
        unique_violations = len(self.violation_track_ids)
        unique_conformant = len(self.conformant_track_ids)
        unique_rate = (
            float(round((unique_conformant / total_tracks) * 100.0, 2))
            if total_tracks > 0
            else 100.0
        )

        return {
            "camera_id": self.camera_id,
            "session_start": self.start_time.isoformat(),
            "session_end": datetime.now().isoformat(),
            "tracking_enabled": total_tracks > 0,
            "dedup_by_track": self.dedup_by_track,
            "total_frames_processed": total_frames,
            "frames_with_violations": frames_with_violations,
            "total_detections_logged": total_records,
            "conformant_helmets_logged": helmets,
            "violations_head_logged": violations,
            "overall_compliance_rate_percent": overall_rate,
            "unique_persons_tracked": total_tracks,
            "unique_conformant": unique_conformant,
            "unique_violators": unique_violations,
            "unique_compliance_rate_percent": unique_rate,
            "mean_confidence": mean_conf,
            "mean_confidence_helmet": mean_conf_helmet,
            "mean_confidence_violation": mean_conf_violation,
            "status": "CONFORME" if violations == 0 and unique_violations == 0 else "ATENCAO: INFRACOES DETECTADAS",
        }

    def export_reports(
        self,
        csv_path: Union[str, Path],
        json_path: Optional[Union[str, Path]] = None,
        verbose: bool = True
    ) -> Tuple[Path, Optional[Path]]:
        """
        Exporta o relatório detalhado em formato CSV e o sumário executivo em JSON.
        """
        csv_p = Path(csv_path).resolve()
        csv_p.parent.mkdir(parents=True, exist_ok=True)

        df = self.to_dataframe()
        df.to_csv(csv_p, index=False, encoding="utf-8")

        json_p = None
        if json_path is not None:
            json_p = Path(json_path).resolve()
            json_p.parent.mkdir(parents=True, exist_ok=True)
            summary_dict = self.get_audit_summary()
            with open(json_p, "w", encoding="utf-8") as f:
                json.dump(summary_dict, f, indent=2, ensure_ascii=False)

        if verbose:
            print(f"📊 Relatório de Conformidade CSV salvo em: {csv_p}")
            if json_p:
                print(f"📋 Sumário Executivo JSON salvo em:      {json_p}")

        return csv_p, json_p


def run_inference(
    source: Union[str, Path, int],
    weights: Union[str, Path] = "models/best.pt",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    img_size: int = 640,
    device: Optional[str] = None,
    output_dir: Union[str, Path] = "runs/inference",
    report_csv: Optional[Union[str, Path]] = "runs/inference/compliance_report.csv",
    summary_json: Optional[Union[str, Path]] = "runs/inference/summary.json",
    camera_id: str = "Camera-01",
    save_crops: bool = False,
    save_media: bool = True,
    show: bool = False,
    alert_labels: bool = True,
    draw_banner: bool = True,
    enable_tracking: bool = False,
    tracker: str = "bytetrack.yaml",
    min_consecutive_frames: int = 3,
    dedup_by_track: bool = True,
    verbose: bool = True,
) -> ComplianceAuditor:
    """
    Executa o pipeline de inferência operacional e auditoria de SST.
    Suporta Multi-Object Tracking (MOT - Etapa 7.1) com desduplicação de logs e evidências.

    Args:
        source: Caminho para imagem, diretório, vídeo ou índice de webcam (0, 1).
        weights: Caminho para os pesos treinados do modelo YOLO (.pt).
        conf_threshold: Limiar mínimo de confiança [0.0 - 1.0].
        iou_threshold: Limiar de IoU para supressão de não-máximos (NMS).
        img_size: Resolução quadrada para redimensionamento de entrada.
        device: Dispositivo de execução ('0' para GPU CUDA ou 'cpu').
        output_dir: Diretório onde serão gravados os resultados.
        report_csv: Caminho para salvar a tabela CSV com as ocorrências.
        summary_json: Caminho para salvar o sumário consolidado em JSON.
        camera_id: Identificador da câmera / local de monitoramento.
        save_crops: Se True, recorta e salva fotos de cada infração identificada.
        save_media: Se True, grava os arquivos de imagem ou vídeo anotados.
        show: Se True, exibe janela com reprodução em tempo real (OpenCV).
        alert_labels: Se True, exibe 'ALERTA: SEM CAPACETE' e 'Capacete'.
        draw_banner: Se True, sobrepõe a barra de telemetria no topo do frame.
        enable_tracking: Se True, ativa Multi-Object Tracking (ByteTrack/BoT-SORT).
        tracker: Nome do arquivo YAML do tracker ('bytetrack.yaml' ou 'botsort.yaml').
        min_consecutive_frames: Limiar de quadros consecutivos para confirmação temporal.
        dedup_by_track: Se True, filtra o compliance_report.csv para conter apenas
                        a primeira detecção de cada track_id (Opção B).
        verbose: Se True, imprime informações no terminal.

    Returns:
        Instância de ComplianceAuditor contendo todo o histórico de eventos.
    """
    # Resolução de caminhos
    weights_path = Path(weights).resolve()
    if not weights_path.exists():
        fallback_weights = Path("yolov8n.pt").resolve()
        if fallback_weights.exists():
            if verbose:
                print(f"⚠️  Pesos '{weights_path.name}' não encontrados. Utilizando fallback: {fallback_weights}")
            weights_path = fallback_weights
        else:
            raise FileNotFoundError(f"Pesos de modelo não encontrados: {weights_path}")

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    violations_dir = out_dir / "violations"
    if save_crops:
        violations_dir.mkdir(parents=True, exist_ok=True)

    active_device = device if device is not None else get_default_device()

    # Determinar tipo de fonte
    is_webcam = False
    source_path = None
    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        is_webcam = True
        source_idx = int(source)
        source_name = f"Webcam_{source_idx}"
    else:
        source_str = str(source)
        source_path = Path(source_str).resolve()
        source_name = source_path.name

    labels_map = ALERT_CLASSES if alert_labels else DEFAULT_CLASSES

    if verbose:
        print("=" * 78)
        print("🛡️  EPI FINDER - INFERÊNCIA OPERACIONAL E AUDITORIA SST (FASE 7.1)")
        print("=" * 78)
        print(f"📦 Modelo / Pesos       : {weights_path}")
        print(f"🎯 Fonte de Dados       : {source_name}")
        print(f"🎥 ID da Câmera         : {camera_id}")
        print(f"⚡ Dispositivo          : {active_device} ({'GPU CUDA' if active_device != 'cpu' else 'CPU'})")
        print(f"🎯 Limiar Confiança     : {conf_threshold} | IoU (NMS): {iou_threshold}")
        print(f"🔄 Rastreamento (MOT)   : {'Ativo (' + tracker + ')' if enable_tracking else 'Desativado'}")
        if enable_tracking:
            print(f"⏳ Filtro Temporal      : {min_consecutive_frames} quadros consecutivos")
            print(f"🧹 Desduplicação CSV    : {'Opção B: 1 registro por track_id' if dedup_by_track else 'Todos os frames'}")
        print(f"📁 Diretório Saída      : {out_dir}")
        print(f"✂️  Recortar Evidências  : {'Sim (desduplicado por indivíduo)' if save_crops and enable_tracking and dedup_by_track else ('Sim' if save_crops else 'Não')}")
        print("=" * 78)

    # Carregar modelo YOLO
    if verbose:
        print(f"\n📥 Inicializando modelo YOLO...")
    model = YOLO(str(weights_path))

    auditor = ComplianceAuditor(
        camera_id=camera_id,
        dedup_by_track=(dedup_by_track and enable_tracking)
    )

    temporal_filter = (
        TemporalTrackerFilter(min_consecutive_frames=min_consecutive_frames)
        if enable_tracking
        else None
    )
    saved_crop_track_ids: Set[int] = set()

    # 1. PROCESSAMENTO DE IMAGEM ÚNICA OU DIRETÓRIO DE IMAGENS
    if not is_webcam and source_path and (source_path.is_dir() or source_path.suffix.lower() in VALID_IMAGE_EXTENSIONS):
        image_files: List[Path] = []
        if source_path.is_dir():
            for p in sorted(source_path.glob("**/*")):
                if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                    image_files.append(p)
        else:
            image_files.append(source_path)

        if not image_files:
            raise FileNotFoundError(f"Nenhuma imagem válida encontrada em: {source_path}")

        if verbose:
            print(f"\n🖼️  Processando {len(image_files)} imagem(ns)...")

        for idx, img_path in enumerate(image_files, start=1):
            frame = cv2.imread(str(img_path))
            if frame is None:
                if verbose:
                    print(f"⚠️  Falha ao ler imagem: {img_path}")
                continue

            frame_ts = datetime.now()
            results = model.predict(
                source=frame,
                conf=conf_threshold,
                iou=iou_threshold,
                imgsz=img_size,
                device=active_device,
                verbose=False
            )

            # Extração de detecções
            boxes_xyxy: List[Tuple[int, int, int, int]] = []
            class_ids: List[int] = []
            confidences: List[float] = []

            for r in results:
                if r.boxes is not None and len(r.boxes) > 0:
                    b_arr = r.boxes.xyxy.cpu().numpy()
                    c_arr = r.boxes.cls.cpu().numpy().astype(int)
                    conf_arr = r.boxes.conf.cpu().numpy()

                    for b, c, cf in zip(b_arr, c_arr, conf_arr):
                        box_tuple = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
                        boxes_xyxy.append(box_tuple)
                        class_ids.append(int(c))
                        confidences.append(float(cf))

            # Contadores do frame
            total_persons = len(class_ids)
            conformant_count = sum(1 for c in class_ids if c == 1)
            violation_count = sum(1 for c in class_ids if c == 0)
            compliance_rate = (
                (conformant_count / total_persons * 100.0) if total_persons > 0 else 100.0
            )

            # Registro de cada detecção no Auditor
            for det_id, (box, c, cf) in enumerate(zip(boxes_xyxy, class_ids, confidences), start=1):
                crop_path_str = None
                if save_crops and c == 0:
                    crop = extract_roi(frame, box)
                    if crop.size > 0:
                        crop_filename = f"violation_{img_path.stem}_det{det_id}_{frame_ts.strftime('%Y%m%d_%H%M%S_%f')[:19]}.jpg"
                        crop_full_path = violations_dir / crop_filename
                        cv2.imwrite(str(crop_full_path), crop)
                        crop_path_str = str(crop_full_path)

                auditor.record_detection(
                    source=str(img_path.name),
                    frame_id=idx,
                    detection_id=det_id,
                    class_id=c,
                    confidence=cf,
                    bbox_xyxy=box,
                    track_id=None,
                    timestamp=frame_ts,
                    crop_path=crop_path_str,
                )

            # Registra resumo do frame
            auditor.record_frame_summary(
                source=str(img_path.name),
                frame_id=idx,
                total_persons=total_persons,
                conformant_count=conformant_count,
                violation_count=violation_count,
                timestamp=frame_ts,
            )

            # Renderização visual
            annotated_frame = frame.copy()
            if boxes_xyxy:
                annotated_frame = draw_bounding_boxes(
                    image=annotated_frame,
                    boxes_xyxy=boxes_xyxy,
                    class_ids=class_ids,
                    confidences=confidences,
                    class_names=labels_map,
                    track_ids=None,
                    thickness=2
                )

            if draw_banner:
                annotated_frame = draw_telemetry_banner(
                    image=annotated_frame,
                    total_persons=total_persons,
                    conformant_count=conformant_count,
                    violation_count=violation_count,
                    compliance_rate=compliance_rate,
                    camera_id=camera_id,
                    timestamp_str=frame_ts.strftime("%Y-%m-%d %H:%M:%S")
                )

            # Salvar mídia anotada
            if save_media:
                out_img_path = out_dir / f"annotated_{img_path.name}"
                cv2.imwrite(str(out_img_path), annotated_frame)

            if show:
                try:
                    cv2.imshow("EPI Finder - Monitoramento Operacional", annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27 or key == ord("q"):
                        break
                except cv2.error:
                    pass

            if verbose and (idx % 10 == 0 or idx == len(image_files)):
                print(f"  [{idx}/{len(image_files)}] {img_path.name} | Total: {total_persons} | Conformes: {conformant_count} | Infrações: {violation_count} | {compliance_rate:.1f}%")

    # 2. PROCESSAMENTO DE VÍDEO OU STREAM DE CÂMERA
    else:
        video_src: Union[int, str] = source_idx if is_webcam else str(source_path)
        cap = cv2.VideoCapture(video_src)

        if not cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir a fonte de vídeo: {video_src}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        total_frames_est = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        video_writer = None
        if save_media and not is_webcam and source_path:
            out_video_path = out_dir / f"annotated_{source_path.stem}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

        active_tracker = (
            resolve_tracker_config(tracker, conf_threshold, out_dir)
            if enable_tracking
            else tracker
        )

        if verbose:
            mode_str = f"com MOT ({tracker})" if enable_tracking else "sem Tracking"
            print(f"\n🎥 Processando fluxo de vídeo ({width}x{height} @ {fps:.1f} FPS) [{mode_str}]...")
            if total_frames_est > 0:
                print(f"   Total aproximado de quadros: {total_frames_est}")

        frame_idx = 0
        start_proc_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                frame_ts = datetime.now()

                if enable_tracking:
                    results = model.track(
                        source=frame,
                        persist=True,
                        tracker=active_tracker,
                        conf=conf_threshold,
                        iou=iou_threshold,
                        imgsz=img_size,
                        device=active_device,
                        verbose=False
                    )
                else:
                    results = model.predict(
                        source=frame,
                        conf=conf_threshold,
                        iou=iou_threshold,
                        imgsz=img_size,
                        device=active_device,
                        verbose=False
                    )

                boxes_xyxy: List[Tuple[int, int, int, int]] = []
                class_ids: List[int] = []
                confidences: List[float] = []
                track_ids: List[Optional[int]] = []

                for r in results:
                    if r.boxes is not None and len(r.boxes) > 0:
                        b_arr = r.boxes.xyxy.cpu().numpy()
                        c_arr = r.boxes.cls.cpu().numpy().astype(int)
                        conf_arr = r.boxes.conf.cpu().numpy()
                        id_arr = (
                            r.boxes.id.int().cpu().numpy()
                            if (enable_tracking and r.boxes.id is not None)
                            else None
                        )

                        for i, (b, c, cf) in enumerate(zip(b_arr, c_arr, conf_arr)):
                            box_tuple = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
                            tid = int(id_arr[i]) if id_arr is not None else None
                            eff_c = int(c)

                            # Aplica estabilização temporal de ruídos transitórios (Etapa 7.1)
                            if tid is not None and temporal_filter is not None:
                                eff_c = temporal_filter.update(track_id=tid, detected_class_id=eff_c)

                            boxes_xyxy.append(box_tuple)
                            class_ids.append(eff_c)
                            confidences.append(float(cf))
                            track_ids.append(tid)

                total_persons = len(class_ids)
                conformant_count = sum(1 for c in class_ids if c == 1)
                violation_count = sum(1 for c in class_ids if c == 0)
                compliance_rate = (
                    (conformant_count / total_persons * 100.0) if total_persons > 0 else 100.0
                )

                # Registro das detecções e recorte de evidências
                for det_id, (box, c, cf, tid) in enumerate(zip(boxes_xyxy, class_ids, confidences, track_ids), start=1):
                    crop_path_str = None
                    if save_crops and c == 0:
                        should_save_crop = True
                        # Desduplicação de evidências fotográficas (Opção B)
                        if enable_tracking and tid is not None and dedup_by_track:
                            if tid in saved_crop_track_ids:
                                should_save_crop = False

                        if should_save_crop:
                            crop = extract_roi(frame, box)
                            if crop.size > 0:
                                tid_tag = f"_track{tid}" if tid is not None else ""
                                crop_filename = f"violation{tid_tag}_frame{frame_idx}_det{det_id}_{frame_ts.strftime('%Y%m%d_%H%M%S_%f')[:19]}.jpg"
                                crop_full_path = violations_dir / crop_filename
                                cv2.imwrite(str(crop_full_path), crop)
                                crop_path_str = str(crop_full_path)
                                if tid is not None:
                                    saved_crop_track_ids.add(tid)

                    auditor.record_detection(
                        source=source_name,
                        frame_id=frame_idx,
                        detection_id=det_id,
                        class_id=c,
                        confidence=cf,
                        bbox_xyxy=box,
                        track_id=tid,
                        timestamp=frame_ts,
                        crop_path=crop_path_str,
                    )

                auditor.record_frame_summary(
                    source=source_name,
                    frame_id=frame_idx,
                    total_persons=total_persons,
                    conformant_count=conformant_count,
                    violation_count=violation_count,
                    timestamp=frame_ts,
                )

                # Renderização
                annotated_frame = frame.copy()
                if boxes_xyxy:
                    annotated_frame = draw_bounding_boxes(
                        image=annotated_frame,
                        boxes_xyxy=boxes_xyxy,
                        class_ids=class_ids,
                        confidences=confidences,
                        class_names=labels_map,
                        track_ids=track_ids if enable_tracking else None,
                        thickness=2
                    )

                if draw_banner:
                    annotated_frame = draw_telemetry_banner(
                        image=annotated_frame,
                        total_persons=total_persons,
                        conformant_count=conformant_count,
                        violation_count=violation_count,
                        compliance_rate=compliance_rate,
                        camera_id=camera_id,
                        timestamp_str=frame_ts.strftime("%Y-%m-%d %H:%M:%S")
                    )

                if video_writer is not None:
                    video_writer.write(annotated_frame)

                if show:
                    try:
                        cv2.imshow("EPI Finder - Monitoramento em Vídeo", annotated_frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == 27 or key == ord("q"):
                            break
                    except cv2.error:
                        pass

                if verbose and (frame_idx % 30 == 0 or frame_idx == total_frames_est):
                    elapsed = time.time() - start_proc_time
                    current_fps = frame_idx / elapsed if elapsed > 0 else 0
                    print(f"  Frame {frame_idx:04d} | {current_fps:.1f} FPS | Pessoas: {total_persons} | Conformes: {conformant_count} | Infrações: {violation_count} | Conformidade: {compliance_rate:.1f}%")

        finally:
            cap.release()
            if video_writer is not None:
                video_writer.release()
            if show:
                try:
                    cv2.destroyAllWindows()
                except cv2.error:
                    pass

    # EXPORTAÇÃO DOS RELATÓRIOS (CSV & JSON)
    if report_csv:
        auditor.export_reports(csv_path=report_csv, json_path=summary_json, verbose=verbose)

    # SUMÁRIO EXECUTIVO FINAL
    summary = auditor.get_audit_summary()
    if verbose:
        print("\n" + "=" * 78)
        print("📋 RESUMO EXECUTIVO DE AUDITORIA (SEGURANÇA DO TRABALHO)")
        print("=" * 78)
        print(f"🏢 Local / Câmera        : {summary['camera_id']}")
        print(f"🎞️  Quadros Processados    : {summary['total_frames_processed']}")
        print(f"🚨 Quadros com Infração   : {summary['frames_with_violations']}")
        print(f"📝 Ocorrências no CSV     : {summary['total_detections_logged']}")
        print(f"✅ Conformes Registrados  : {summary['conformant_helmets_logged']}")
        print(f"❌ Infrações Registradas  : {summary['violations_head_logged']}")
        if summary.get("tracking_enabled"):
            print(f"🔄 Rastreamento (MOT)     : Ativo ({tracker})")
            print(f"🧹 Desduplicação (Opção B): {'Ativa (1 registro/indivíduo)' if summary['dedup_by_track'] else 'Desativada'}")
            print(f"👥 Indivíduos Únicos (IDs): {summary['unique_persons_tracked']}")
            print(f"🛡️  Conformes Únicos       : {summary['unique_conformant']}")
            print(f"⚠️  Infratores Únicos      : {summary['unique_violators']}")
            print(f"📊 Conformidade Única     : {summary['unique_compliance_rate_percent']:.1f}%")
        else:
            print(f"📈 Taxa de Conformidade   : {summary['overall_compliance_rate_percent']:.1f}%")
        print(f"🎯 Confiança Média        : {summary['mean_confidence']:.2f}")
        print(f"📢 Status Operacional     : {summary['status']}")
        print("=" * 78)

    return auditor


def load_model(weights: Union[str, Path] = "models/best.pt") -> YOLO:
    """
    Carrega o modelo YOLOv8 a partir de arquivo de pesos local com fallback seguro.

    Args:
        weights: Caminho para os pesos .pt do modelo.

    Returns:
        Instância do modelo YOLO carregado.
    """
    weights_path = Path(weights).resolve()
    if not weights_path.exists():
        fallback_weights = PROJECT_ROOT / "yolov8n.pt"
        if fallback_weights.exists():
            weights_path = fallback_weights
        else:
            fallback_cwd = Path("yolov8n.pt").resolve()
            if fallback_cwd.exists():
                weights_path = fallback_cwd
            else:
                raise FileNotFoundError(f"Pesos de modelo não encontrados: {weights}")
    return YOLO(str(weights_path))


def infer_single_image(
    model: YOLO,
    image_bgr: np.ndarray,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    img_size: int = 640,
    device: Optional[str] = None,
    alert_labels: bool = True,
    draw_banner: bool = True,
    camera_id: str = "Camera-01",
) -> Dict[str, Any]:
    """
    Executa inferência rápida em uma única imagem (formato BGR).

    Retorna a imagem anotada com bounding boxes estilizadas, painel de telemetria,
    contagem de detecções e lista de recortes das infrações para apresentação
    imediata em interfaces interativas (Streamlit).

    Args:
        model: Instância de ultralytics.YOLO.
        image_bgr: Array NumPy da imagem em formato BGR (OpenCV).
        conf_threshold: Limiar de confiança para detecção.
        iou_threshold: Limiar de IoU (NMS).
        img_size: Tamanho de entrada para inferência da rede.
        device: Dispositivo de execução ('0' ou 'cpu').
        alert_labels: Se True, utiliza descrições focadas em SST (ALERTA: SEM CAPACETE).
        draw_banner: Se True, sobrepõe a barra de telemetria superior.
        camera_id: Identificador do local/câmera.

    Returns:
        Dicionário com imagem anotada (BGR e RGB), caixas, contagens e recortes.
    """
    active_device = device if device is not None else get_default_device()
    labels_map = ALERT_CLASSES if alert_labels else DEFAULT_CLASSES
    now_ts = datetime.now()

    results = model.predict(
        source=image_bgr,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=img_size,
        device=active_device,
        verbose=False,
    )

    boxes_xyxy: List[Tuple[int, int, int, int]] = []
    class_ids: List[int] = []
    confidences: List[float] = []
    detections: List[Dict[str, Any]] = []
    violation_crops: List[Dict[str, Any]] = []

    for r in results:
        if r.boxes is not None and len(r.boxes) > 0:
            b_arr = r.boxes.xyxy.cpu().numpy()
            c_arr = r.boxes.cls.cpu().numpy().astype(int)
            conf_arr = r.boxes.conf.cpu().numpy()

            for idx, (b, c, cf) in enumerate(zip(b_arr, c_arr, conf_arr), start=1):
                box_tuple = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
                c_int = int(c)
                cf_float = float(cf)

                boxes_xyxy.append(box_tuple)
                class_ids.append(c_int)
                confidences.append(cf_float)

                class_name = DEFAULT_CLASSES.get(c_int, f"class_{c_int}")
                is_violation = (c_int == 0)

                det_dict = {
                    "detection_id": idx,
                    "class_id": c_int,
                    "class_name": class_name,
                    "confidence": round(cf_float, 4),
                    "bbox_xyxy": box_tuple,
                    "is_violation": is_violation,
                }
                detections.append(det_dict)

                if is_violation:
                    crop_bgr = extract_roi(image_bgr, box_tuple)
                    if crop_bgr.size > 0:
                        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                        violation_crops.append({
                            "detection_id": idx,
                            "crop_bgr": crop_bgr,
                            "crop_rgb": crop_rgb,
                            "confidence": round(cf_float, 4),
                            "bbox_xyxy": box_tuple,
                        })

    total_persons = len(class_ids)
    conformant_count = sum(1 for c in class_ids if c == 1)
    violation_count = sum(1 for c in class_ids if c == 0)
    compliance_rate = (
        (conformant_count / total_persons * 100.0) if total_persons > 0 else 100.0
    )

    annotated_bgr = image_bgr.copy()
    if boxes_xyxy:
        annotated_bgr = draw_bounding_boxes(
            image=annotated_bgr,
            boxes_xyxy=boxes_xyxy,
            class_ids=class_ids,
            confidences=confidences,
            class_names=labels_map,
            track_ids=None,
            thickness=2,
        )

    if draw_banner:
        annotated_bgr = draw_telemetry_banner(
            image=annotated_bgr,
            total_persons=total_persons,
            conformant_count=conformant_count,
            violation_count=violation_count,
            compliance_rate=compliance_rate,
            camera_id=camera_id,
            timestamp_str=now_ts.strftime("%Y-%m-%d %H:%M:%S"),
        )

    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    return {
        "annotated_bgr": annotated_bgr,
        "annotated_rgb": annotated_rgb,
        "detections": detections,
        "boxes_xyxy": boxes_xyxy,
        "class_ids": class_ids,
        "confidences": confidences,
        "total_persons": total_persons,
        "conformant_count": conformant_count,
        "violation_count": violation_count,
        "compliance_rate_percent": round(compliance_rate, 1),
        "violation_crops": violation_crops,
    }


def process_video_stream(
    video_source: Union[str, Path, int],
    model: YOLO,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    img_size: int = 640,
    device: Optional[str] = None,
    camera_id: str = "Camera-01",
    enable_tracking: bool = True,
    tracker: str = "bytetrack.yaml",
    min_consecutive_frames: int = 3,
    dedup_by_track: bool = True,
    alert_labels: bool = True,
    draw_banner: bool = True,
) -> Generator[Dict[str, Any], None, ComplianceAuditor]:
    """
    Gerador (stream) que processa um vídeo ou stream quadro a quadro,
    emitindo telemetria e o frame anotado para exibição suave no Streamlit.

    Yields:
        Dicionário com o frame anotado, progresso e auditor atualizado.

    Returns:
        Instância de ComplianceAuditor com todos os registros acumulados.
    """
    active_device = device if device is not None else get_default_device()
    labels_map = ALERT_CLASSES if alert_labels else DEFAULT_CLASSES

    cap = cv2.VideoCapture(video_source if isinstance(video_source, int) else str(video_source))
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o fluxo de vídeo: {video_source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_name = Path(video_source).name if isinstance(video_source, (str, Path)) else f"Cam_{video_source}"

    auditor = ComplianceAuditor(
        camera_id=camera_id,
        dedup_by_track=(dedup_by_track and enable_tracking),
    )

    temporal_filter = (
        TemporalTrackerFilter(min_consecutive_frames=min_consecutive_frames)
        if enable_tracking
        else None
    )

    saved_crop_track_ids: Set[int] = set()
    frame_idx = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_idx += 1
            frame_ts = datetime.now()
            new_violation_crops: List[Dict[str, Any]] = []

            if enable_tracking:
                results = model.track(
                    source=frame,
                    persist=True,
                    tracker=tracker,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    imgsz=img_size,
                    device=active_device,
                    verbose=False,
                )
            else:
                results = model.predict(
                    source=frame,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    imgsz=img_size,
                    device=active_device,
                    verbose=False,
                )

            boxes_xyxy: List[Tuple[int, int, int, int]] = []
            class_ids: List[int] = []
            confidences: List[float] = []
            track_ids: Optional[List[int]] = [] if enable_tracking else None

            for r in results:
                if r.boxes is not None and len(r.boxes) > 0:
                    b_arr = r.boxes.xyxy.cpu().numpy()
                    c_arr = r.boxes.cls.cpu().numpy().astype(int)
                    conf_arr = r.boxes.conf.cpu().numpy()

                    t_arr = None
                    if enable_tracking and r.boxes.id is not None:
                        t_arr = r.boxes.id.cpu().numpy().astype(int)

                    for idx_box, (b, c, cf) in enumerate(zip(b_arr, c_arr, conf_arr)):
                        box_tuple = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
                        detected_class_id = int(c)
                        conf_val = float(cf)

                        tid = None
                        effective_class_id = detected_class_id

                        if t_arr is not None and idx_box < len(t_arr):
                            tid = int(t_arr[idx_box])
                            if temporal_filter is not None:
                                effective_class_id = temporal_filter.update(tid, detected_class_id)

                        boxes_xyxy.append(box_tuple)
                        class_ids.append(effective_class_id)
                        confidences.append(conf_val)
                        if track_ids is not None:
                            track_ids.append(tid if tid is not None else -1)

            total_persons = len(class_ids)
            conformant_count = sum(1 for c in class_ids if c == 1)
            violation_count = sum(1 for c in class_ids if c == 0)
            compliance_rate = (
                (conformant_count / total_persons * 100.0) if total_persons > 0 else 100.0
            )

            # Grava ocorrências no auditor
            for det_idx, (box, c, cf) in enumerate(zip(boxes_xyxy, class_ids, confidences), start=1):
                tid = track_ids[det_idx - 1] if track_ids else None
                if c == 0:
                    should_crop = False
                    if enable_tracking and dedup_by_track and tid is not None and tid != -1:
                        if tid not in saved_crop_track_ids:
                            saved_crop_track_ids.add(tid)
                            should_crop = True
                    elif not enable_tracking:
                        should_crop = True

                    if should_crop:
                        crop_bgr = extract_roi(frame, box)
                        if crop_bgr.size > 0:
                            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                            new_violation_crops.append({
                                "track_id": tid,
                                "frame_idx": frame_idx,
                                "crop_bgr": crop_bgr,
                                "crop_rgb": crop_rgb,
                                "confidence": round(cf, 4),
                                "bbox_xyxy": box,
                            })

                auditor.record_detection(
                    source=source_name,
                    frame_id=frame_idx,
                    detection_id=det_idx,
                    class_id=c,
                    confidence=cf,
                    bbox_xyxy=box,
                    track_id=tid,
                    timestamp=frame_ts,
                )

            auditor.record_frame_summary(
                source=source_name,
                frame_id=frame_idx,
                total_persons=total_persons,
                conformant_count=conformant_count,
                violation_count=violation_count,
                timestamp=frame_ts,
            )

            annotated_frame = frame.copy()
            if boxes_xyxy:
                annotated_frame = draw_bounding_boxes(
                    image=annotated_frame,
                    boxes_xyxy=boxes_xyxy,
                    class_ids=class_ids,
                    confidences=confidences,
                    class_names=labels_map,
                    track_ids=track_ids if enable_tracking else None,
                    thickness=2,
                )

            if draw_banner:
                annotated_frame = draw_telemetry_banner(
                    image=annotated_frame,
                    total_persons=total_persons,
                    conformant_count=conformant_count,
                    violation_count=violation_count,
                    compliance_rate=compliance_rate,
                    camera_id=camera_id,
                    timestamp_str=frame_ts.strftime("%Y-%m-%d %H:%M:%S"),
                )

            annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            progress_percent = (frame_idx / total_frames * 100.0) if total_frames > 0 else 0.0

            yield {
                "frame_idx": frame_idx,
                "total_frames": total_frames,
                "progress_percent": min(100.0, progress_percent),
                "annotated_rgb": annotated_rgb,
                "annotated_bgr": annotated_frame,
                "total_persons": total_persons,
                "conformant_count": conformant_count,
                "violation_count": violation_count,
                "compliance_rate_percent": round(compliance_rate, 1),
                "auditor": auditor,
                "new_violation_crops": new_violation_crops,
            }

    finally:
        cap.release()

    return auditor


def parse_args() -> argparse.Namespace:
    """
    Interface de Linha de Comando (CLI) para execução do módulo de inferência.
    """
    parser = argparse.ArgumentParser(
        description="EPI Finder - Módulo de Inferência Operacional e Auditoria de SST (Fase 7.1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Caminho para imagem, diretório com imagens, arquivo de vídeo ou índice de webcam (ex: 0).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="models/best.pt",
        help="Caminho para o checkpoint de pesos (.pt) treinado.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Limiar mínimo de confiança para aceitar uma detecção [0.0 - 1.0].",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="Limiar de IoU para o algoritmo de Non-Maximum Suppression (NMS).",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Dimensão quadrada de entrada do modelo YOLOv8.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Dispositivo de execução ('0', '1' para GPU CUDA ou 'cpu'). Padrão detecta automático.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/inference",
        help="Diretório onde serão salvos imagens/vídeos anotados e relatórios.",
    )
    parser.add_argument(
        "--report-csv",
        type=str,
        default="runs/inference/compliance_report.csv",
        help="Caminho para salvar o relatório de conformidade em CSV.",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default="runs/inference/summary.json",
        help="Caminho para salvar o resumo estruturado em JSON.",
    )
    parser.add_argument(
        "--camera-id",
        type=str,
        default="Camera-01",
        help="Nome ou identificador da câmera/posto de trabalho para o relatório.",
    )
    parser.add_argument(
        "--save-crops",
        action="store_true",
        help="Se ativado, salva recortes (ROIs) das infrações de sem capacete na pasta 'violations'.",
    )
    parser.add_argument(
        "--no-media",
        action="store_true",
        help="Se ativado, não salva as imagens ou vídeos anotados em disco (apenas gera relatórios).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Exibe janela gráfica OpenCV com a transmissão em tempo real.",
    )
    parser.add_argument(
        "--standard-labels",
        action="store_true",
        help="Usa rótulos padrão ('head'/'helmet') em vez dos alertas operacionais ('ALERTA: SEM CAPACETE').",
    )
    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="Desativa o painel superior de telemetria nos frames.",
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help="Habilita Multi-Object Tracking (MOT - Etapa 7.1) no processamento de vídeo/webcam.",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack.yaml",
        help="Algoritmo de rastreamento nativo do YOLOv8 (ex: bytetrack.yaml, botsort.yaml).",
    )
    parser.add_argument(
        "--min-consecutive-frames",
        type=int,
        default=3,
        help="Número mínimo de frames consecutivos para confirmação temporal de infração (debounce).",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Desativa a desduplicação da Opção B (registra todas as ocorrências a cada frame no CSV).",
    )

    return parser.parse_args()


def main() -> None:
    """Ponto de entrada para execução via terminal."""
    args = parse_args()

    try:
        run_inference(
            source=args.source,
            weights=args.weights,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            img_size=args.img_size,
            device=args.device,
            output_dir=args.output_dir,
            report_csv=args.report_csv,
            summary_json=args.summary_json,
            camera_id=args.camera_id,
            save_crops=args.save_crops,
            save_media=not args.no_media,
            show=args.show,
            alert_labels=not args.standard_labels,
            draw_banner=not args.no_banner,
            enable_tracking=args.track,
            tracker=args.tracker,
            min_consecutive_frames=args.min_consecutive_frames,
            dedup_by_track=not args.no_dedup,
            verbose=True,
        )
    except Exception as exc:
        print(f"\n❌ Erro durante a execução da inferência: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
