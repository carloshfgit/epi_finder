"""
Testes unitários para a Etapa 7.1 do EPI Finder:
- Multi-Object Tracking (MOT)
- Desduplicação de Logs no ComplianceAuditor (Opção B)
- Filtro de Estabilização Temporal (TemporalTrackerFilter)
- Renderização visual com Track IDs
"""

import os
from pathlib import Path
import sys
import unittest
from datetime import datetime
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import ComplianceAuditor, TemporalTrackerFilter
from src.utils import draw_bounding_boxes


class TestTemporalTrackerFilter(unittest.TestCase):
    """Valida o comportamento do filtro de suavização temporal."""

    def test_smoothing_transient_noise(self):
        """Um ruído de 1 frame de 'head' não deve desestabilizar quem está de 'helmet'."""
        f = TemporalTrackerFilter(min_consecutive_frames=3)
        track_id = 101

        # Trabalhador começa com capacete (1) por 3 frames -> estabiliza em 1
        self.assertEqual(f.update(track_id, 1), 1)
        self.assertEqual(f.update(track_id, 1), 1)
        self.assertEqual(f.update(track_id, 1), 1)
        self.assertFalse(f.is_violation_confirmed(track_id))

        # Ocorre um ruído de 1 quadro (head = 0) devido a sombra ou rotação brusca
        stabilized_val = f.update(track_id, 0)
        # Deve permanecer 1 (helmet) porque não atingiu 3 quadros consecutivos de 0
        self.assertEqual(stabilized_val, 1)
        self.assertFalse(f.is_violation_confirmed(track_id))

        # No próximo quadro volta para helmet (1)
        self.assertEqual(f.update(track_id, 1), 1)

    def test_confirmation_after_consecutive_frames(self):
        """Após N quadros consecutivos, a nova classe deve ser devidamente confirmada."""
        f = TemporalTrackerFilter(min_consecutive_frames=3)
        track_id = 202

        # Inicia com capacete
        for _ in range(3):
            f.update(track_id, 1)
        self.assertEqual(f.stabilized_classes[track_id], 1)

        # Agora o trabalhador tira o capacete (classe 0 por 3 quadros seguidos)
        f.update(track_id, 0)  # frame 1 de infração -> ainda estabilizado em 1
        self.assertEqual(f.stabilized_classes[track_id], 1)
        self.assertFalse(f.is_violation_confirmed(track_id))

        f.update(track_id, 0)  # frame 2 de infração -> ainda estabilizado em 1
        self.assertEqual(f.stabilized_classes[track_id], 1)
        self.assertFalse(f.is_violation_confirmed(track_id))

        f.update(track_id, 0)  # frame 3 consecutivo -> transição confirmada!
        self.assertEqual(f.stabilized_classes[track_id], 0)
        self.assertTrue(f.is_violation_confirmed(track_id))


class TestComplianceAuditorDedup(unittest.TestCase):
    """Valida a desduplicação de logs (Opção B) no ComplianceAuditor."""

    def test_dedup_enabled_option_b(self):
        """Com dedup_by_track=True (Opção B), grava apenas o 1º evento de cada track_id no CSV."""
        auditor = ComplianceAuditor(camera_id="Cam-Test", dedup_by_track=True)
        box = (10, 10, 50, 50)

        # Track 1 detectado no frame 1 (salvo)
        saved_1 = auditor.record_detection("test.mp4", 1, 1, 0, 0.9, box, track_id=1)
        self.assertTrue(saved_1)

        # Track 1 detectado no frame 2 (ignorado pelo dedup)
        saved_2 = auditor.record_detection("test.mp4", 2, 1, 0, 0.92, box, track_id=1)
        self.assertFalse(saved_2)

        # Track 1 detectado no frame 3 (ignorado pelo dedup)
        saved_3 = auditor.record_detection("test.mp4", 3, 1, 0, 0.89, box, track_id=1)
        self.assertFalse(saved_3)

        # Track 2 detectado no frame 3 (novo indivíduo -> salvo)
        saved_4 = auditor.record_detection("test.mp4", 3, 2, 1, 0.95, box, track_id=2)
        self.assertTrue(saved_4)

        df = auditor.to_dataframe()
        # O DataFrame deve conter exatamente 2 linhas (1 para cada track_id)
        self.assertEqual(len(df), 2)
        self.assertListEqual(list(df["track_id"]), [1, 2])

        # Métricas do sumário executivo
        summary = auditor.get_audit_summary()
        self.assertEqual(summary["unique_persons_tracked"], 2)
        self.assertEqual(summary["unique_violators"], 1)
        self.assertEqual(summary["unique_conformant"], 1)
        self.assertEqual(summary["unique_compliance_rate_percent"], 50.0)

    def test_dedup_disabled(self):
        """Com dedup_by_track=False, grava todos os frames normalmente."""
        auditor = ComplianceAuditor(camera_id="Cam-Test", dedup_by_track=False)
        box = (10, 10, 50, 50)

        auditor.record_detection("test.mp4", 1, 1, 1, 0.9, box, track_id=5)
        auditor.record_detection("test.mp4", 2, 1, 1, 0.91, box, track_id=5)
        auditor.record_detection("test.mp4", 3, 1, 1, 0.92, box, track_id=5)

        df = auditor.to_dataframe()
        self.assertEqual(len(df), 3)
        summary = auditor.get_audit_summary()
        self.assertEqual(summary["unique_persons_tracked"], 1)
        self.assertEqual(summary["unique_conformant"], 1)


class TestVisualDrawingWithTrackIDs(unittest.TestCase):
    """Valida se as anotações visuais incorporam o Track ID."""

    def test_draw_bounding_boxes_with_track_ids(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        boxes = [(10, 10, 50, 50), (60, 60, 90, 90)]
        classes = [0, 1]
        confs = [0.85, 0.95]
        tids = [7, 8]

        annotated = draw_bounding_boxes(
            image=img,
            boxes_xyxy=boxes,
            class_ids=classes,
            confidences=confs,
            track_ids=tids
        )
        self.assertEqual(annotated.shape, img.shape)
        # Verifica que a imagem anotada foi modificada (não é toda preta)
        self.assertTrue(np.any(annotated > 0))


if __name__ == "__main__":
    unittest.main()
