"""
Testes unitários para o módulo src/analytics.py (EPI Finder - Fase 7.2).
"""

from pathlib import Path
import pandas as pd
import pytest
import plotly.graph_objects as go

from src.analytics import (
    calculate_kpis,
    create_camera_ranking_chart,
    create_compliance_donut_chart,
    create_confidence_distribution_chart,
    create_temporal_series_chart,
    load_compliance_report,
    load_frame_summary,
)
from src.inference import ComplianceAuditor


def test_calculate_kpis_empty():
    """Valida cálculo de KPIs com dados ausentes/vazios."""
    kpis = calculate_kpis(None, None, None)
    assert kpis["total_detections"] == 0
    assert kpis["total_frames"] == 0
    assert kpis["helmet_count"] == 0
    assert kpis["head_count"] == 0
    assert kpis["compliance_rate_percent"] == 100.0
    assert kpis["status"] == "Conforme (Excelente)"


def test_calculate_kpis_standard_detections():
    """Valida cálculo de KPIs com detecções sem rastreamento."""
    data = {
        "timestamp": ["2026-08-30 10:00:00", "2026-08-30 10:00:01", "2026-08-30 10:00:02"],
        "frame_idx": [1, 2, 3],
        "class_name": ["helmet", "helmet", "head"],
        "confidence": [0.95, 0.88, 0.76],
        "camera_id": ["Cam-A", "Cam-A", "Cam-B"],
    }
    df = pd.DataFrame(data)
    kpis = calculate_kpis(df_detections=df)

    assert kpis["total_detections"] == 3
    assert kpis["helmet_count"] == 2
    assert kpis["head_count"] == 1
    # 2/3 = 66.7%
    assert 66.0 <= kpis["compliance_rate_percent"] <= 67.0
    assert kpis["status"] == "Crítico (Risco de Acidentes)"


def test_calculate_kpis_with_tracking():
    """Valida cálculo de KPIs com rastreamento (dedup por track_id)."""
    # Sujeito 1 (helmet em todos os frames)
    # Sujeito 2 (head no frame 2)
    # Sujeito 3 (helmet no frame 3)
    data = {
        "timestamp": ["2026-08-30 10:00:00"] * 5,
        "frame_idx": [1, 2, 3, 4, 5],
        "track_id": [1, 1, 2, 2, 3],
        "class_name": ["helmet", "helmet", "head", "head", "helmet"],
        "confidence": [0.9, 0.9, 0.8, 0.8, 0.95],
        "camera_id": ["Camera-01"] * 5,
    }
    df = pd.DataFrame(data)
    kpis = calculate_kpis(df_detections=df)

    assert kpis["has_tracking"] is True
    assert kpis["unique_persons"] == 3
    assert kpis["unique_violators"] == 1  # Track 2
    assert kpis["unique_conformant"] == 2 # Tracks 1 e 3
    # 2 / 3 = 66.7%
    assert 66.0 <= kpis["unique_compliance_rate_percent"] <= 67.0


def test_donut_chart_generation():
    """Valida geração do gráfico de rosca de conformidade."""
    # Cenário com dados
    fig = create_compliance_donut_chart(conformant_count=8, violation_count=2)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].hole == 0.68

    # Cenário vazio
    fig_empty = create_compliance_donut_chart(conformant_count=0, violation_count=0)
    assert isinstance(fig_empty, go.Figure)


def test_temporal_series_chart():
    """Valida gráfico de série temporal."""
    df_frames = pd.DataFrame({
        "frame_idx": [1, 2, 3],
        "helmet_count": [5, 4, 6],
        "head_count": [0, 1, 0],
        "compliance_rate_percent": [100.0, 80.0, 100.0],
    })
    fig = create_temporal_series_chart(df_frames)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3

    # Vazio
    fig_empty = create_temporal_series_chart(pd.DataFrame())
    assert isinstance(fig_empty, go.Figure)


def test_temporal_series_chart_with_auditor_columns():
    """O frame_summary real do ComplianceAuditor deve gerar as 3 séries."""
    auditor = ComplianceAuditor(camera_id="Cam-Test")
    auditor.record_frame_summary("test.mp4", 1, total_persons=2, conformant_count=2, violation_count=0)
    auditor.record_frame_summary("test.mp4", 2, total_persons=2, conformant_count=1, violation_count=1)

    fig = create_temporal_series_chart(auditor.to_frame_summary_dataframe())
    assert len(fig.data) == 3
    assert list(fig.data[0].x) == [1, 2]
    assert list(fig.data[1].y) == [0, 1]


def test_load_frame_summary_normalizes_auditor_csv(tmp_path: Path):
    """O frame_summary.csv exportado pelo app é lido com as colunas dos gráficos."""
    csv_file = tmp_path / "frame_summary.csv"
    csv_file.write_text(
        "frame_id,total_persons,conformant_helmets,violations_head,compliance_rate\n"
        "1,2,2,0,100.0\n"
    )
    df = load_frame_summary(csv_file)
    for col in ["frame_idx", "helmet_count", "head_count", "compliance_rate_percent"]:
        assert col in df.columns


def test_calculate_kpis_uses_real_audit_summary_keys():
    """KPIs vêm do sumário real do auditor e não são sobrescritos pelo CSV desduplicado."""
    auditor = ComplianceAuditor(camera_id="Cam-Test", dedup_by_track=True)
    box = (10, 10, 50, 50)
    # Track 1: começa de capacete e depois infringe (o CSV desduplicado só guarda o 'helmet')
    auditor.record_detection("v.mp4", 1, 1, 1, 0.9, box, track_id=1)
    auditor.record_detection("v.mp4", 2, 1, 0, 0.8, box, track_id=1)
    auditor.record_detection("v.mp4", 2, 2, 1, 0.9, box, track_id=2)
    auditor.record_frame_summary("v.mp4", 1, 1, 1, 0)
    auditor.record_frame_summary("v.mp4", 2, 2, 1, 1)

    summary = auditor.get_audit_summary()
    kpis = calculate_kpis(
        df_detections=auditor.to_dataframe(),
        audit_summary=summary,
    )
    assert kpis["total_frames"] == 2
    assert kpis["has_tracking"] is True
    assert kpis["unique_persons"] == 2
    assert kpis["unique_violators"] == 1
    assert kpis["unique_conformant"] == 1
    assert kpis["unique_compliance_rate_percent"] == 50.0

    # Só com o sumário (sem CSV), as contagens por detecção também vêm das chaves reais
    kpis_summary_only = calculate_kpis(audit_summary=summary)
    assert kpis_summary_only["total_detections"] == summary["total_detections_logged"]
    assert kpis_summary_only["helmet_count"] == summary["conformant_helmets_logged"]
    assert kpis_summary_only["head_count"] == summary["violations_head_logged"]


def test_confidence_distribution_chart():
    """Valida gráfico de distribuição de confiança."""
    df = pd.DataFrame({
        "confidence": [0.85, 0.92, 0.45, 0.78],
        "class_name": ["helmet", "helmet", "head", "head"],
    })
    fig = create_confidence_distribution_chart(df)
    assert isinstance(fig, go.Figure)

    # Vazio
    fig_empty = create_confidence_distribution_chart(pd.DataFrame())
    assert isinstance(fig_empty, go.Figure)


def test_camera_ranking_chart():
    """Valida gráfico de ranking por câmera."""
    df = pd.DataFrame({
        "camera_id": ["Portaria", "Portaria", "Galpão", "Galpão"],
        "class_name": ["helmet", "head", "helmet", "helmet"],
    })
    fig = create_camera_ranking_chart(df)
    assert isinstance(fig, go.Figure)

    # Vazio
    fig_empty = create_camera_ranking_chart(pd.DataFrame())
    assert isinstance(fig_empty, go.Figure)


def test_load_reports(tmp_path: Path):
    """Valida carga e normalização de relatórios CSV."""
    csv_file = tmp_path / "compliance.csv"
    csv_file.write_text("frame_idx,class_name,confidence\n1,head,0.85\n2,helmet,0.92\n")

    df = load_compliance_report(csv_file)
    assert len(df) == 2
    assert "is_violation" in df.columns
    assert df.loc[0, "is_violation"] == True
    assert df.loc[1, "is_violation"] == False

    # Arquivo inexistente
    df_missing = load_compliance_report(tmp_path / "nao_existe.csv")
    assert df_missing.empty
