"""
Módulo de inteligência analítica e visualização de dados de SST (EPI Finder).

Responsável pelo processamento estatístico de relatórios de conformidade e
geração de gráficos interativos com Plotly para subsidiar diagnósticos da CIPA
e auditorias de Segurança e Saúde no Trabalho.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# Cores temáticas para SST (Harmoniosas com interfaces modernas e temas escuro/claro)
COLOR_CONFORMANT = "#10b981"  # Verde esmeralda (Conforme / Helmet)
COLOR_VIOLATION = "#ef4444"   # Vermelho vibrante (Infração / Head)
COLOR_BACKGROUND = "rgba(0,0,0,0)"
COLOR_TEXT = "#e2e8f0"


def load_compliance_report(source: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    """
    Carrega e normaliza o relatório de conformidade de detecções/infrações.

    Args:
        source: Caminho para o arquivo CSV ou DataFrame existente.

    Returns:
        DataFrame normalizado com as colunas esperadas.
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        path = Path(source)
        if not path.exists():
            return pd.DataFrame()
        try:
            df = pd.read_csv(path)
        except Exception:
            return pd.DataFrame()

    expected_cols = [
        "timestamp",
        "frame_idx",
        "track_id",
        "class_id",
        "class_name",
        "confidence",
        "camera_id",
        "is_violation",
    ]
    for col in expected_cols:
        if col not in df.columns:
            if col == "is_violation" and "class_name" in df.columns:
                df["is_violation"] = df["class_name"].astype(str).str.lower() == "head"
            elif col == "confidence":
                df["confidence"] = 1.0
            elif col == "camera_id":
                df["camera_id"] = "Camera-01"

    return df


def load_frame_summary(source: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    """
    Carrega e normaliza o resumo frame a frame da telemetria de vídeo.

    Args:
        source: Caminho para o CSV de frame_summary ou DataFrame existente.

    Returns:
        DataFrame com histórico temporal de conformidade.
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        path = Path(source)
        if not path.exists():
            return pd.DataFrame()
        try:
            df = pd.read_csv(path)
        except Exception:
            return pd.DataFrame()

    return df


def calculate_kpis(
    df_detections: Optional[pd.DataFrame] = None,
    df_frames: Optional[pd.DataFrame] = None,
    audit_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Calcula indicadores-chave de desempenho (KPIs) de conformidade e SST.

    Args:
        df_detections: DataFrame com registros individuais de detecção.
        df_frames: DataFrame com resumo consolidado por frame.
        audit_summary: Dicionário opcional pré-calculado pelo ComplianceAuditor.

    Returns:
        Dicionário com métricas agregadas formatadas para exibição em dashboard.
    """
    # Valores padrão iniciais
    kpis: Dict[str, Any] = {
        "total_detections": 0,
        "total_frames": 0,
        "helmet_count": 0,
        "head_count": 0,
        "unique_persons": 0,
        "unique_conformant": 0,
        "unique_violators": 0,
        "compliance_rate_percent": 100.0,
        "unique_compliance_rate_percent": 100.0,
        "has_tracking": False,
        "status": "Excelente",
        "status_color": COLOR_CONFORMANT,
    }

    # Se já tivermos o sumário consolidado do ComplianceAuditor, usa como base
    if audit_summary:
        kpis["total_detections"] = int(audit_summary.get("total_detections", 0))
        kpis["total_frames"] = int(audit_summary.get("total_frames_processed", 0))
        counts = audit_summary.get("class_counts", {})
        kpis["head_count"] = int(counts.get("head", 0))
        kpis["helmet_count"] = int(counts.get("helmet", 0))
        kpis["compliance_rate_percent"] = float(
            audit_summary.get("compliance_rate_percent", 100.0)
        )
        if audit_summary.get("tracking_enabled", False):
            kpis["has_tracking"] = True
            kpis["unique_persons"] = int(audit_summary.get("total_unique_tracks", 0))
            kpis["unique_conformant"] = int(audit_summary.get("unique_conformant", 0))
            kpis["unique_violators"] = int(audit_summary.get("unique_violators", 0))
            kpis["unique_compliance_rate_percent"] = float(
                audit_summary.get("unique_compliance_rate_percent", 100.0)
            )

    # Se fornecido DataFrame de detecções, calcula/refina métricas
    if df_detections is not None and not df_detections.empty:
        kpis["total_detections"] = len(df_detections)
        if "class_name" in df_detections.columns:
            classes = df_detections["class_name"].astype(str).str.lower()
            kpis["head_count"] = int((classes == "head").sum())
            kpis["helmet_count"] = int((classes == "helmet").sum())

        total = kpis["head_count"] + kpis["helmet_count"]
        if total > 0:
            kpis["compliance_rate_percent"] = round((kpis["helmet_count"] / total) * 100, 1)

        if "track_id" in df_detections.columns:
            valid_tracks = df_detections["track_id"].dropna()
            # Filtra IDs válidos (não nulos e diferentes de -1)
            valid_tracks = valid_tracks[valid_tracks != -1]
            if not valid_tracks.empty:
                kpis["has_tracking"] = True
                unique_ids = df_detections[df_detections["track_id"] != -1].groupby("track_id")
                violator_ids = set()
                conformant_ids = set()

                for tid, group in unique_ids:
                    has_violation = (group["class_name"].astype(str).str.lower() == "head").any()
                    if has_violation:
                        violator_ids.add(tid)
                    else:
                        conformant_ids.add(tid)

                kpis["unique_persons"] = len(unique_ids)
                kpis["unique_violators"] = len(violator_ids)
                kpis["unique_conformant"] = len(conformant_ids)
                if kpis["unique_persons"] > 0:
                    kpis["unique_compliance_rate_percent"] = round(
                        (kpis["unique_conformant"] / kpis["unique_persons"]) * 100, 1
                    )

    if df_frames is not None and not df_frames.empty:
        kpis["total_frames"] = len(df_frames)

    # Define status qualitativo de SST
    effective_rate = (
        kpis["unique_compliance_rate_percent"]
        if kpis["has_tracking"]
        else kpis["compliance_rate_percent"]
    )
    if effective_rate >= 90.0:
        kpis["status"] = "Conforme (Excelente)"
        kpis["status_color"] = COLOR_CONFORMANT
    elif effective_rate >= 75.0:
        kpis["status"] = "Atenção (Alerta SST)"
        kpis["status_color"] = "#f59e0b"  # Âmbar
    else:
        kpis["status"] = "Crítico (Risco de Acidentes)"
        kpis["status_color"] = COLOR_VIOLATION

    return kpis


def _empty_figure(message: str = "Nenhum dado disponível para visualização.") -> go.Figure:
    """Cria uma figura Plotly vazia amigável informando ausência de dados."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=14, color="#94a3b8"),
    )
    fig.update_layout(
        paper_bgcolor=COLOR_BACKGROUND,
        plot_bgcolor=COLOR_BACKGROUND,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=30, b=20),
        height=300,
    )
    return fig


def create_compliance_donut_chart(
    conformant_count: int,
    violation_count: int,
    title: str = "Taxa Geral de Conformidade",
) -> go.Figure:
    """
    Renderiza gráfico de rosca (Donut Chart) com a proporção de uso de EPI.

    Args:
        conformant_count: Quantidade de pessoas/detecções com capacete.
        violation_count: Quantidade de pessoas/detecções sem capacete.
        title: Título do gráfico.

    Returns:
        Figura do Plotly pronta para st.plotly_chart.
    """
    total = conformant_count + violation_count
    if total == 0:
        return _empty_figure("Aguardando detecções para compor a rosca de conformidade.")

    compliance_rate = (conformant_count / total) * 100

    labels = ["Com Capacete (Conforme)", "Sem Capacete (Infração)"]
    values = [conformant_count, violation_count]
    colors = [COLOR_CONFORMANT, COLOR_VIOLATION]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.68,
                marker=dict(colors=colors, line=dict(color="#1e293b", width=2)),
                textinfo="percent+value",
                textposition="inside",
                hoverinfo="label+value+percent",
                sort=False,
            )
        ]
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            x=0.5,
            xanchor="center",
            font=dict(size=16, color="#f8fafc"),
        ),
        paper_bgcolor=COLOR_BACKGROUND,
        plot_bgcolor=COLOR_BACKGROUND,
        annotations=[
            dict(
                text=f"<b>{compliance_rate:.1f}%</b><br><span style='font-size:11px;color:#94a3b8;'>Conformidade</span>",
                x=0.5,
                y=0.5,
                font=dict(size=20, color="#f8fafc"),
                showarrow=False,
            )
        ],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(color="#cbd5e1"),
        ),
        margin=dict(l=15, r=15, t=40, b=30),
        height=320,
    )
    return fig


def create_temporal_series_chart(
    df_frames: pd.DataFrame,
    title: str = "Evolução Temporal da Conformidade",
) -> go.Figure:
    """
    Renderiza gráfico de série temporal com a evolução de pessoas e conformidade.

    Args:
        df_frames: DataFrame extraído de frame_summary.csv.
        title: Título do gráfico.

    Returns:
        Figura do Plotly pronta para renderização.
    """
    if df_frames is None or df_frames.empty:
        return _empty_figure("Nenhum dado temporal registrado.")

    fig = go.Figure()

    # Eixo X pode ser frame_idx ou timestamp
    x_axis = df_frames["frame_idx"] if "frame_idx" in df_frames.columns else df_frames.index

    if "helmet_count" in df_frames.columns:
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=df_frames["helmet_count"],
                mode="lines",
                name="Capacete (Conforme)",
                line=dict(color=COLOR_CONFORMANT, width=2.5),
                fill="tozeroy",
                fillcolor="rgba(16, 185, 129, 0.15)",
            )
        )

    if "head_count" in df_frames.columns:
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=df_frames["head_count"],
                mode="lines",
                name="Sem Capacete (Infração)",
                line=dict(color=COLOR_VIOLATION, width=2.5),
                fill="tozeroy",
                fillcolor="rgba(239, 68, 68, 0.15)",
            )
        )

    if "compliance_rate_percent" in df_frames.columns:
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=df_frames["compliance_rate_percent"],
                mode="lines",
                name="Taxa de Conformidade (%)",
                line=dict(color="#38bdf8", width=1.8, dash="dot"),
                yaxis="y2",
            )
        )

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            x=0.02,
            font=dict(size=16, color="#f8fafc"),
        ),
        paper_bgcolor=COLOR_BACKGROUND,
        plot_bgcolor=COLOR_BACKGROUND,
        xaxis=dict(
            title=dict(text="Quadro / Frame", font=dict(color="#cbd5e1")),
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.1)",
            tickfont=dict(color="#94a3b8"),
        ),
        yaxis=dict(
            title=dict(text="Quantidade de Pessoas", font=dict(color="#cbd5e1")),
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.1)",
            tickfont=dict(color="#94a3b8"),
        ),
        yaxis2=dict(
            title=dict(text="Taxa (%)", font=dict(color="#38bdf8")),
            overlaying="y",
            side="right",
            range=[0, 105],
            showgrid=False,
            tickfont=dict(color="#38bdf8"),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#cbd5e1"),
        ),
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode="x unified",
        height=350,
    )
    return fig


def create_confidence_distribution_chart(
    df_detections: pd.DataFrame,
    title: str = "Distribuição de Confiança das Detecções",
) -> go.Figure:
    """
    Renderiza histograma comparativo da confiança das predições por classe.

    Args:
        df_detections: DataFrame com registros individuais de detecção.
        title: Título do gráfico.

    Returns:
        Figura do Plotly.
    """
    if df_detections is None or df_detections.empty or "confidence" not in df_detections.columns:
        return _empty_figure("Nenhuma detecção com métrica de confiança registrada.")

    df_plot = df_detections.dropna(subset=["confidence"]).copy()
    if df_plot.empty:
        return _empty_figure("Dados de confiança vazios.")

    if "class_name" not in df_plot.columns:
        df_plot["class_name"] = "Detecção"

    fig = px.histogram(
        df_plot,
        x="confidence",
        color="class_name",
        barmode="overlay",
        nbins=20,
        color_discrete_map={
            "head": COLOR_VIOLATION,
            "helmet": COLOR_CONFORMANT,
            "Head": COLOR_VIOLATION,
            "Helmet": COLOR_CONFORMANT,
        },
        opacity=0.75,
        title=f"<b>{title}</b>",
        labels={"confidence": "Confiança (Score)", "class_name": "Classe"},
    )

    fig.update_layout(
        paper_bgcolor=COLOR_BACKGROUND,
        plot_bgcolor=COLOR_BACKGROUND,
        font=dict(color="#cbd5e1"),
        xaxis=dict(
            range=[0.0, 1.02],
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.1)",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.1)",
            title="Frequência",
        ),
        margin=dict(l=30, r=30, t=50, b=30),
        height=320,
    )
    return fig


def create_camera_ranking_chart(
    df_detections: pd.DataFrame,
    title: str = "Conformidade por Câmera / Área",
) -> go.Figure:
    """
    Renderiza gráfico de barras agrupadas comparando conformidade entre câmeras.

    Args:
        df_detections: DataFrame de detecções.
        title: Título do gráfico.

    Returns:
        Figura do Plotly.
    """
    if (
        df_detections is None
        or df_detections.empty
        or "camera_id" not in df_detections.columns
        or "class_name" not in df_detections.columns
    ):
        return _empty_figure("Dados insuficientes para ranking de câmeras.")

    grouped = (
        df_detections.groupby(["camera_id", "class_name"])
        .size()
        .reset_index(name="count")
    )

    if grouped.empty:
        return _empty_figure("Nenhuma câmera registrada.")

    fig = px.bar(
        grouped,
        x="camera_id",
        y="count",
        color="class_name",
        barmode="group",
        color_discrete_map={
            "head": COLOR_VIOLATION,
            "helmet": COLOR_CONFORMANT,
            "Head": COLOR_VIOLATION,
            "Helmet": COLOR_CONFORMANT,
        },
        title=f"<b>{title}</b>",
        labels={"camera_id": "Câmera / Local", "count": "Detecções", "class_name": "Classe"},
    )

    fig.update_layout(
        paper_bgcolor=COLOR_BACKGROUND,
        plot_bgcolor=COLOR_BACKGROUND,
        font=dict(color="#cbd5e1"),
        xaxis=dict(showgrid=False),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.1)",
        ),
        margin=dict(l=30, r=30, t=50, b=30),
        height=320,
    )
    return fig
