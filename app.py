"""
EPI Finder - Dashboard Interativo de SST & CIPA (Etapa 7.2).

Painel de controle em Streamlit para monitoramento de conformidade de EPIs
(head vs helmet), rastreamento em vídeo (MOT ByteTrack / BoT-SORT),
teste estático com webcam e diagnósticos analíticos com Plotly.
"""

from datetime import datetime
import hashlib
import html
from pathlib import Path
import sys
import tempfile
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Garante acesso aos módulos em src/
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics import (
    calculate_kpis,
    create_camera_ranking_chart,
    create_compliance_donut_chart,
    create_confidence_distribution_chart,
    create_temporal_series_chart,
    load_compliance_report,
    load_frame_summary,
)
from src.inference import (
    get_default_device,
    infer_single_image,
    load_model,
    process_video_stream,
)


# Configuração inicial da página
st.set_page_config(
    page_title="EPI Finder · Monitoramento de EPI",
    page_icon="👷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilização visual customizada (Tema escuro de SST, alinhado a .streamlit/config.toml)
CUSTOM_CSS = """
<style>
    :root {
        --epi-surface: #1e293b;
        --epi-surface-2: #172033;
        --epi-border: #334155;
        --epi-text: #f8fafc;
        --epi-muted: #94a3b8;
        --epi-primary: #3b82f6;
        --epi-good: #10b981;
        --epi-warn: #f59e0b;
        --epi-bad: #ef4444;
    }

    /* Layout geral */
    .block-container {
        padding-top: 3.5rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Cabeçalho (hero) */
    .hero {
        display: flex;
        flex-wrap: wrap;
        align-items: flex-end;
        justify-content: space-between;
        gap: 1rem;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
        border: 1px solid var(--epi-border);
        border-radius: 16px;
        background:
            radial-gradient(circle at 0% 0%, rgba(59, 130, 246, 0.18), transparent 55%),
            linear-gradient(135deg, var(--epi-surface) 0%, var(--epi-surface-2) 100%);
    }
    .hero-eyebrow {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #60a5fa;
        margin-bottom: 0.3rem;
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 750;
        line-height: 1.15;
        color: var(--epi-text);
    }
    .hero-sub {
        font-size: 0.98rem;
        color: var(--epi-muted);
        margin-top: 0.35rem;
        max-width: 640px;
    }
    .chips {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
    }
    .chip {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        padding: 0.3rem 0.7rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 500;
        color: #cbd5e1;
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid var(--epi-border);
        white-space: nowrap;
    }

    /* Abas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.25rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.55rem 1rem;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [data-baseweb="tab"] p {
        font-size: 0.95rem;
        font-weight: 600;
    }

    /* Introdução de cada aba e títulos de seção */
    .tab-intro {
        color: var(--epi-muted);
        font-size: 0.95rem;
        margin: 0.4rem 0 1rem;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 650;
        color: var(--epi-text);
        margin: 1.4rem 0 0.15rem;
    }
    .section-sub {
        font-size: 0.85rem;
        color: var(--epi-muted);
        margin-bottom: 0.6rem;
    }
    .card-title {
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: var(--epi-muted);
        margin-bottom: 0.5rem;
    }

    /* Cards de KPI */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 0.75rem;
        margin: 0.4rem 0 1rem;
    }
    .kpi {
        background: var(--epi-surface);
        border: 1px solid var(--epi-border);
        border-left: 4px solid #475569;
        border-radius: 12px;
        padding: 0.85rem 1rem;
    }
    .kpi-label {
        font-size: 0.78rem;
        font-weight: 500;
        color: var(--epi-muted);
    }
    .kpi-value {
        font-size: 1.75rem;
        font-weight: 700;
        line-height: 1.2;
        color: var(--epi-text);
        margin-top: 0.15rem;
        font-variant-numeric: tabular-nums;
    }
    .kpi-hint {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 0.15rem;
    }
    .kpi-good { border-left-color: var(--epi-good); }
    .kpi-good .kpi-value { color: #34d399; }
    .kpi-warn { border-left-color: var(--epi-warn); }
    .kpi-warn .kpi-value { color: #fbbf24; }
    .kpi-bad { border-left-color: var(--epi-bad); }
    .kpi-bad .kpi-value { color: #f87171; }
    .kpi-info { border-left-color: var(--epi-primary); }

    /* Veredito / status */
    .verdict {
        display: flex;
        align-items: center;
        gap: 0.9rem;
        padding: 0.9rem 1.1rem;
        border-radius: 12px;
        border: 1px solid;
        margin: 0.3rem 0 0.6rem;
    }
    .verdict-icon { font-size: 1.6rem; line-height: 1; }
    .verdict-title { font-size: 1.05rem; font-weight: 700; }
    .verdict-text { font-size: 0.88rem; color: #cbd5e1; margin-top: 0.1rem; }
    .verdict-good { background: rgba(16, 185, 129, 0.12); border-color: rgba(16, 185, 129, 0.45); }
    .verdict-good .verdict-title { color: #34d399; }
    .verdict-warn { background: rgba(245, 158, 11, 0.12); border-color: rgba(245, 158, 11, 0.45); }
    .verdict-warn .verdict-title { color: #fbbf24; }
    .verdict-bad { background: rgba(239, 68, 68, 0.12); border-color: rgba(239, 68, 68, 0.45); }
    .verdict-bad .verdict-title { color: #f87171; }
    .verdict-neutral { background: rgba(148, 163, 184, 0.08); border-color: var(--epi-border); }
    .verdict-neutral .verdict-title { color: var(--epi-text); }

    /* Estado vazio */
    .empty {
        text-align: center;
        padding: 2.6rem 1.5rem;
        border: 1px dashed var(--epi-border);
        border-radius: 14px;
        background: rgba(30, 41, 59, 0.35);
        margin: 0.5rem 0 1rem;
    }
    .empty-icon { font-size: 2.2rem; margin-bottom: 0.4rem; }
    .empty-title { font-size: 1.05rem; font-weight: 650; color: var(--epi-text); }
    .empty-text {
        font-size: 0.9rem;
        color: var(--epi-muted);
        max-width: 520px;
        margin: 0.3rem auto 0;
    }

    /* Lista chave/valor (resumo de configuração) */
    .kv { display: grid; grid-template-columns: auto 1fr; gap: 0.35rem 1rem; font-size: 0.88rem; margin-bottom: 0.9rem; }
    .kv dt { color: var(--epi-muted); }
    .kv dd { margin: 0; color: var(--epi-text); font-weight: 500; text-align: right; }

    /* Barra lateral */
    .brand {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        padding: 0.2rem 0 0.9rem;
    }
    .brand-logo {
        width: 42px;
        height: 42px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.4rem;
        background: linear-gradient(135deg, #2563eb, #0ea5e9);
    }
    .brand-name { font-size: 1.1rem; font-weight: 700; color: var(--epi-text); line-height: 1.1; }
    .brand-tag { font-size: 0.75rem; color: var(--epi-muted); }
    .device-pill {
        margin-top: 0.8rem;
        padding: 0.55rem 0.8rem;
        border-radius: 10px;
        font-size: 0.82rem;
        color: #cbd5e1;
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid var(--epi-border);
    }

    /* Rodapé */
    .footer {
        margin-top: 2.5rem;
        padding-top: 1rem;
        border-top: 1px solid #1e293b;
        font-size: 0.8rem;
        color: #64748b;
        text-align: center;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Componentes visuais reutilizáveis
# ---------------------------------------------------------------------------

def _html(markup: str, target: Any = None) -> None:
    """Renderiza HTML confiável (gerado internamente) no destino informado."""
    (target or st).markdown(markup, unsafe_allow_html=True)


def rate_tone(rate: Optional[float]) -> str:
    """Classifica uma taxa de conformidade nas faixas usadas pelo dashboard da CIPA."""
    if rate is None:
        return "neutral"
    if rate >= 90.0:
        return "good"
    if rate >= 75.0:
        return "warn"
    return "bad"


def format_rate(rate: float, total: int) -> str:
    """Exibe '—' quando não há pessoas, evitando um 100% enganoso."""
    return f"{rate:.1f}%" if total > 0 else "—"


def render_kpis(items: List[Dict[str, Any]], target: Any = None) -> None:
    """Renderiza uma grade responsiva de cards de KPI (label, value, tone, hint)."""
    cards = []
    for item in items:
        hint = item.get("hint")
        hint_html = f"<div class='kpi-hint'>{html.escape(hint)}</div>" if hint else ""
        cards.append(
            f"<div class='kpi kpi-{item.get('tone', 'neutral')}'>"
            f"<div class='kpi-label'>{html.escape(item['label'])}</div>"
            f"<div class='kpi-value'>{html.escape(str(item['value']))}</div>"
            f"{hint_html}</div>"
        )
    _html(f"<div class='kpi-grid'>{''.join(cards)}</div>", target)


def render_person_kpis(result: Dict[str, Any], target: Any = None, scope: str = "") -> None:
    """KPIs padrão de uma inspeção (imagem, webcam ou quadro de vídeo)."""
    total = result["total_persons"]
    rate = result["compliance_rate_percent"]
    render_kpis(
        [
            {"label": f"Pessoas{scope}", "value": total, "tone": "info"},
            {"label": "Com capacete", "value": result["conformant_count"], "tone": "good"},
            {
                "label": "Sem capacete",
                "value": result["violation_count"],
                "tone": "bad" if result["violation_count"] else "neutral",
            },
            {
                "label": "Conformidade",
                "value": format_rate(rate, total),
                "tone": rate_tone(rate if total else None),
            },
        ],
        target,
    )


def render_verdict(tone: str, icon: str, title: str, text: str, target: Any = None) -> None:
    """Faixa de destaque com o veredito da inspeção."""
    _html(
        f"<div class='verdict verdict-{tone}'>"
        f"<div class='verdict-icon'>{icon}</div>"
        f"<div><div class='verdict-title'>{html.escape(title)}</div>"
        f"<div class='verdict-text'>{html.escape(text)}</div></div></div>",
        target,
    )


def render_inspection_verdict(result: Dict[str, Any]) -> None:
    """Traduz o resultado de uma inspeção em um veredito de SST legível."""
    persons = result["total_persons"]
    violations = result["violation_count"]
    if persons == 0:
        render_verdict(
            "neutral",
            "🔍",
            "Nenhuma pessoa detectada",
            "Tente outra imagem ou reduza o limiar de confiança na barra lateral.",
        )
    elif violations > 0:
        render_verdict(
            "bad",
            "⚠️",
            f"{violations} de {persons} pessoa(s) sem capacete",
            "Infração à NR-6 identificada. Confira as evidências fotográficas abaixo.",
        )
    else:
        render_verdict(
            "good",
            "✅",
            "Todos usando capacete",
            f"{persons} pessoa(s) em conformidade com a NR-6.",
        )


def render_tab_intro(text: str) -> None:
    _html(f"<div class='tab-intro'>{text}</div>")


def render_section(title: str, subtitle: Optional[str] = None) -> None:
    sub = f"<div class='section-sub'>{subtitle}</div>" if subtitle else ""
    _html(f"<div class='section-title'>{title}</div>{sub}")


def render_card_title(title: str) -> None:
    _html(f"<div class='card-title'>{title}</div>")


def render_empty_state(icon: str, title: str, text: str) -> None:
    _html(
        f"<div class='empty'><div class='empty-icon'>{icon}</div>"
        f"<div class='empty-title'>{title}</div>"
        f"<div class='empty-text'>{text}</div></div>"
    )


def render_crop_gallery(
    crops: List[Dict[str, Any]],
    caption_fn: Callable[[Dict[str, Any]], str],
    columns: int = 6,
) -> None:
    """Galeria de recortes com número fixo de colunas (evita miniaturas gigantes)."""
    cols = st.columns(columns)
    for idx, crop in enumerate(crops):
        with cols[idx % columns]:
            st.image(crop["crop_rgb"], caption=caption_fn(crop), width="stretch")


def detections_table(detections: List[Dict[str, Any]]) -> None:
    """Tabela de detecções com rótulos legíveis e barra de confiança."""
    df = pd.DataFrame(
        {
            "#": [d["detection_id"] for d in detections],
            "Classe": [
                "🔴 Sem capacete" if d["is_violation"] else "🟢 Com capacete" for d in detections
            ],
            "Confiança": [d["confidence"] for d in detections],
            "Caixa (x1, y1, x2, y2)": [", ".join(map(str, d["bbox_xyxy"])) for d in detections],
        }
    )
    st.dataframe(
        df,
        hide_index=True,
        width="stretch",
        column_config={
            "Confiança": st.column_config.ProgressColumn(
                "Confiança", min_value=0.0, max_value=1.0, format="%.2f"
            ),
        },
    )


def persist_upload(uploaded_file: Any, suffix: str) -> str:
    """
    Grava um upload em disco com nome derivado do conteúdo, para que reruns
    do Streamlit reutilizem o mesmo caminho (e o cache do modelo) em vez de
    criar um novo arquivo temporário a cada interação.
    """
    data = uploaded_file.getvalue()
    digest = hashlib.sha1(data).hexdigest()[:16]
    path = Path(tempfile.gettempdir()) / f"epi_finder_{digest}{suffix}"
    if not path.exists():
        path.write_bytes(data)
    return str(path)


def decode_image(uploaded_file: Any) -> Optional[np.ndarray]:
    file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------------
# Modelo, barra lateral e cabeçalho
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Carregando pesos do modelo YOLOv8...")
def get_cached_model(weights_path: str):
    """Carrega o modelo YOLOv8 em cache para evitar recarregamentos a cada interação."""
    return load_model(weights_path)


def render_sidebar() -> Dict[str, Any]:
    """Renderiza a barra lateral com configurações operacionais e limiares."""
    with st.sidebar:
        _html(
            "<div class='brand'><div class='brand-logo'>🦺</div>"
            "<div><div class='brand-name'>EPI Finder</div>"
            "<div class='brand-tag'>Painel de controle</div></div></div>"
        )

        with st.expander("🧠 Modelo", expanded=True):
            model_options = {
                "Pesos treinados (best.pt)": "models/best.pt",
                "Baseline pré-treinado (yolov8n.pt)": "yolov8n.pt",
            }
            selected_model_label = st.selectbox(
                "Checkpoint",
                list(model_options.keys()),
                index=0 if Path("models/best.pt").exists() else 1,
            )
            weights_path = model_options[selected_model_label]
            model_label = Path(weights_path).name

            uploaded_weights = st.file_uploader(
                "Ou envie pesos customizados (.pt)", type=["pt"], key="custom_weights"
            )
            if uploaded_weights is not None:
                weights_path = persist_upload(uploaded_weights, ".pt")
                model_label = uploaded_weights.name

        with st.expander("🎯 Detecção", expanded=True):
            conf_threshold = st.slider(
                "Limiar de confiança",
                min_value=0.05,
                max_value=1.0,
                value=0.25,
                step=0.05,
                help="Pontuação mínima do modelo para aceitar uma detecção de cabeça ou capacete. "
                "Valores menores aumentam o recall de infrações.",
            )
            iou_threshold = st.slider(
                "Limiar de NMS (IoU)",
                min_value=0.10,
                max_value=1.0,
                value=0.45,
                step=0.05,
                help="Limiar de sobreposição para supressão de não-máximos entre caixas adjacentes.",
            )
            camera_id = st.text_input(
                "Câmera / área monitorada",
                value="Posto Operacional - Câmera 01",
            )

        with st.expander("🔄 Rastreamento (vídeo)"):
            enable_tracking = st.toggle(
                "Ativar rastreamento (MOT)",
                value=True,
                help="Atribui IDs únicos a cada trabalhador em movimento nos vídeos.",
            )
            tracker_algo = st.selectbox(
                "Algoritmo",
                ["bytetrack.yaml", "botsort.yaml"],
                index=0,
                format_func=lambda x: "ByteTrack" if x.startswith("bytetrack") else "BoT-SORT",
                disabled=not enable_tracking,
            )
            min_consecutive_frames = st.slider(
                "Estabilização temporal (quadros)",
                min_value=1,
                max_value=10,
                value=3,
                disabled=not enable_tracking,
                help="Exige N frames consecutivos com o mesmo estado antes de confirmar infração.",
            )
            dedup_by_track = st.toggle(
                "Um registro por pessoa",
                value=True,
                disabled=not enable_tracking,
                help="Opção B: registra no relatório apenas a primeira infração de cada pessoa rastreada.",
            )

        with st.expander("🎨 Visualização"):
            alert_labels = st.toggle("Rótulos de alerta SST", value=True)
            draw_banner = st.toggle("Barra de telemetria no quadro", value=True)

        device = get_default_device()
        device_label = "GPU CUDA ⚡" if device != "cpu" else "CPU 💻"
        _html(f"<div class='device-pill'>Executando em <b>{device_label}</b></div>")

    return {
        "weights_path": weights_path,
        "model_label": model_label,
        "conf_threshold": conf_threshold,
        "iou_threshold": iou_threshold,
        "camera_id": camera_id,
        "enable_tracking": enable_tracking,
        "tracker_algo": tracker_algo,
        "min_consecutive_frames": min_consecutive_frames,
        "dedup_by_track": dedup_by_track,
        "alert_labels": alert_labels,
        "draw_banner": draw_banner,
        "device": device,
    }


def tracker_name(config: Dict[str, Any]) -> str:
    return "ByteTrack" if config["tracker_algo"].startswith("bytetrack") else "BoT-SORT"


def render_header(config: Dict[str, Any]) -> None:
    """Cabeçalho com identidade do produto e resumo da configuração ativa."""
    tracking = f"Tracking: {tracker_name(config)}" if config["enable_tracking"] else "Tracking desligado"
    chips = [
        f"🧠 {html.escape(config['model_label'])}",
        f"🎯 Confiança ≥ {config['conf_threshold']:.2f}",
        f"🔄 {tracking}",
        "⚡ GPU" if config["device"] != "cpu" else "💻 CPU",
    ]
    chips_html = "".join(f"<span class='chip'>{c}</span>" for c in chips)
    _html(
        "<div class='hero'><div>"
        "<div class='hero-eyebrow'>Segurança do Trabalho · NR-6</div>"
        "<div class='hero-title'>🛡️ EPI Finder</div>"
        "<div class='hero-sub'>Detecção e rastreamento de capacetes de segurança com YOLOv8 "
        "para auditorias de SST e CIPA.</div>"
        f"</div><div class='chips'>{chips_html}</div></div>"
    )


# ---------------------------------------------------------------------------
# Abas
# ---------------------------------------------------------------------------

def render_tab_images(model, config: Dict[str, Any]) -> None:
    """Aba 1: Detecção e análise em imagens estáticas."""
    render_tab_intro(
        "Envie fotos de postos de trabalho, canteiros de obra ou linhas de produção "
        "para uma inspeção instantânea do uso de capacete."
    )

    # Busca imagens de amostra locais para conveniência
    sample_images = []
    test_dir = PROJECT_ROOT / "data" / "dataset" / "test" / "images"
    if test_dir.exists():
        sample_images = [str(p) for p in sorted(test_dir.glob("*.jpg"))[:10]]

    image_bgr = None
    with st.container(border=True):
        source = "upload"
        if sample_images:
            source = st.radio(
                "Origem da imagem",
                ["upload", "sample"],
                format_func=lambda x: "📤 Enviar arquivo" if x == "upload" else "🗂️ Amostra do dataset",
                horizontal=True,
                key="image_source",
                label_visibility="collapsed",
            )

        if source == "upload":
            uploaded_image = st.file_uploader(
                "Selecione uma imagem (JPG, PNG ou WEBP)",
                type=["jpg", "jpeg", "png", "webp"],
                key="image_uploader",
            )
            if uploaded_image is not None:
                image_bgr = decode_image(uploaded_image)
        else:
            sample_choice = st.selectbox(
                "Imagem de amostra",
                sample_images,
                format_func=lambda x: Path(x).name,
            )
            image_bgr = cv2.imread(sample_choice)

    if image_bgr is None:
        render_empty_state(
            "📸",
            "Nenhuma imagem selecionada",
            "Envie uma foto acima para ver as detecções, o veredito de conformidade "
            "e os recortes de evidência de cada infração.",
        )
        return

    with st.spinner("Executando inferência e gerando diagnóstico de SST..."):
        result = infer_single_image(
            model=model,
            image_bgr=image_bgr,
            conf_threshold=config["conf_threshold"],
            iou_threshold=config["iou_threshold"],
            device=config["device"],
            alert_labels=config["alert_labels"],
            draw_banner=config["draw_banner"],
            camera_id=config["camera_id"],
        )

    render_inspection_verdict(result)
    render_person_kpis(result)

    c_annot, c_orig = st.columns(2, gap="medium")
    with c_annot:
        with st.container(border=True):
            render_card_title("Diagnóstico (detecções)")
            st.image(result["annotated_rgb"], width="stretch")
            _, encoded_img = cv2.imencode(".jpg", result["annotated_bgr"])
            st.download_button(
                label="💾 Baixar imagem anotada",
                data=encoded_img.tobytes(),
                file_name=f"epi_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                mime="image/jpeg",
            )
    with c_orig:
        with st.container(border=True):
            render_card_title("Imagem original")
            st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), width="stretch")

    # Galeria de Evidências das Infrações
    if result["violation_crops"]:
        render_section(
            "🚨 Evidências de infração",
            f"{len(result['violation_crops'])} recorte(s) de pessoas sem capacete",
        )
        render_crop_gallery(
            result["violation_crops"],
            lambda c: f"#{c['detection_id']} · conf. {c['confidence']:.2f}",
        )

    # Tabela Detalhada
    if result["detections"]:
        with st.expander(f"📋 Detalhes das detecções ({len(result['detections'])})"):
            detections_table(result["detections"])


def render_video_results() -> None:
    """Resumo persistente da última auditoria de vídeo (sobrevive a reruns)."""
    video_audit = st.session_state.get("video_audit")
    summary = st.session_state.get("audit_summary")
    df_rep = st.session_state.get("compliance_report_df")
    df_frames = st.session_state.get("frame_summary_df")
    if not video_audit or summary is None or df_rep is None or df_frames is None:
        return

    render_section(
        "📑 Resultado da auditoria",
        f"{html.escape(video_audit['source'])} · {summary['total_frames_processed']} quadros processados",
    )

    if summary["tracking_enabled"]:
        persons = summary["unique_persons_tracked"]
        violators = summary["unique_violators"]
        rate = summary["unique_compliance_rate_percent"]
        items = [
            {"label": "Pessoas únicas", "value": persons, "tone": "info", "hint": "via rastreamento MOT"},
            {"label": "Conformes", "value": summary["unique_conformant"], "tone": "good"},
            {"label": "Infratores", "value": violators, "tone": "bad" if violators else "neutral"},
            {"label": "Conformidade real", "value": format_rate(rate, persons), "tone": rate_tone(rate if persons else None)},
        ]
    else:
        total = summary["total_detections_logged"]
        violators = summary["violations_head_logged"]
        rate = summary["overall_compliance_rate_percent"]
        items = [
            {"label": "Detecções registradas", "value": total, "tone": "info"},
            {"label": "Com capacete", "value": summary["conformant_helmets_logged"], "tone": "good"},
            {"label": "Sem capacete", "value": violators, "tone": "bad" if violators else "neutral"},
            {"label": "Conformidade", "value": format_rate(rate, total), "tone": rate_tone(rate if total else None)},
        ]
    items.append(
        {
            "label": "Quadros com infração",
            "value": summary["frames_with_violations"],
            "tone": "warn" if summary["frames_with_violations"] else "neutral",
        }
    )

    if violators:
        render_verdict("bad", "⚠️", "Infrações identificadas no vídeo", "Revise as evidências e exporte o relatório para a CIPA.")
    else:
        render_verdict("good", "✅", "Nenhuma infração confirmada", "Todas as pessoas rastreadas estavam usando capacete.")
    render_kpis(items)

    crops = video_audit["crops"]
    if crops:
        render_section("🚨 Evidências por indivíduo", f"{len(crops)} recorte(s)")
        render_crop_gallery(crops[:18], lambda c: f"Track {c['track_id']} · quadro {c['frame_idx']}")

    c_dl1, c_dl2, c_hint = st.columns([1, 1, 2], vertical_alignment="center")
    with c_dl1:
        st.download_button(
            "📥 Relatório de ocorrências",
            data=df_rep.to_csv(index=False).encode("utf-8"),
            file_name="compliance_report.csv",
            mime="text/csv",
            help="compliance_report.csv",
        )
    with c_dl2:
        st.download_button(
            "📥 Resumo por quadro",
            data=df_frames.to_csv(index=False).encode("utf-8"),
            file_name="frame_summary.csv",
            mime="text/csv",
            help="frame_summary.csv",
        )
    with c_hint:
        st.caption("📊 Os gráficos desta auditoria estão na aba **Dashboard**.")


def render_tab_videos(model, config: Dict[str, Any]) -> None:
    """Aba 2: Processamento e rastreamento (MOT) em arquivos de vídeo."""
    render_tab_intro(
        "Monitore gravações de CFTV com rastreamento de múltiplos objetos, estabilização "
        "temporal contra ruídos e um registro por indivíduo."
    )

    sample_video_path = PROJECT_ROOT / "data" / "sample_cctv.mp4"
    target_video_path = None
    source_name = ""
    with st.container(border=True):
        source = "upload"
        if sample_video_path.exists():
            source = st.radio(
                "Origem do vídeo",
                ["upload", "sample"],
                format_func=lambda x: "📤 Enviar vídeo" if x == "upload" else "🎞️ Vídeo sintético de exemplo",
                horizontal=True,
                key="video_source",
                label_visibility="collapsed",
            )
        if source == "upload":
            uploaded_video = st.file_uploader(
                "Selecione um vídeo (MP4, AVI, MOV ou MKV)",
                type=["mp4", "avi", "mov", "mkv"],
                key="video_uploader",
            )
            if uploaded_video is not None:
                target_video_path = persist_upload(uploaded_video, Path(uploaded_video.name).suffix or ".mp4")
                source_name = uploaded_video.name
        else:
            target_video_path = str(sample_video_path)
            source_name = sample_video_path.name

    if not target_video_path:
        render_empty_state(
            "🎥",
            "Nenhum vídeo selecionado",
            "Envie uma gravação para acompanhar a análise quadro a quadro, com IDs de "
            "rastreamento e galeria de infratores em tempo real.",
        )
        render_video_results()
        return

    col_prev, col_cfg = st.columns([3, 2], gap="medium")
    with col_prev:
        with st.container(border=True):
            render_card_title("Pré-visualização")
            st.video(target_video_path)
    with col_cfg:
        with st.container(border=True):
            render_card_title("Configuração da análise")
            rows = {
                "Câmera / área": html.escape(config["camera_id"]),
                "Modelo": html.escape(config["model_label"]),
                "Confiança mínima": f"{config['conf_threshold']:.2f}",
                "Rastreamento": tracker_name(config) if config["enable_tracking"] else "Desligado",
            }
            if config["enable_tracking"]:
                rows["Estabilização"] = f"{config['min_consecutive_frames']} quadro(s)"
                rows["Registro"] = "1 por pessoa" if config["dedup_by_track"] else "Todos os quadros"
            kv = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in rows.items())
            _html(f"<dl class='kv'>{kv}</dl>")
            st.caption("Ajuste esses parâmetros na barra lateral.")
            start_btn = st.button("🚀 Iniciar análise", type="primary", width="stretch")

    if start_btn:
        render_section("⏱️ Análise em andamento")
        progress_bar = st.progress(0, text="Iniciando processamento...")
        col_frame, col_side = st.columns([3, 1.4], gap="medium")
        with col_frame:
            frame_placeholder = st.empty()
        with col_side:
            render_card_title("Quadro atual")
            kpi_placeholder = st.empty()
            gallery_placeholder = st.empty()

        accumulated_crops: List[Dict[str, Any]] = []
        last_auditor = None

        stream_gen = process_video_stream(
            video_source=target_video_path,
            model=model,
            conf_threshold=config["conf_threshold"],
            iou_threshold=config["iou_threshold"],
            device=config["device"],
            camera_id=config["camera_id"],
            enable_tracking=config["enable_tracking"],
            tracker=config["tracker_algo"],
            min_consecutive_frames=config["min_consecutive_frames"],
            dedup_by_track=config["dedup_by_track"],
            alert_labels=config["alert_labels"],
            draw_banner=config["draw_banner"],
        )

        for step in stream_gen:
            progress_bar.progress(
                step["progress_percent"] / 100.0,
                text=f"Quadro {step['frame_idx']} de {step['total_frames']} ({step['progress_percent']:.0f}%)",
            )
            frame_placeholder.image(step["annotated_rgb"], width="stretch")
            render_person_kpis(step, target=kpi_placeholder)

            if step["new_violation_crops"]:
                accumulated_crops.extend(step["new_violation_crops"])
                with gallery_placeholder.container():
                    render_card_title(f"Infratores recentes ({len(accumulated_crops)})")
                    g_cols = st.columns(3)
                    for idx, crop in enumerate(accumulated_crops[-6:]):
                        with g_cols[idx % 3]:
                            st.image(crop["crop_rgb"], caption=f"Track {crop['track_id']}", width="stretch")

            last_auditor = step["auditor"]

        progress_bar.progress(1.0, text="Processamento concluído.")

        if last_auditor:
            # Salva os relatórios na sessão para alimentar a aba Analítica
            st.session_state["compliance_report_df"] = last_auditor.to_dataframe()
            st.session_state["frame_summary_df"] = last_auditor.to_frame_summary_dataframe()
            st.session_state["audit_summary"] = last_auditor.get_audit_summary()
            st.session_state["video_audit"] = {"source": source_name, "crops": accumulated_crops}
            st.toast("Auditoria de vídeo finalizada!", icon="✅")

    render_video_results()


def render_tab_webcam(model, config: Dict[str, Any]) -> None:
    """Aba 3: Captura pontual via webcam do navegador (Streamlit camera_input)."""
    render_tab_intro(
        "Tire uma foto pela câmera do navegador e veja na hora se o capacete foi reconhecido. "
        "Funciona também no deploy em nuvem."
    )

    col_cam, col_res = st.columns(2, gap="medium")
    with col_cam:
        with st.container(border=True):
            render_card_title("Câmera")
            camera_photo = st.camera_input(
                "Posicione-se em frente à câmera e clique para capturar",
                label_visibility="collapsed",
            )

    with col_res:
        if camera_photo is None:
            render_empty_state(
                "📷",
                "Aguardando captura",
                "Permita o acesso à câmera, posicione-se no enquadramento e tire uma foto.",
            )
            return

        image_bgr = decode_image(camera_photo)
        with st.spinner("Processando foto da webcam..."):
            result = infer_single_image(
                model=model,
                image_bgr=image_bgr,
                conf_threshold=config["conf_threshold"],
                iou_threshold=config["iou_threshold"],
                device=config["device"],
                alert_labels=config["alert_labels"],
                draw_banner=config["draw_banner"],
                camera_id="Webcam-Nuvem",
            )

        render_inspection_verdict(result)
        render_person_kpis(result)
        with st.container(border=True):
            render_card_title("Resultado da inspeção")
            st.image(result["annotated_rgb"], width="stretch")


def render_tab_analytics() -> None:
    """Aba 4: Dashboard analítico consolidado para CIPA e Auditoria SST."""
    render_tab_intro(
        "Indicadores consolidados de uso de capacete para a CIPA. Os dados vêm da última "
        "análise de vídeo desta sessão ou de relatórios CSV enviados abaixo."
    )

    # Carrega dados da sessão ou de arquivos locais padrão
    df_detections = st.session_state.get("compliance_report_df", None)
    df_frames = st.session_state.get("frame_summary_df", None)
    audit_summary = st.session_state.get("audit_summary", None)
    data_source = "última análise de vídeo desta sessão" if df_detections is not None else None

    # Permite upload de relatórios CSV anteriores
    with st.expander("📁 Carregar relatórios anteriores (CSV)", expanded=(df_detections is None)):
        c_up1, c_up2 = st.columns(2)
        with c_up1:
            up_rep = st.file_uploader("Relatório de ocorrências (compliance_report.csv)", type=["csv"], key="csv_rep")
            if up_rep is not None:
                df_detections = pd.read_csv(up_rep)
                data_source = "relatório CSV enviado"
        with c_up2:
            up_frm = st.file_uploader("Resumo por quadro (frame_summary.csv)", type=["csv"], key="csv_frm")
            if up_frm is not None:
                df_frames = pd.read_csv(up_frm)
                data_source = data_source or "relatório CSV enviado"

    # Se ainda estiver vazio, tenta ler os arquivos padrão de runs/inference
    if df_detections is None:
        default_csv = PROJECT_ROOT / "runs" / "inference" / "compliance_report.csv"
        if default_csv.exists():
            df_detections = load_compliance_report(default_csv)
            data_source = data_source or "runs/inference (execução local da CLI)"

    if df_frames is None:
        default_frames = PROJECT_ROOT / "runs" / "inference" / "frame_summary.csv"
        if default_frames.exists():
            df_frames = load_frame_summary(default_frames)
            data_source = data_source or "runs/inference (execução local da CLI)"

    if df_detections is None and df_frames is None:
        render_empty_state(
            "📊",
            "Ainda não há dados para o dashboard",
            "Execute uma análise na aba <b>Vídeos</b> ou carregue relatórios CSV de auditorias anteriores.",
        )
        return

    kpis = calculate_kpis(
        df_detections=df_detections,
        df_frames=df_frames,
        audit_summary=audit_summary,
    )

    # Faixa de status da auditoria
    effective_rate = (
        kpis["unique_compliance_rate_percent"] if kpis["has_tracking"] else kpis["compliance_rate_percent"]
    )
    status_tone = rate_tone(effective_rate)
    status_icon = {"good": "✅", "warn": "⚠️", "bad": "🚨"}[status_tone]
    render_verdict(
        status_tone,
        status_icon,
        f"Status da auditoria: {kpis['status']}",
        f"Conformidade efetiva de {effective_rate:.1f}% · fonte: {data_source}",
    )

    # Cards de KPIs
    items = [
        {"label": "Inspeções registradas", "value": kpis["total_detections"], "tone": "info"},
        {"label": "Com capacete", "value": kpis["helmet_count"], "tone": "good"},
        {
            "label": "Sem capacete",
            "value": kpis["head_count"],
            "tone": "bad" if kpis["head_count"] else "neutral",
        },
        {
            "label": "Conformidade geral",
            "value": f"{kpis['compliance_rate_percent']:.1f}%",
            "tone": rate_tone(kpis["compliance_rate_percent"]),
            "hint": "por detecção",
        },
    ]
    if kpis["has_tracking"]:
        items.append(
            {
                "label": "Conformidade real",
                "value": f"{kpis['unique_compliance_rate_percent']:.1f}%",
                "tone": rate_tone(kpis["unique_compliance_rate_percent"]),
                "hint": f"{kpis['unique_persons']} pessoa(s) única(s)",
            }
        )
    render_kpis(items)

    # Gráficos em Grade (2x2)
    has_detections = df_detections is not None and not df_detections.empty
    has_frames = df_frames is not None and not df_frames.empty

    g1, g2 = st.columns(2, gap="medium")
    with g1:
        with st.container(border=True):
            st.plotly_chart(
                create_compliance_donut_chart(
                    conformant_count=kpis["helmet_count"],
                    violation_count=kpis["head_count"],
                    title="Taxa Geral de Uso de EPI",
                ),
                width="stretch",
            )
    with g2:
        with st.container(border=True):
            st.plotly_chart(
                create_temporal_series_chart(
                    df_frames=df_frames if has_frames else pd.DataFrame(),
                    title="Série Temporal de Conformidade",
                ),
                width="stretch",
            )

    g3, g4 = st.columns(2, gap="medium")
    with g3:
        with st.container(border=True):
            st.plotly_chart(
                create_confidence_distribution_chart(
                    df_detections=df_detections if has_detections else pd.DataFrame(),
                    title="Distribuição de Confiança das Predições",
                ),
                width="stretch",
            )
    with g4:
        with st.container(border=True):
            st.plotly_chart(
                create_camera_ranking_chart(
                    df_detections=df_detections if has_detections else pd.DataFrame(),
                    title="Conformidade por Câmera / Área",
                ),
                width="stretch",
            )

    # Tabela Interativa de Ocorrências
    if has_detections:
        with st.expander(f"📋 Registro consolidado de ocorrências ({len(df_detections)})"):
            st.dataframe(df_detections, hide_index=True, width="stretch")


def main() -> None:
    """Função orquestradora da aplicação Streamlit."""
    config = render_sidebar()
    render_header(config)

    # Carrega modelo YOLO em cache
    try:
        model = get_cached_model(config["weights_path"])
    except Exception as exc:
        st.error(f"Erro ao carregar o modelo YOLO: {exc}")
        return

    tab_images, tab_videos, tab_webcam, tab_analytics = st.tabs([
        "📸 Imagens",
        "🎥 Vídeos",
        "📷 Webcam",
        "📊 Dashboard",
    ])

    with tab_images:
        render_tab_images(model, config)

    with tab_videos:
        render_tab_videos(model, config)

    with tab_webcam:
        render_tab_webcam(model, config)

    with tab_analytics:
        render_tab_analytics()

    _html(
        "<div class='footer'>EPI Finder · YOLOv8 + ByteTrack · "
        "Projeto de estudo em Visão Computacional aplicada à NR-6</div>"
    )


if __name__ == "__main__":
    main()
