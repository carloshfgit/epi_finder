"""
EPI Finder - Dashboard Interativo de SST & CIPA (Etapa 7.2).

Painel de controle em Streamlit para monitoramento de conformidade de EPIs
(head vs helmet), rastreamento em vídeo (MOT ByteTrack / BoT-SORT),
teste estático com webcam e diagnósticos analíticos com Plotly.
"""

from datetime import datetime
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

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
    ALERT_CLASSES,
    DEFAULT_CLASSES,
    get_default_device,
    infer_single_image,
    load_model,
    process_video_stream,
)


# Configuração inicial da página
st.set_page_config(
    page_title="EPI Finder - Monitoramento de SST & CIPA",
    page_icon="👷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilização visual customizada (Tema profissional de SST)
CUSTOM_CSS = """
<style>
    /* Estilos gerais */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #f8fafc;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    /* Cards de métricas */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
    }
    /* Destaque de status */
    .status-badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .badge-success {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid #10b981;
    }
    .badge-warning {
        background-color: rgba(245, 158, 11, 0.2);
        color: #f59e0b;
        border: 1px solid #f59e0b;
    }
    .badge-danger {
        background-color: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid #ef4444;
    }
    .crop-card {
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 8px;
        background-color: #1e293b;
        text-align: center;
        margin-bottom: 10px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Carregando pesos do modelo YOLOv8...")
def get_cached_model(weights_path: str):
    """Carrega o modelo YOLOv8 em cache para evitar recarregamentos a cada interação."""
    return load_model(weights_path)


def render_sidebar() -> Dict[str, Any]:
    """Renderiza a barra lateral com configurações operacionais e limiares."""
    st.sidebar.image(
        "https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png",
        use_container_width=True,
    )
    st.sidebar.markdown("### ⚙️ Parâmetros Operacionais")

    # Seleção de Pesos do Modelo
    model_options = {
        "Pesos Treinados (models/best.pt)": "models/best.pt",
        "Baseline Pré-Treinado (yolov8n.pt)": "yolov8n.pt",
    }
    selected_model_label = st.sidebar.selectbox(
        "Checkpoint do Modelo:",
        list(model_options.keys()),
        index=0 if Path("models/best.pt").exists() else 1,
    )
    weights_path = model_options[selected_model_label]

    # Upload opcional de pesos externos
    uploaded_weights = st.sidebar.file_uploader(
        "Ou envie pesos customizados (.pt):", type=["pt"], key="custom_weights"
    )
    if uploaded_weights is not None:
        temp_weights = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
        temp_weights.write(uploaded_weights.read())
        temp_weights.flush()
        weights_path = temp_weights.name

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Limiares de Detecção")
    conf_threshold = st.sidebar.slider(
        "Limiar de Confiança (Confidence)",
        min_value=0.05,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Pontuação mínima do modelo para aceitar uma detecção de cabeça ou capacete.",
    )

    iou_threshold = st.sidebar.slider(
        "Limiar de NMS (IoU)",
        min_value=0.10,
        max_value=1.0,
        value=0.45,
        step=0.05,
        help="Limiar de sobreposição para supressão de não-máximos entre caixas adjacentes.",
    )

    camera_id = st.sidebar.text_input(
        "Identificador da Câmera / Área:",
        value="Posto Operacional - Câmera 01",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔄 Multi-Object Tracking (MOT)")
    enable_tracking = st.sidebar.checkbox(
        "Ativar Rastreamento (MOT)",
        value=True,
        help="Atribui IDs únicos a cada trabalhador em movimento nos vídeos.",
    )

    tracker_algo = st.sidebar.selectbox(
        "Algoritmo de Tracking:",
        ["bytetrack.yaml", "botsort.yaml"],
        index=0,
        disabled=not enable_tracking,
    )

    min_consecutive_frames = st.sidebar.slider(
        "Estabilização Temporal (Quadros)",
        min_value=1,
        max_value=10,
        value=3,
        disabled=not enable_tracking,
        help="Exige N frames consecutivos com o mesmo estado antes de confirmar infração.",
    )

    dedup_by_track = st.sidebar.checkbox(
        "Desduplicação de Logs por Indivíduo (Opção B)",
        value=True,
        disabled=not enable_tracking,
        help="Registra no relatório apenas a primeira infração de cada pessoa rastreada.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎨 Visualização & Alertas")
    alert_labels = st.sidebar.checkbox("Rótulos de Alerta SST", value=True)
    draw_banner = st.sidebar.checkbox("Exibir Barra Superior de Telemetria", value=True)

    device = get_default_device()
    device_label = "GPU CUDA ⚡" if device != "cpu" else "CPU 💻"
    st.sidebar.info(f"Dispositivo de Execução: **{device_label}**")

    return {
        "weights_path": weights_path,
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


def render_tab_images(model, config: Dict[str, Any]) -> None:
    """Aba 1: Detecção e análise em imagens estáticas."""
    st.subheader("📸 Detecção de EPI em Imagens Estáticas")
    st.markdown(
        "Faça o upload de fotos de postos de trabalho, canteiros de obra ou linhas de produção para inspeção visual instantânea de capacetes."
    )

    col_upload, col_sample = st.columns([2, 1])
    with col_upload:
        uploaded_image = st.file_uploader(
            "Selecione uma imagem (.jpg, .png, .jpeg, .webp):",
            type=["jpg", "jpeg", "png", "webp"],
            key="image_uploader",
        )

    # Busca imagens de amostra locais para conveniência
    sample_images = []
    test_dir = PROJECT_ROOT / "data" / "dataset" / "test" / "images"
    if test_dir.exists():
        sample_images = [str(p) for p in sorted(test_dir.glob("*.jpg"))[:10]]

    with col_sample:
        selected_sample = None
        if sample_images:
            use_sample = st.checkbox("Usar imagem de amostra do dataset", value=False)
            if use_sample:
                sample_choice = st.selectbox(
                    "Selecione a amostra:",
                    sample_images,
                    format_func=lambda x: Path(x).name,
                )
                selected_sample = sample_choice

    image_bgr = None
    if uploaded_image is not None:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    elif selected_sample:
        image_bgr = cv2.imread(selected_sample)

    if image_bgr is not None:
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

        # Painel de Métricas Rápidas
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Pessoas Detectadas", result["total_persons"])
        k2.metric("Com Capacete (Conforme)", result["conformant_count"])
        k3.metric("Sem Capacete (Infração)", result["violation_count"])
        k4.metric("Taxa de Conformidade", f"{result['compliance_rate_percent']:.1f}%")

        # Exibição das Imagens (Lado a Lado)
        c_orig, c_annot = st.columns(2)
        with c_orig:
            st.markdown("**Imagem Original:**")
            st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

        with c_annot:
            st.markdown("**Diagnóstico Operacional (Bounding Boxes):**")
            st.image(result["annotated_rgb"], use_container_width=True)

        # Botão de download da imagem processada
        _, encoded_img = cv2.imencode(".jpg", result["annotated_bgr"])
        st.download_button(
            label="💾 Baixar Imagem com Anotações",
            data=encoded_img.tobytes(),
            file_name=f"epi_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
            mime="image/jpeg",
        )

        # Galeria de Evidências das Infrações
        if result["violation_crops"]:
            st.markdown("### 🚨 Evidências Fotográficas de Infrações")
            cols = st.columns(min(4, len(result["violation_crops"])))
            for idx, crop_info in enumerate(result["violation_crops"]):
                col_target = cols[idx % len(cols)]
                with col_target:
                    st.image(
                        crop_info["crop_rgb"],
                        caption=f"Infração #{crop_info['detection_id']} | Conf: {crop_info['confidence']:.2f}",
                        use_container_width=True,
                    )

        # Tabela Detalhada
        if result["detections"]:
            st.markdown("### 📋 Tabela Detalhada de Detecções")
            df_det = pd.DataFrame(result["detections"])
            st.dataframe(df_det, use_container_width=True)
    else:
        st.info("💡 Faça o upload de uma imagem ou marque 'Usar imagem de amostra' para iniciar a inspeção.")


def render_tab_videos(model, config: Dict[str, Any]) -> None:
    """Aba 2: Processamento e rastreamento (MOT) em arquivos de vídeo."""
    st.subheader("🎥 Detecção e Rastreamento em Vídeo (MOT)")
    st.markdown(
        "Monitore fluxos contínuos de CFTV com **Multi-Object Tracking** (ByteTrack/BoT-SORT), "
        "estabilização temporal contra ruídos e desduplicação de alertas por indivíduo."
    )

    col_upload, col_sample = st.columns([2, 1])
    with col_upload:
        uploaded_video = st.file_uploader(
            "Selecione um arquivo de vídeo (.mp4, .avi, .mov, .mkv):",
            type=["mp4", "avi", "mov", "mkv"],
            key="video_uploader",
        )

    sample_video_path = PROJECT_ROOT / "data" / "sample_cctv.mp4"
    with col_sample:
        use_sample_video = False
        if sample_video_path.exists():
            use_sample_video = st.checkbox("Usar vídeo sintético de exemplo (sample_cctv.mp4)", value=False)

    target_video_path = None
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.flush()
        target_video_path = tfile.name
    elif use_sample_video and sample_video_path.exists():
        target_video_path = str(sample_video_path)

    if target_video_path:
        st.video(target_video_path)
        start_btn = st.button("🚀 Iniciar Análise de Vídeo com MOT", type="primary")

        if start_btn:
            progress_bar = st.progress(0, text="Iniciando processamento...")
            frame_placeholder = st.empty()

            m1, m2, m3, m4 = st.columns(4)
            metric_persons = m1.empty()
            metric_conformant = m2.empty()
            metric_violations = m3.empty()
            metric_rate = m4.empty()

            gallery_placeholder = st.empty()
            accumulated_crops = []
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
                    text=f"Processando quadro {step['frame_idx']} de {step['total_frames']} ({step['progress_percent']:.1f}%)",
                )
                frame_placeholder.image(step["annotated_rgb"], use_container_width=True)

                metric_persons.metric("Pessoas no Quadro", step["total_persons"])
                metric_conformant.metric("Conformes (Quadro)", step["conformant_count"])
                metric_violations.metric("Infrações (Quadro)", step["violation_count"])
                metric_rate.metric("Taxa Instantânea", f"{step['compliance_rate_percent']:.1f}%")

                if step["new_violation_crops"]:
                    accumulated_crops.extend(step["new_violation_crops"])
                    with gallery_placeholder.container():
                        st.markdown("#### 🚨 Novas Infrações Identificadas")
                        g_cols = st.columns(min(4, len(accumulated_crops)))
                        for idx, crop in enumerate(accumulated_crops[-4:]):
                            with g_cols[idx % len(g_cols)]:
                                st.image(
                                    crop["crop_rgb"],
                                    caption=f"Track ID: {crop['track_id']} (Quadro {crop['frame_idx']})",
                                    use_container_width=True,
                                )

                last_auditor = step["auditor"]

            progress_bar.progress(1.0, text="Processamento concluído com sucesso!")
            st.success("✅ Auditoria de vídeo finalizada!")

            if last_auditor:
                # Salva os relatórios na sessão para alimentar a aba Analítica
                df_rep = last_auditor.to_dataframe()
                df_frames = last_auditor.to_frame_summary_dataframe()
                st.session_state["compliance_report_df"] = df_rep
                st.session_state["frame_summary_df"] = df_frames
                st.session_state["audit_summary"] = last_auditor.get_audit_summary()

                # Botões para download do relatório
                c_dl1, c_dl2 = st.columns(2)
                with c_dl1:
                    csv_rep = df_rep.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Baixar Relatório de Ocorrências (compliance_report.csv)",
                        data=csv_rep,
                        file_name="compliance_report.csv",
                        mime="text/csv",
                    )
                with c_dl2:
                    csv_fr = df_frames.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Baixar Resumo Temporal (frame_summary.csv)",
                        data=csv_fr,
                        file_name="frame_summary.csv",
                        mime="text/csv",
                    )
    else:
        st.info("💡 Envie um vídeo ou ative o vídeo de exemplo para iniciar o monitoramento contínuo.")


def render_tab_webcam(model, config: Dict[str, Any]) -> None:
    """Aba 3: Captura pontual via webcam do navegador (Streamlit camera_input)."""
    st.subheader("📷 Teste com Webcam (Foto Estática)")
    st.markdown(
        "Tire uma foto pontual diretamente da sua câmera usando o componente nativo do Streamlit. "
        "Esta abordagem é totalmente compatível com deploys gratuitos em nuvem (ex: Streamlit Community Cloud)."
    )

    camera_photo = st.camera_input("Posicione-se em frente à câmera e clique para capturar:")

    if camera_photo is not None:
        file_bytes = np.asarray(bytearray(camera_photo.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

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

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Pessoas Detectadas", result["total_persons"])
        k2.metric("Com Capacete", result["conformant_count"])
        k3.metric("Sem Capacete", result["violation_count"])
        k4.metric("Conformidade", f"{result['compliance_rate_percent']:.1f}%")

        st.image(
            result["annotated_rgb"],
            caption="Resultado da Inspeção com Bounding Boxes",
            use_container_width=True,
        )

        if result["violation_count"] > 0:
            st.error("⚠️ ALERTA: Usuário identificado sem capacete de proteção!")
        else:
            st.success("✅ Parabéns: Usuário em total conformidade com as normas de SST!")


def render_tab_analytics() -> None:
    """Aba 4: Dashboard analítico consolidado para CIPA e Auditoria SST."""
    st.subheader("📊 Dashboard Analítico de Conformidade SST & CIPA")
    st.markdown(
        "Consolide relatórios de auditoria, monitore taxas reais de uso de capacetes e visualize diagnósticos visuais com Plotly."
    )

    # Carrega dados da sessão ou de arquivos locais padrão
    df_detections = st.session_state.get("compliance_report_df", None)
    df_frames = st.session_state.get("frame_summary_df", None)
    audit_summary = st.session_state.get("audit_summary", None)

    # Permite upload de relatórios CSV anteriores
    with st.expander("📁 Carregar Relatórios Anteriores (CSV)", expanded=(df_detections is None)):
        c_up1, c_up2 = st.columns(2)
        with c_up1:
            up_rep = st.file_uploader("Relatório de Detecções (compliance_report.csv):", type=["csv"], key="csv_rep")
            if up_rep is not None:
                df_detections = pd.read_csv(up_rep)
        with c_up2:
            up_frm = st.file_uploader("Resumo por Frame (frame_summary.csv):", type=["csv"], key="csv_frm")
            if up_frm is not None:
                df_frames = pd.read_csv(up_frm)

    # Se ainda estiver vazio, tenta ler os arquivos padrão de runs/inference
    if df_detections is None:
        default_csv = PROJECT_ROOT / "runs" / "inference" / "compliance_report.csv"
        if default_csv.exists():
            df_detections = load_compliance_report(default_csv)

    if df_frames is None:
        default_frames = PROJECT_ROOT / "runs" / "inference" / "frame_summary.csv"
        if default_frames.exists():
            df_frames = load_frame_summary(default_frames)

    kpis = calculate_kpis(
        df_detections=df_detections,
        df_frames=df_frames,
        audit_summary=audit_summary,
    )

    # Exibição do Badge de Status
    badge_class = (
        "badge-success"
        if "Excelente" in kpis["status"]
        else ("badge-warning" if "Atenção" in kpis["status"] else "badge-danger")
    )
    st.markdown(
        f"<div class='status-badge {badge_class}'>Status da Auditoria: {kpis['status']}</div>",
        unsafe_allow_html=True,
    )

    # Cards de KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Inspeções Registradas", kpis["total_detections"])
    c2.metric("Com Capacete (Conforme)", kpis["helmet_count"])
    c3.metric("Sem Capacete (Infração)", kpis["head_count"])

    if kpis["has_tracking"]:
        c4.metric(
            "Taxa Real por Trabalhador Único",
            f"{kpis['unique_compliance_rate_percent']:.1f}%",
            help="Calculada desduplicando indivíduos rastreados via MOT (ByteTrack).",
        )
    else:
        c4.metric("Taxa Geral de Conformidade", f"{kpis['compliance_rate_percent']:.1f}%")

    st.markdown("---")

    # Gráficos em Grade (2x2)
    g1, g2 = st.columns(2)
    with g1:
        fig_donut = create_compliance_donut_chart(
            conformant_count=kpis["helmet_count"],
            violation_count=kpis["head_count"],
            title="Taxa Geral de Uso de EPI",
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with g2:
        if df_frames is not None and not df_frames.empty:
            fig_temporal = create_temporal_series_chart(
                df_frames=df_frames,
                title="Série Temporal de Conformidade",
            )
            st.plotly_chart(fig_temporal, use_container_width=True)
        else:
            fig_temporal = create_temporal_series_chart(pd.DataFrame())
            st.plotly_chart(fig_temporal, use_container_width=True)

    g3, g4 = st.columns(2)
    with g3:
        if df_detections is not None and not df_detections.empty:
            fig_conf = create_confidence_distribution_chart(
                df_detections=df_detections,
                title="Distribuição de Confiança das Predições",
            )
            st.plotly_chart(fig_conf, use_container_width=True)
        else:
            st.plotly_chart(create_confidence_distribution_chart(pd.DataFrame()), use_container_width=True)

    with g4:
        if df_detections is not None and not df_detections.empty:
            fig_cam = create_camera_ranking_chart(
                df_detections=df_detections,
                title="Conformidade por Câmera / Área",
            )
            st.plotly_chart(fig_cam, use_container_width=True)
        else:
            st.plotly_chart(create_camera_ranking_chart(pd.DataFrame()), use_container_width=True)

    # Tabela Interativa de Ocorrências
    if df_detections is not None and not df_detections.empty:
        st.markdown("### 📋 Registro Consolidado de Ocorrências")
        st.dataframe(df_detections, use_container_width=True)


def main() -> None:
    """Função orquestradora da aplicação Streamlit."""
    st.markdown("<div class='main-title'>🛡️ EPI Finder - Monitoramento de SST & CIPA</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-title'>Plataforma de Visão Computacional para detecção de conformidade de capacetes em tempo real</div>",
        unsafe_allow_html=True,
    )

    config = render_sidebar()

    # Carrega modelo YOLO em cache
    try:
        model = get_cached_model(config["weights_path"])
    except Exception as exc:
        st.error(f"Erro ao carregar o modelo YOLO: {exc}")
        return

    tab_images, tab_videos, tab_webcam, tab_analytics = st.tabs([
        "📸 Detecção em Imagens",
        "🎥 Vídeos & Rastreamento (MOT)",
        "📷 Teste com Webcam (Estático)",
        "📊 Dashboard Analítico SST & CIPA",
    ])

    with tab_images:
        render_tab_images(model, config)

    with tab_videos:
        render_tab_videos(model, config)

    with tab_webcam:
        render_tab_webcam(model, config)

    with tab_analytics:
        render_tab_analytics()


if __name__ == "__main__":
    main()
