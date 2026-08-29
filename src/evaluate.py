"""
Módulo de avaliação e métricas para o detector de capacete (EPI Finder) com YOLOv8.

Permite a auditoria formal e padronizada do modelo treinado em conjuntos de validação
ou teste cego ('val' ou 'test'), gerando relatórios estatísticos detalhados por classe,
análises de risco para Segurança do Trabalho (SST) e exportação em formato JSON/tabelar.
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional

import torch
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset


def get_default_device() -> str:
    """
    Identifica dinamicamente o dispositivo mais adequado (CUDA GPU ou CPU).

    Returns:
        '0' se GPU com CUDA estiver disponível, senão 'cpu'.
    """
    return "0" if torch.cuda.is_available() else "cpu"


def calculate_f1(precision: float, recall: float) -> float:
    """
    Calcula a média harmônica entre Precision e Recall (F1-Score).

    Args:
        precision: Valor de precisão entre 0.0 e 1.0.
        recall: Valor de revocação entre 0.0 e 1.0.

    Returns:
        Score F1 no intervalo [0.0, 1.0].
    """
    if (precision + recall) <= 0.0:
        return 0.0
    return 2.0 * (precision * recall) / (precision + recall)


def evaluate_yolo(
    weights: str = "models/best.pt",
    data_yaml: str = "data/data.yaml",
    split: str = "test",
    img_size: int = 640,
    batch_size: Optional[int] = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.60,
    device: Optional[str] = None,
    save_json: Optional[str] = "models/test_metrics.json",
    project_dir: Optional[str] = None,
    experiment_name: str = "evaluate",
    save_plots: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Executa a avaliação formal do modelo YOLOv8 na partição de dados especificada.

    Args:
        weights: Caminho relativo ou absoluto para os pesos treinados (.pt).
        data_yaml: Caminho para o arquivo de configuração do dataset (data.yaml).
        split: Partição do dataset a ser avaliada ('test' ou 'val').
        img_size: Resolução quadrada para redimensionamento das imagens de entrada.
        batch_size: Quantidade de amostras por lote. Se None, define 16 (GPU) ou 8 (CPU).
        conf_threshold: Limiar mínimo de confiança para considerar uma predição.
        iou_threshold: Limiar de IoU utilizado no algoritmo Non-Maximum Suppression (NMS).
        device: Dispositivo de execução ('cpu', '0', etc.). Se None, detecta automaticamente.
        save_json: Caminho para exportar o resumo estruturado em JSON (ou None para não salvar).
        project_dir: Diretório raiz para armazenamento dos artefatos da avaliação.
        experiment_name: Nome do diretório do experimento de avaliação.
        save_plots: Se True, gera e salva matrizes de confusão e curvas PR na pasta de saída.
        verbose: Se True, imprime relatórios formatados no stdout.

    Returns:
        Dicionário completo com métricas globais, por classe e diagnósticos operacionais.
    """
    weights_path = Path(weights).resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"Arquivo de pesos não encontrado: {weights_path}")

    yaml_path = Path(data_yaml).resolve()
    if not yaml_path.exists():
        raise FileNotFoundError(f"Arquivo de configuração do dataset não encontrado: {yaml_path}")

    active_device = device if device is not None else get_default_device()

    if batch_size is None:
        batch_size = 16 if active_device not in ("cpu", "") else 8

    if verbose:
        print("=" * 78)
        print(f"🛡️  EPI FINDER - AVALIAÇÃO DE MODELO (FASE 5: MÉTRICAS & SST)")
        print("=" * 78)
        print(f"📦 Pesos Avaliados : {weights_path}")
        print(f"📋 Dataset Config  : {yaml_path}")
        print(f"🎯 Partição (Split): {split.upper()} (Validação Cega)")
        print(f"📐 Resolução Img   : {img_size}x{img_size}")
        print(f"📦 Batch Size      : {batch_size}")
        print(f"🎯 Limiares        : Confiança={conf_threshold} | IoU (NMS)={iou_threshold}")
        print(f"⚡ Dispositivo     : {active_device} ({'GPU CUDA' if active_device != 'cpu' else 'CPU'})")
        print("=" * 78)

    # Validar informações do dataset
    dataset_info = check_det_dataset(str(yaml_path))
    class_names_map = dataset_info.get("names", {0: "head", 1: "helmet"})

    # Carregar modelo
    if verbose:
        print(f"\n📥 Carregando modelo YOLO a partir de '{weights_path.name}'...")
    model = YOLO(str(weights_path))

    # Configuração dos argumentos para model.val
    val_args: Dict[str, Any] = {
        "data": str(yaml_path),
        "split": split,
        "imgsz": img_size,
        "batch": batch_size,
        "conf": conf_threshold,
        "iou": iou_threshold,
        "device": active_device,
        "plots": save_plots,
        "save_json": False,
        "verbose": False,
    }
    if project_dir is not None:
        val_args["project"] = project_dir
        val_args["name"] = experiment_name

    if verbose:
        print(f"🚀 Executando inferência e cômputo de métricas no split '{split}'...")
    metrics = model.val(**val_args)

    # Extração de Métricas Globais
    global_map50 = float(metrics.box.map50)
    global_map50_95 = float(metrics.box.map)
    global_precision = float(metrics.box.mp)
    global_recall = float(metrics.box.mr)
    global_f1 = calculate_f1(global_precision, global_recall)

    # Extração de Métricas por Classe
    classes_metrics: Dict[str, Dict[str, Any]] = {}
    class_indices = metrics.box.ap_class_index if hasattr(metrics.box, "ap_class_index") else []

    for i, c_idx in enumerate(class_indices):
        c_idx_int = int(c_idx)
        c_name = class_names_map.get(c_idx_int, f"classe_{c_idx_int}")
        
        p = float(metrics.box.p[i]) if i < len(metrics.box.p) else 0.0
        r = float(metrics.box.r[i]) if i < len(metrics.box.r) else 0.0
        f1 = float(metrics.box.f1[i]) if hasattr(metrics.box, "f1") and i < len(metrics.box.f1) else calculate_f1(p, r)
        ap50 = float(metrics.box.ap50[i]) if i < len(metrics.box.ap50) else 0.0
        ap50_95 = float(metrics.box.ap[i]) if i < len(metrics.box.ap) else 0.0

        classes_metrics[str(c_idx_int)] = {
            "class_id": c_idx_int,
            "class_name": c_name,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1_score": round(f1, 4),
            "map50": round(ap50, 4),
            "map50_95": round(ap50_95, 4),
        }

    # Análise da Matriz de Confusão
    confusion_matrix_data: Optional[Dict[str, Any]] = None
    if hasattr(metrics, "confusion_matrix") and metrics.confusion_matrix is not None:
        try:
            raw_matrix = metrics.confusion_matrix.matrix.tolist()
            confusion_matrix_data = {
                "matrix": raw_matrix,
                "description": "Matriz com shape (nc+1, nc+1) incluindo coluna e linha de background.",
            }
        except Exception:
            confusion_matrix_data = None

    # Montagem do relatório consolidado
    report: Dict[str, Any] = {
        "evaluation_date": datetime.now().isoformat(),
        "model_weights": str(weights_path.relative_to(Path.cwd())) if weights_path.is_relative_to(Path.cwd()) else str(weights_path),
        "dataset_yaml": str(yaml_path.relative_to(Path.cwd())) if yaml_path.is_relative_to(Path.cwd()) else str(yaml_path),
        "split_evaluated": split,
        "hyperparameters": {
            "img_size": img_size,
            "conf_threshold": conf_threshold,
            "iou_threshold": iou_threshold,
            "batch_size": batch_size,
            "device": active_device,
        },
        "global_metrics": {
            "precision": round(global_precision, 4),
            "recall": round(global_recall, 4),
            "f1_score": round(global_f1, 4),
            "map50": round(global_map50, 4),
            "map50_95": round(global_map50_95, 4),
        },
        "class_metrics": classes_metrics,
        "speed_ms": {
            "preprocess": round(float(metrics.speed.get("preprocess", 0.0)), 2),
            "inference": round(float(metrics.speed.get("inference", 0.0)), 2),
            "loss": round(float(metrics.speed.get("loss", 0.0)), 2),
            "postprocess": round(float(metrics.speed.get("postprocess", 0.0)), 2),
        },
        "artifacts_dir": str(metrics.save_dir) if hasattr(metrics, "save_dir") else None,
    }

    if confusion_matrix_data:
        report["confusion_matrix"] = confusion_matrix_data

    # Diagnóstico Específico para Segurança do Trabalho (SST)
    head_metrics = classes_metrics.get("0")
    helmet_metrics = classes_metrics.get("1")
    sst_analysis: Dict[str, str] = {}

    if head_metrics:
        head_r = head_metrics["recall"]
        head_p = head_metrics["precision"]
        if head_r < 0.70:
            sst_analysis["risco_infracao"] = (
                f"ALERTA CRÍTICO: Recall de 'head' em {head_r:.1%}. "
                "Mais de 30% das infrações reais não estão sendo identificadas pelo modelo. "
                "Priorizar aumento de dados de infração ou ajuste de limiar de confiança."
            )
        else:
            sst_analysis["risco_infracao"] = (
                f"ADEQUADO: Recall de 'head' em {head_r:.1%}. O modelo captura a maioria dos trabalhadores desprotegidos."
            )

        if head_p < 0.60:
            sst_analysis["risco_alarme_falso"] = (
                f"ATENÇÃO: Precision de 'head' em {head_p:.1%}. Taxa considerável de alarmes falsos "
                "pode causar fadiga de alertas na equipe de monitoramento da obra."
            )
        else:
            sst_analysis["risco_alarme_falso"] = (
                f"EXCELENTE: Precision de 'head' em {head_p:.1%}. Alta confiabilidade nas infrações apontadas."
            )

    report["sst_audit"] = sst_analysis

    # Exibição formatada no terminal
    if verbose:
        print("\n" + "=" * 78)
        print("📊 TABELA CONSOLIDADA DE MÉTRICAS (SPLIT: " + split.upper() + ")")
        print("=" * 78)
        header = f"{'Classe':<18} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'mAP@50':<10} | {'mAP@50-95':<10}"
        print(header)
        print("-" * len(header))

        for c_id, c_data in classes_metrics.items():
            line = (
                f"{c_data['class_name'] + ' (' + str(c_id) + ')':<18} | "
                f"{c_data['precision']:<10.4f} | "
                f"{c_data['recall']:<10.4f} | "
                f"{c_data['f1_score']:<10.4f} | "
                f"{c_data['map50']:<10.4f} | "
                f"{c_data['map50_95']:<10.4f}"
            )
            print(line)

        print("-" * len(header))
        summary_line = (
            f"{'MÉDIA GERAL (ALL)':<18} | "
            f"{global_precision:<10.4f} | "
            f"{global_recall:<10.4f} | "
            f"{global_f1:<10.4f} | "
            f"{global_map50:<10.4f} | "
            f"{global_map50_95:<10.4f}"
        )
        print(summary_line)
        print("=" * 78)

        print("\n⏱️  Velocidade de Processamento por Imagem:")
        print(f"   - Pré-processamento : {report['speed_ms']['preprocess']} ms")
        print(f"   - Inferência da Rede: {report['speed_ms']['inference']} ms")
        print(f"   - Pós-processamento : {report['speed_ms']['postprocess']} ms")

        if sst_analysis:
            print("\n🦺 Diagnóstico para Segurança do Trabalho (SST):")
            for k, msg in sst_analysis.items():
                print(f"   • {msg}")

        if report["artifacts_dir"]:
            print(f"\n🖼️  Gráficos e matrizes salvos em: {report['artifacts_dir']}")

    # Salvamento de arquivo JSON
    if save_json:
        json_path = Path(save_json).resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        if verbose:
            print(f"📄 Relatório de auditoria salvo em: {json_path}")

    return report


def parse_args() -> argparse.Namespace:
    """Configura e processa os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="EPI Finder - Avaliação e Auditoria de Métricas YOLOv8 (Fase 5)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="models/best.pt",
        help="Caminho para o arquivo de pesos treinado (.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/data.yaml",
        help="Caminho relativo ou absoluto para o arquivo data.yaml",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val"],
        help="Partição do dataset a ser avaliada ('test' para validação cega ou 'val')",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Resolução das imagens de entrada em pixels",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Tamanho do lote (batch size). Se omitido, define 16 (GPU) ou 8 (CPU)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Limiar mínimo de confiança para considerar detecções",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.60,
        help="Limiar de IoU para supressão de não-máximos (NMS)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Dispositivo ('0' para GPU CUDA ou 'cpu'). Se omitido, detecta automaticamente",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default="models/test_metrics.json",
        help="Caminho para salvar o relatório consolidado em formato JSON",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Desativa a geração de gráficos visuais (matriz de confusão, PR curve)",
    )
    return parser.parse_args()


def main() -> None:
    """Ponto de entrada do script CLI."""
    args = parse_args()
    try:
        evaluate_yolo(
            weights=args.weights,
            data_yaml=args.data,
            split=args.split,
            img_size=args.imgsz,
            batch_size=args.batch,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            device=args.device,
            save_json=args.save_json,
            save_plots=not args.no_plots,
            verbose=True,
        )
    except Exception as e:
        print(f"\n❌ Erro durante a avaliação: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
