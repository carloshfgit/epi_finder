"""
Módulo de treinamento para o detector de capacete (EPI Finder) com YOLOv8.

Permite a execução modular ou via linha de comando (CLI) do treinamento de
Transfer Learning com a arquitetura YOLOv8 da Ultralytics.
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, Optional

import pandas as pd
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


def train_yolo(
    data_yaml: str = "data/data.yaml",
    base_model: str = "yolov8n.pt",
    epochs: int = 50,
    batch_size: Optional[int] = None,
    img_size: int = 640,
    device: Optional[str] = None,
    experiment_name: str = "helmet_detector_exp1",
    patience: int = 15,
    workers: Optional[int] = None,
    project_dir: Optional[str] = None,
    copy_to_models: bool = True,
    exist_ok: bool = True,
    verbose: bool = True,
    fraction: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Executa o ciclo de treinamento do YOLOv8 com Transfer Learning.

    Args:
        data_yaml: Caminho para o arquivo data.yaml com as configurações do dataset.
        base_model: Nome ou caminho do modelo base pré-treinado (ex: 'yolov8n.pt').
        epochs: Número total de épocas de treinamento.
        batch_size: Quantidade de amostras por lote. Se None, define 16 (GPU) ou 8 (CPU).
        img_size: Resolução quadrada para redimensionamento das imagens de entrada.
        device: Dispositivo de execução ('cpu', '0', 'cuda', etc.). Se None, detecta automaticamente.
        experiment_name: Nome do diretório do experimento onde os pesos e gráficos serão salvos.
        patience: Quantidade de épocas sem melhoria no mAP para disparar Early Stopping.
        workers: Número de subprocessos do DataLoader para carregar dados.
        project_dir: Diretório raiz onde os experimentos são armazenados.
        copy_to_models: Se True, copia uma duplicata dos pesos 'best.pt' para 'models/'.
        exist_ok: Se True, permite sobrescrever/reutilizar pasta de experimento existente.
        verbose: Se True, imprime informações detalhadas no stdout durante a execução.

    Returns:
        Dicionário contendo os caminhos dos pesos salvos e o objeto de resultados do Ultralytics.
    """
    # Resolução de caminhos com base no diretório atual
    yaml_path = Path(data_yaml).resolve()
    if not yaml_path.exists():
        raise FileNotFoundError(f"Arquivo de dataset não encontrado: {yaml_path}")

    # Detecção automática de dispositivo
    active_device = device if device is not None else get_default_device()

    # Ajuste automático do tamanho de batch se não especificado
    if batch_size is None:
        batch_size = 16 if active_device not in ("cpu", "") else 8

    # Ajuste de workers
    if workers is None:
        workers = 4 if active_device not in ("cpu", "") else min(2, os.cpu_count() or 1)

    print("=" * 70)
    print("🛡️  EPI FINDER - INICIANDO TREINAMENTO YOLOv8")
    print("=" * 70)
    print(f"📁 Dataset Config : {yaml_path}")
    print(f"🧠 Modelo Base    : {base_model}")
    print(f"🔄 Épocas         : {epochs}")
    print(f"📦 Batch Size     : {batch_size}")
    print(f"📐 Resolução      : {img_size}x{img_size}")
    print(f"⚡ Dispositivo    : {active_device} ({'GPU CUDA' if active_device != 'cpu' else 'CPU'})")
    print(f"⏱️  Patience       : {patience} épocas (Early Stopping)")
    print(f"🧵 Workers        : {workers}")
    print(f"📂 Experimento    : {project_dir}/{experiment_name}")
    print("=" * 70)

    # Valida estrutura do dataset via Ultralytics
    print("\n🔍 Validando estrutura do dataset...")
    dataset_info = check_det_dataset(str(yaml_path))
    print(f"✅ Classes mapeadas: {dataset_info.get('names')}")

    # Instancia o modelo YOLO
    print(f"\n📥 Carregando modelo base '{base_model}'...")
    model = YOLO(base_model)

    # Parâmetros de treino
    train_kwargs = {
        "data": str(yaml_path),
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": img_size,
        "device": active_device,
        "name": experiment_name,
        "patience": patience,
        "workers": workers,
        "exist_ok": exist_ok,
        "verbose": verbose,
        "save": True,
        "fraction": fraction if fraction is not None else 1.0,
    }
    if project_dir is not None:
        train_kwargs["project"] = project_dir

    # Dispara o treinamento
    print(f"\n🚀 Iniciando loop de treinamento ({epochs} épocas)...")
    results = model.train(**train_kwargs)

    # Identifica o diretório exato de saída através de results.save_dir
    exp_dir = Path(results.save_dir) if hasattr(results, "save_dir") else Path("runs/detect") / experiment_name
    best_weights_path = exp_dir / "weights" / "best.pt"
    last_weights_path = exp_dir / "weights" / "last.pt"

    output_summary = {
        "experiment_dir": str(exp_dir),
        "best_weights": str(best_weights_path) if best_weights_path.exists() else None,
        "last_weights": str(last_weights_path) if last_weights_path.exists() else None,
        "results": results,
    }

    print("\n" + "=" * 70)
    print("🎉 TREINAMENTO CONCLUÍDO COM SUCESSO!")
    print("=" * 70)
    if best_weights_path.exists():
        print(f"🏆 Melhores pesos salvos em : {best_weights_path}")
    if last_weights_path.exists():
        print(f"💾 Últimos pesos salvos em  : {last_weights_path}")

    # Cópia e organização profissional para models/
    if copy_to_models and best_weights_path.exists():
        models_dir = Path("models").resolve()
        models_dir.mkdir(parents=True, exist_ok=True)

        dest_best = models_dir / "best.pt"
        dest_named = models_dir / f"{experiment_name}_best.pt"
        shutil.copy2(best_weights_path, dest_best)
        shutil.copy2(best_weights_path, dest_named)

        dest_last = models_dir / "last.pt"
        if last_weights_path.exists():
            shutil.copy2(last_weights_path, dest_last)

        print("📦 Pesos organizados e exportados para models/:")
        print(f"   - {dest_best} (Melhor modelo para inferência/deploy)")
        print(f"   - {dest_named}")
        if dest_last.exists():
            print(f"   - {dest_last} (Último checkpoint para retomada)")
        output_summary["exported_best"] = str(dest_best)

        # Extração de métricas da melhor época via results.csv
        results_csv = exp_dir / "results.csv"
        metadata: Dict[str, Any] = {
            "model_name": experiment_name,
            "base_model": base_model,
            "training_date": datetime.now().isoformat(),
            "epochs_configured": epochs,
            "batch_size": batch_size,
            "image_size": img_size,
            "device": active_device,
            "classes": dataset_info.get("names", {}),
            "weights": {
                "best": str(dest_best.relative_to(Path.cwd())) if dest_best.is_relative_to(Path.cwd()) else str(dest_best),
                "last": str(dest_last.relative_to(Path.cwd())) if dest_last.exists() and dest_last.is_relative_to(Path.cwd()) else str(dest_last),
            },
        }

        if results_csv.exists():
            try:
                df = pd.read_csv(results_csv)
                df.columns = [c.strip() for c in df.columns]
                map50_col = "metrics/mAP50(B)" if "metrics/mAP50(B)" in df.columns else "metrics/mAP50"
                if map50_col in df.columns and not df.empty:
                    best_row_idx = df[map50_col].idxmax()
                    best_row = df.loc[best_row_idx]
                    metadata["best_epoch"] = int(best_row["epoch"])
                    metadata["best_metrics"] = {
                        "mAP50": float(best_row[map50_col]),
                        "mAP50-95": float(best_row.get("metrics/mAP50-95(B)", best_row.get("metrics/mAP50-95", 0.0))),
                        "precision": float(best_row.get("metrics/precision(B)", best_row.get("metrics/precision", 0.0))),
                        "recall": float(best_row.get("metrics/recall(B)", best_row.get("metrics/recall", 0.0))),
                    }
                    print(f"🏆 Melhor Época Registrada: {metadata['best_epoch']} com mAP@50 = {metadata['best_metrics']['mAP50']:.4f}")
            except Exception as meta_err:
                print(f"⚠️ Não foi possível extrair métricas de {results_csv}: {meta_err}")

        # Salva o arquivo metadata.json
        meta_file = models_dir / "metadata.json"
        meta_file.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"📄 Metadados do modelo salvos em : {meta_file}")
        output_summary["metadata_file"] = str(meta_file)

    results_csv = exp_dir / "results.csv"
    if results_csv.exists():
        print(f"📊 Histórico de métricas salvo em : {results_csv}")

    results_png = exp_dir / "results.png"
    if results_png.exists():
        print(f"📈 Gráficos de treinamento gerados: {results_png}")

    print("=" * 70)
    return output_summary


def parse_args() -> argparse.Namespace:
    """Configura e processa os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="EPI Finder - Script de Treinamento com YOLOv8 (Transfer Learning)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/data.yaml",
        help="Caminho relativo ou absoluto para o arquivo data.yaml do dataset",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Modelo base pré-treinado da Ultralytics (ex: yolov8n.pt, yolov8s.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Número total de épocas de treinamento",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Tamanho do lote (batch size). Se omitido, define 16 (GPU) ou 8 (CPU)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Tamanho das imagens de entrada em pixels",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Dispositivo de execução ('0' para GPU ou 'cpu'). Se omitido, detecta automaticamente",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="helmet_detector_exp1",
        help="Nome da pasta do experimento dentro de runs/detect/",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Número de épocas sem melhora para acionar o Early Stopping",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Número de threads de carregamento de dados (DataLoader workers)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Diretório raiz onde os experimentos serão salvos (padrão: 'runs/detect')",
    )
    parser.add_argument(
        "--no-copy-best",
        action="store_true",
        help="Se informado, desativa a cópia automática dos pesos 'best.pt' para a pasta 'models/'",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=None,
        help="Fração do dataset a utilizar (ex: 0.05 para 5%%), útil para testes rápidos/smoke tests",
    )
    return parser.parse_args()


def main() -> None:
    """Ponto de entrada do script CLI."""
    args = parse_args()
    try:
        train_yolo(
            data_yaml=args.data,
            base_model=args.model,
            epochs=args.epochs,
            batch_size=args.batch,
            img_size=args.imgsz,
            device=args.device,
            experiment_name=args.name,
            patience=args.patience,
            workers=args.workers,
            project_dir=args.project,
            copy_to_models=not args.no_copy_best,
            fraction=args.fraction,
        )
    except Exception as e:
        print(f"\n❌ Erro durante o treinamento: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
