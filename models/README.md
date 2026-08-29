# 📦 Modelos Treinados - EPI Finder

Este diretório centraliza os pesos treinados (*checkpoints*) e os metadados de auditoria do detector de uso de capacete com YOLOv8.

---

## 📁 Estrutura de Arquivos

| Arquivo | Descrição |
|---|---|
| `best.pt` | Checkpoint do modelo que obteve o maior $mAP@50$ no conjunto de validação durante o treinamento. Este é o arquivo padrão carregado para inferência em produção nas **Fases 5 e 6**. |
| `last.pt` | Estado dos pesos na última época de treinamento concluída. Utilizado para retomada de treino (*resume*) ou comparação de convergência. |
| `<nome_experimento>_best.pt` | Cópia versionada dos melhores pesos vinculada ao nome específico do experimento executado. |
| `metadata.json` | Registro dos parâmetros de treino, hiperparâmetros, classes e métricas oficiais ($mAP@50$, $mAP@50\text{-}95$, Precision, Recall). |

---

## 🔍 Como Carregar os Pesos para Inferência em Python

```python
from ultralytics import YOLO

# Carrega o melhor modelo treinado
model = YOLO("models/best.pt")

# Executa inferência em imagem ou vídeo
results = model.predict(source="data/dataset/test/images/sample.jpg", conf=0.25)
```

---

> ℹ️ **Nota sobre Controle de Versão:** Arquivos de pesos binários (`*.pt`) são automaticamente ignorados pelo `.gitignore` para não onerar o repositório Git, enquanto a estrutura e os metadados permanecem organizados.
