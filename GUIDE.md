# 🛡️ Guia Completo: EPI Finder - Detector de Uso de Capacete

Este guia foi elaborado para orientar você, passo a passo, no desenvolvimento de um projeto prático de **Detecção de Objetos** focado na identificação de uso de **Capacete de Segurança** (`head` - sem capacete vs `helmet` - com capacete) em imagens e vídeos.

---

## 📌 Visão Geral do Projeto

* **Problema:** Identificar automaticamente se trabalhadores e pessoas em áreas de risco / construção civil estão utilizando ou não o capacete de proteção obrigatório.
* **Classes do Modelo (2 classes):**
  1. `0: head` (sem capacete / cabeça desprotegida / infração)
  2. `1: helmet` (com capacete de proteção / conforme)
* **Tecnologias Principais:** Python, OpenCV, PyTorch / YOLOv8 (Ultralytics), Roboflow / CVAT, NumPy, Pandas, Matplotlib.
* **Resultado Esperado:** Um modelo treinado e um pipeline em Python capaz de receber fotos ou vídeos de câmeras de segurança, desenhar *bounding boxes* coloridas (verde para seguro, vermelho para infração) e gerar relatórios de conformidade.

---

## 🗺️ Roadmap: Fases e Etapas

```
[ Fase 1: Ambiente & Estrutura ] ➔ [ Fase 2: Dataset & Rotulagem ] ➔ [ Fase 3: Pipeline & data.yaml ]
                                                                             │
[ Fase 6: Aplicação & Inferência ] ⬅ [ Fase 5: Avaliação & Métricas ] ⬅ [ Fase 4: Treinamento YOLO ]
```

---

## ⚙️ Fase 1: Preparação do Ambiente e Estrutura do Projeto

### Etapa 1.1: Executando com Docker (Recomendado)
Para rodar todo o ambiente de forma isolada, padronizada e com o **Jupyter Lab** pronto:

```bash
# 1. Construir a imagem e subir o container
docker compose up --build

# 2. Acessar o Jupyter Lab no navegador
# Abra http://localhost:8888 no seu navegador!
```

Para rodar scripts Python diretamente dentro do container:
```bash
docker compose exec epi-finder python src/inference.py
```

---

### *(Alternativa sem Docker: Ambiente Virtual Local)*
<details>
<summary>Clique para ver como rodar com venv</summary>

```bash
# Criar e ativar o ambiente virtual
python3 -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt
```
</details>

---

### Etapa 1.2: Organização de Pastas
Mantenha uma estrutura limpa e profissional:

```text
epi_finder/
├── data/
│   ├── raw/               # Imagens originais baixadas
│   ├── dataset/           # Dataset organizado em train/val/test
│   │   ├── train/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   ├── valid/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   └── test/
│   │       ├── images/
│   │       └── labels/
│   └── data.yaml          # Arquivo de configuração de classes e caminhos
├── docs/                  # Documentação do projeto
│   ├── DESCRIPTION.md
│   └── GUIDE.md
├── notebooks/             # Jupyter Notebooks para EDA e treino
│   ├── 01_eda_dataset.ipynb
│   └── 02_training_yolov8.ipynb
├── src/                   # Scripts reutilizáveis
│   ├── __init__.py
│   ├── utils.py           # Funções auxiliares (desenho, IoU com NumPy)
│   └── inference.py       # Script para rodar predições em fotos/vídeos
├── models/                # Pesos salvos (.pt / .onnx)
├── Dockerfile             # Configuração da imagem Docker
├── docker-compose.yml     # Orquestração do container
├── .dockerignore
└── requirements.txt
```


---

## 🏷️ Fase 2: Coleta, Seleção e Rotulagem de Dados

### Etapa 2.1: Definição das Classes
Focaremos em um problema binário bem definido:
* `0`: `head` (pessoa sem capacete / cabeça desprotegida — infração)
* `1`: `helmet` (pessoa usando capacete de proteção — seguro)

> **Vantagem de focar em 2 classes:** É mais fácil e rápido rotular, exige menos volume de dados para convergir e atinge maior precisão rapidamente.

### Etapa 2.2: Coleta de Imagens
* Obtenção de dataset balanceado via Roboflow Universe (**946 imagens**, **4.131 anotações**).
* Distribuição representativa com canteiros de obra, indústrias, trabalhadores em diferentes distâncias e ângulos.

### Etapa 2.3: Rotulagem (Anotação)
Utilize uma ferramenta de anotação:
* **Roboflow** *(Recomendado)*: Crie um projeto Object Detection, rotule as caixas ao redor da cabeça/capacete e exporte no formato **YOLOv8**.
* **CVAT / LabelImg**: Alternativas locais.

**O que é o Formato YOLO?**
Para cada imagem `foto_001.jpg`, existirá um arquivo `foto_001.txt` com:
```text
<class_id> <x_center> <y_center> <width> <height>
```
*Valores normalizados entre 0.0 e 1.0.*

### Etapa 2.4: Divisão do Dataset
* **Treino (Train):** 87.8% (831 imagens | 3.666 caixas)
* **Validação (Valid):** 8.0% (76 imagens | 337 caixas)
* **Teste (Test):** 4.1% (39 imagens | 128 caixas)

---

## 🔄 Fase 3: Pipeline de Dados e Pré-processamento (Prática com NumPy e Pandas)

### Etapa 3.1: Análise Exploratória do Dataset com Pandas (EDA)
No notebook `01_eda_dataset.ipynb`, carregue as anotações para auditar a qualidade:
* Carregar os arquivos `.txt` em um `pandas.DataFrame`: `[image, class_id, class_name, x_center, y_center, width, height]`.
* **Análises:**
  * Verificar proporção entre `head` e `helmet` (`df['class_name'].value_counts()`).
  * Calcular áreas das caixas (`df['area'] = df['width'] * df['height']`) para ver se há muitos objetos muito pequenos ou distantes.

### Etapa 3.2: Manipulação de Imagens e Vetorização com NumPy
* **Recorte de Região de Interesse (ROI):** Praticar recortes da cabeça usando indexação de matrizes: `crop = img[y1:y2, x1:x2]`.
* **Canais de Cores e Normalização:** Conversões BGR/RGB via slicing (`img[:, :, ::-1]`) e escala para `float32` [0.0, 1.0].
* **Implementação do IoU:** Escrever a função matemática de Intersection over Union do zero usando funções vetorizadas do NumPy (`np.maximum`, `np.minimum`).

### Etapa 3.3: Configuração do `data.yaml`
Crie o arquivo `data/data.yaml`:

```yaml
path: ../data/dataset
train: train/images
val: valid/images
test: test/images

names:
  0: head
  1: helmet
```

---

## 🚀 Fase 4: Treinamento e Experimentos

### Etapa 4.1: Modelo Base e Transfer Learning
Usaremos pesos pré-treinados do **YOLOv8n (Nano)** para acelerar o aprendizado:

```python
from ultralytics import YOLO

# Carregar modelo pré-treinado
model = YOLO("yolov8n.pt")

# Iniciar treinamento
results = model.train(
    data="data/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,         # 'cpu' ou 0 para GPU NVIDIA
    name="helmet_detector_exp1"
)
```

### Etapa 4.2: Monitoramento de Perdas
Acompanhe os gráficos em `runs/detect/helmet_detector_exp1/`:
* `box_loss` (localização da caixa)
* `cls_loss` (classificação correta entre `head` e `helmet`)
* `dfl_loss`

---

## 📊 Fase 5: Avaliação, Métricas e Análise com Pandas

### Etapa 5.1: Entendendo as Métricas
* **Precision:** Das detecções de `head`, quantas eram realmente pessoas sem capacete? (evita falsos alarmes).
* **Recall:** De todas as pessoas sem capacete na imagem, quantas foram pegas pelo modelo? (essencial para segurança do trabalho).
* **$mAP@50$ e $mAP@50\text{-}95$:** Desempenho geral do detector.

### Etapa 5.2: Análise do `results.csv` com Pandas
* Ler o arquivo de histórico de épocas: `df_results = pd.read_csv("runs/detect/helmet_detector_exp1/results.csv")`.
* Filtrar a época com maior $mAP@50$ e plotar evolução da acurácia.

### Etapa 5.3: Validação no Conjunto de Teste
```python
model = YOLO("runs/detect/helmet_detector_exp1/weights/best.pt")
metrics = model.val(data="data/data.yaml", split="test")

print(f"mAP@50: {metrics.box.map50}")
```

---

## 🎥 Fase 6: Aplicação de Inferência e Relatórios (NumPy + Pandas)

### Etapa 6.1: Script de Inferência com Cores Dinâmicas
Em `src/inference.py`:
* Se a classe for `head` (0) ➔ desenhar caixa **Vermelha** (`(0, 0, 255)`) com texto `"ALERTA: SEM CAPACETE"`.
* Se a classe for `helmet` (1) ➔ desenhar caixa **Verde** (`(0, 255, 0)`) com texto `"Capacete"`.

```python
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("runs/detect/helmet_detector_exp1/weights/best.pt")
results = model("caminho/para/imagem.jpg")

for r in results:
    boxes = r.boxes.xyxy.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy().astype(int)
    confs = r.boxes.conf.cpu().numpy()
    
    img = r.orig_img.copy()
    for box, cls, conf in zip(boxes, classes, confs):
        x1, y1, x2, y2 = map(int, box)
        color = (0, 0, 255) if cls == 0 else (0, 255, 0)
        label = f"{'ALERTA: SEM CAPACETE' if cls == 0 else 'Capacete'} {conf:.2f}"
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("EPI Finder - Detector de Capacete", img)
    cv2.waitKey(0)
```

### Etapa 6.2: Registro de Infrações com Pandas
Ao processar vídeos contínuos de câmeras:
* Salvar cada ocorrência de `head` em uma lista: `{'timestamp': ..., 'confianca': ..., 'camera': 'Entrada 1'}`.
* Exportar ao final do dia um relatório de conformidade em `.csv` usando `pd.DataFrame.to_csv()`.

---

## 🏆 Fase 7: Próximos Passos (Extensões Opcionais)

1. **Dashboard com Streamlit:** Interface web onde o usuário faz upload de um vídeo e vê a contagem de conformidade em tempo real.
2. **Rastreamento de Pessoas (Tracking):** Adicionar `model.track(source="video.mp4")` para não contar a mesma pessoa sem capacete múltiplas vezes no mesmo vídeo.

---

## 📝 Checklist do Projeto

1. [x] Criar ambiente virtual / container Docker (`docker-compose up --build`).
2. [x] Coletar dataset de pessoas com e sem capacete de segurança (Roboflow Universe).
3. [x] Obter anotações das classes (`head` / `helmet`) via Roboflow.
4. [x] Exportar dataset em formato YOLOv8 e descompactar em `data/dataset/`.
5. [x] Configurar `data/data.yaml` e executar Análise Exploratória (EDA) com Pandas/NumPy.
6. [ ] Executar primeiro treino de baseline com `yolov8n.pt`.
7. [ ] Validar métricas e construir o script de inferência com alertas visuais (`src/inference.py`).

