# EPI Finder - Detector Inteligente de Capacete de Segurança

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00599C?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter_Lab-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

> **Nota de Propósito:** Este projeto foi desenvolvido para fins de **estudo, aprendizado prático e consolidação de conceitos** em **Visão Computacional**, **Deep Learning (YOLOv8)**, **Manipulação Matricial (NumPy)**, **Análise de Dados (Pandas)** e **Práticas de Engenharia de Machine Learning (MLOps)**.

---

## Visão Geral do Projeto

O **EPI Finder** é um sistema de detecção de objetos focado em Segurança e Saúde no Trabalho (SST), com ênfase na verificação de conformidade da norma **NR-6** (Equipamentos de Proteção Individual). A aplicação identifica automaticamente, a partir de imagens e transmissões de vídeo, se pessoas em canteiros de obra ou ambientes industriais estão ou não utilizando o capacete de segurança obrigatório.

### Mapeamento de Classes (Problema Binário):
* `0: head` (Sem capacete / cabeça desprotegida — **Infração / Alerta Vermelho**)
* `1: helmet` (Com capacete de proteção — **Conforme / Seguro Verde**)

> **Por que um modelo binário?** Focar especificamente em `head` vs `helmet` otimiza o balanceamento do dataset, agiliza o tempo de convergência do treinamento e permite máxima atenção à classe crítica de infração.

---

## 🧠 Tópicos Teóricos Estudados e Consolidados

Durante a concepção e implementação deste projeto, diversos conceitos matemáticos, computacionais e de engenharia de software foram explorados em profundidade:

### 1. Visão Computacional: Classificação vs. Detecção de Objetos
* **Classificação de Imagens:** Atribui um único rótulo para toda a imagem ("há um trabalhador nesta foto").
* **Detecção de Objetos (Object Detection):** Localiza simultaneamente **onde** cada objeto está no espaço 2D (por meio de *bounding boxes* delimitadas por coordenadas $[x_1, y_1, x_2, y_2]$) e classifica cada instância individualmente.

### 2. Arquiteturas YOLO (*You Only Look Once*) e Detecção Single-Stage
* Compreensão dos detectores de estágio único (*single-stage detectors*) em contraposição aos detectores de dois estágios (*two-stage*, ex: Faster R-CNN).
* O **YOLOv8** processa a imagem inteira em uma única passagem pela rede convolucional (*feedforward*), permitindo taxas de quadros elevadas necessárias para inferência em tempo real em fluxos de vídeo (CFTV).

### 3. Transfer Learning e Fine-Tuning
* Redes neurais profundas necessitam de dezenas de milhares de imagens se treinadas do zero.
* Com **Transfer Learning**, utilizamos os pesos pré-treinados da rede base **YOLOv8n (Nano)** no dataset COCO (80 classes gerais). A rede já possui filtros que reconhecem primitivas visuais (bordas, contornos, texturas e silhuetas humanas), restando apenas o ajuste fino das camadas finais para as classes `head` e `helmet`.

### 4. Componentes da Função de Perda (Loss Functions)
O YOLOv8 equilibra simultaneamente três funções de custo para guiar o gradiente:
1. **`box_loss` (Complete IoU - CIoU):** Penaliza discrepâncias geométricas entre a caixa prevista e a real, considerando sobreposição de área, distância euclidiana entre centros e proporção de aspecto ($W/H$).
2. **`cls_loss` (Binary Cross-Entropy - BCE):** Penaliza erros de classificação entre `head` e `helmet`.
3. **`dfl_loss` (Distribution Focal Loss):** Refina a regressão das bordas em caixas com limites difusos ou oclusões parciais.

### 5. Fundamentos Matemáticos das Métricas de Avaliação
* **IoU (Intersection over Union):**
  $$\text{IoU} = \frac{\text{Área da Interseção}}{\text{Área da União}} = \frac{B_{\text{pred}} \cap B_{\text{gt}}}{B_{\text{pred}} \cup B_{\text{gt}}}$$
* **Precision (Precisão):** Proporção de acertos dentre as caixas detectadas pela IA ($\frac{TP}{TP + FP}$). Evita falsos alarmes nos monitores de segurança.
* **Recall (Revocação / Sensibilidade):** Proporção de objetos reais capturados pelo modelo ($\frac{TP}{TP + FN}$). Essencial para não deixar nenhum trabalhador sem capacete passar despercebido.
* **Score F1:** Média harmônica balanceada entre Precision e Recall.
* **mAP@50 e mAP@50-95:** Precisão média sob limiares de IoU a $0.50$ e com variação progressiva de $0.50$ a $0.95$ com passo de $0.05$.

### 6. Matriz de Confusão com Classe de Fundo (*Background*)
* Na detecção de objetos, a matriz de confusão possui uma dimensão extra: a classe **Background (Fundo)**.
* **Falso Negativo:** Quando um objeto real não é detectado pelo modelo (cai na linha de *Background*).
* **Falso Positivo:** Quando o modelo cria uma caixa em um local onde não havia nada anotado (cai na coluna de *Background*).

### 7. Trade-off e Engenharia de Risco em SST
* Em aplicações industriais e canteiros de obra, **o Recall da classe `head` é prioritário sobre a Precision**.
* O custo de um Falso Negativo (um trabalhador desprotegido não visto pela IA) é um potencial acidente fatal e autuações legais.
* O custo de um Falso Positivo (um alarme falso em alguém de boné ou capuz) é apenas um incômodo operacional. Portanto, o sistema deve ser calibrado com limiares de corte que maximizem o Recall de infrações.

---

## Estrutura do Repositório

O projeto segue boas práticas de engenharia de software e ciência de dados, com separação estrita de responsabilidades:

```text
epi_finder/
├── data/
│   ├── dataset/              # Dataset particionado (Roboflow Universe)
│   │   ├── train/            # 831 imagens (3.666 anotações - 87,8%)
│   │   ├── valid/            # 76 imagens  (337 anotações  - 8,0%)
│   │   └── test/             # 39 imagens  (128 anotações  - 4,1% - validação cega)
│   └── data.yaml             # Arquivo de configuração de classes e caminhos
├── notebooks/                # Cadernos interativos de estudo e experimentação
│   ├── 01_eda_dataset.ipynb         # Fase 3: EDA com Pandas & manipulação com NumPy
│   ├── 02_training_yolov8.ipynb     # Fase 4: Treinamento e curvas de aprendizado
│   └── 03_model_evaluation.ipynb   # Fase 5: Validação no teste cego e diagnósticos SST
├── src/                      # Código modular e ferramentas CLI reutilizáveis
│   ├── __init__.py
│   ├── utils.py              # Cálculo de IoU matricial, conversão de caixas e OpenCV
│   ├── train.py              # CLI para treinamento modular e versionamento de pesos
│   └── evaluate.py           # CLI para auditoria de métricas e exportação em JSON
├── models/                   # Centralização de modelos e auditoria
│   ├── best.pt               # Melhores pesos treinados (.pt)
│   ├── metadata.json         # Certidão de nascimento e hiperparâmetros do modelo
│   └── test_metrics.json     # Relatório consolidado da avaliação cega no teste
├── Dockerfile                # Imagem oficial Python 3.12 com OpenCV e PyTorch
├── docker-compose.yml        # Orquestração do container com Jupyter Lab (porta 8888)
├── requirements.txt          # Dependências do projeto
├── GUIDE.md                  # Roadmap e guia passo a passo das fases
├── ANOTACOES.md              # Diário de bordo detalhado com fundamentos teóricos
└── README.md                 # Documentação principal
```

---

## Como Executar o Projeto em Qualquer Máquina

Você pode executar o projeto de duas formas:
1. **Com Docker (Recomendado):** Ambiente 100% isolado, sem necessidade de configurar Python ou bibliotecas gráficas no seu sistema operacional.
2. **Com Ambiente Virtual Local (`venv`):** Instalação direta no Python da sua máquina.

---

### Opção 1: Executando com Docker e Docker Compose (Recomendado)

#### 1. Pré-requisitos
* Ter o **Docker** e o **Docker Compose** instalados na sua máquina ([Instruções Oficiais do Docker](https://docs.docker.com/get-docker/)).

#### 2. Clonar o repositório
```bash
git clone https://github.com/carloshfgit/epi_finder.git
cd epi_finder
```

#### 3. Construir a imagem e iniciar o ambiente
```bash
docker compose up --build
```
> O Docker irá compilar o ambiente Python 3.12 com todas as dependências de sistema (OpenCV, FFmpeg, PyTorch, Pandas, Ultralytics) e inicializará o **Jupyter Lab** automaticamente.

#### 4. Acessar o Jupyter Lab
Abra o seu navegador web favorito e acesse:
**[http://localhost:8888](http://localhost:8888)**

Você terá acesso direto aos notebooks interativos em `notebooks/`:
* `01_eda_dataset.ipynb`
* `02_training_yolov8.ipynb`
* `03_model_evaluation.ipynb`

#### 5. Executar os scripts Python via CLI no Container
Você pode rodar comandos diretamente dentro do container sem precisar do navegador:

* **Para rodar o treinamento:**
  ```bash
  docker compose exec epi-finder python src/train.py --epochs 25 --batch 8
  ```
* **Para rodar a avaliação formal de métricas no split de teste:**
  ```bash
  docker compose exec epi-finder python src/evaluate.py --weights models/best.pt --split test
  ```
* **Para verificar ajuda e parâmetros de qualquer script CLI:**
  ```bash
  docker compose exec epi-finder python src/train.py --help
  docker compose exec epi-finder python src/evaluate.py --help
  ```

#### 6. Parar o container
```bash
docker compose down
```

---

### Opção 2: Executando Localmente (Sem Docker / via Virtualenv)

Caso prefira rodar diretamente no seu sistema operacional:

#### 1. Pré-requisitos
* **Python 3.10, 3.11 ou 3.12** instalado.
* `git` instalado.

#### 2. Criar e ativar o ambiente virtual
```bash
# Clone o repositório
git clone https://github.com/carloshfgit/epi_finder.git
cd epi_finder

# No Linux ou macOS:
python3 -m venv venv
source venv/bin/activate

# No Windows (Prompt de Comando ou PowerShell):
python -m venv venv
.\venv\Scripts\activate
```

#### 3. Instalar as dependências
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Iniciar o Jupyter Lab ou rodar os scripts
```bash
# Iniciar o ambiente interativo
jupyter lab

# Ou executar o script de avaliação diretamente:
python src/evaluate.py --weights models/best.pt --split test
```

---


## Documentações Complementares

* Para o roteiro didático e checklist completo: consulte o [GUIDE.md](GUIDE.md).
* Para o diário de bordo e detalhamento dos fundamentos teóricos: consulte o [ANOTACOES.md](ANOTACOES.md).
* Para a gestão e auditoria de pesos treinados: consulte o [models/README.md](models/README.md).

---

