# Anotações de Aprendizado e Progresso - EPI Finder

Neste documento, registro as minhas anotações pessoais de estudo, decisões técnicas e o progresso realizado durante o desenvolvimento do projeto EPI Finder.

---

## 1. Visão Geral do Problema e Motivação

Iniciei este projeto com o objetivo de alavancar e praticar meu aprendizado sobre Visão Computacional e IA, desenvolvendo uma solução aplicada à Segurança do Trabalho. O problema central consiste em identificar automaticamente, por meio de imagens e transmissões de vídeo, se pessoas em áreas operacionais ou canteiros de obras estão utilizando o capacete de proteção obrigatório.

Defini o escopo de detecção em duas classes fundamentais:
* **helmet (classe 0):** Pessoa utilizando capacete de segurança.
* **no_helmet (classe 1):** Pessoa sem capacete / cabeça desprotegida.

Optei por um problema binário bem delimitado para garantir maior agilidade na rotulagem, menor necessidade de volume massivo de dados para convergência e foco na precisão das detecções críticas de infração.

---

## 2. Bases Teóricas Consolidadas

Durante a etapa inicial, aprofundei os conceitos teóricos essenciais para a execução do projeto:

### 2.1. Detecção de Objetos vs. Classificação de Imagens
Compreendi a distinção entre tarefas de visão:
* A classificação apenas rotula a imagem inteira com uma categoria geral.
* A detecção de objetos (Object Detection) localiza onde cada objeto está por meio de caixas delimitadoras (*bounding boxes*) e classifica individualmente cada instância presente no quadro.

### 2.2. Arquitetura YOLO e Transfer Learning
* Entendi que a família de modelos YOLO (*You Only Look Once*), em especial o YOLOv8 da Ultralytics, opera como um detector de estágio único (*single-stage detector*). Isso permite analisar a imagem completa em uma única passagem pela rede neural, atingindo a taxa de quadros necessária para inferência em tempo real.
* Adotei a estratégia de *Transfer Learning* utilizando o modelo base leve `yolov8n.pt` (YOLOv8 Nano). Dessa forma, aproveito pesos pré-treinados no dataset COCO para extração de características visuais primitivas (bordas, texturas, formas humanas) e realizo o ajuste fino (*fine-tuning*) no meu conjunto de dados específico de capacetes.
* Estudei as três componentes principais da função de perda da rede:
  * `box_loss`: erro de regressão na precisão das bordas das caixas delimitadoras.
  * `cls_loss`: erro na classificação correta entre `helmet` e `no_helmet`.
  * `dfl_loss` (*Distribution Focal Loss*): refinamento de coordenadas em casos de oclusão e limites difusos.

### 2.3. Formato de Anotações YOLO
Consolidei a estrutura do formato YOLO para anotações. Cada arquivo de texto associado a uma imagem deve conter linhas com 5 valores normalizados entre 0.0 e 1.0:
```text
<class_id> <x_center> <y_center> <width> <height>
```
Onde:
* `x_center` e `y_center` são as coordenadas do centro da caixa divididas pela largura e altura totais da imagem, respectivamente.
* `width` e `height` são a largura e altura da caixa normalizadas.

### 2.4. Métricas de Avaliação
Identifiquei as métricas que utilizarei para validar o modelo:
* **IoU (Intersection over Union):** Razão entre a área de sobreposição e a área de união entre a caixa prevista e a anotação real (*ground truth*).
* **Precision:** Proporção de detecções corretas dentre todas as caixas que o modelo classificou como infração (minimiza falsos positivos e alarmes falsos).
* **Recall:** Proporção de infrações reais detectadas pelo modelo (garante que trabalhadores sem capacete não passem despercebidos).
* **mAP@50 e mAP@50-95:** Média da precisão média sob diferentes limiares de IoU para quantificar a acurácia global do detector.

---

## 3. Fase 1: Ambiente, Infraestrutura e Estrutura do Projeto

Concluí integralmente a Fase 1 do projeto, estabelecendo um ambiente de desenvolvimento reprodutível e uma estrutura de diretórios padronizada.

### 3.1. Conteinerização com Docker e Docker Compose
Para evitar problemas de incompatibilidade de dependências e garantir que o ecossistema funcione em qualquer máquina, criei a infraestrutura baseada em Docker:
* **Dockerfile:**
  * Utilizei a imagem base `python:3.12-slim`.
  * Instalei bibliotecas de sistema operacional indispensáveis para processamento de imagem e vídeo (`libgl1`, `libglib2.0-0`, `libgomp1`, `ffmpeg`, `build-essential`).
  * Configurei a instalação das dependências Python sem cache para otimizar o tamanho da imagem.
  * Configurei o Jupyter Lab como serviço padrão acessível na porta `8888`.
* **docker-compose.yml:**
  * Configurei o mapeamento de volumes para sincronizar os arquivos locais com o diretório `/app` do container em tempo real.
  * Mapeei a porta `8888:8888`.

### 3.2. Gerenciamento de Dependências
Estruturei o arquivo `requirements.txt` contemplando as bibliotecas necessárias para todas as fases:
* `ultralytics` para modelagem e treinamento do YOLOv8.
* `opencv-python` para processamento de imagem e vídeo.
* `numpy` e `pandas` para manipulação de tensores, cálculos manuais de IoU e relatórios de auditoria.
* `matplotlib` e `seaborn` para visualizações gráficas e curvas de aprendizado.
* `albumentations` para aumento de dados (*data augmentation*).
* `pyyaml` para manipulação do arquivo de configuração do dataset.
* `jupyterlab` para desenvolvimento interativo.

### 3.3. Estruturação do Repositório
Defini a árvore de diretórios do projeto para manter uma separação clara de responsabilidades:

```text
epi_finder/
├── data/                  # Conjunto de dados (raw e dataset particionado)
│   ├── raw/               # Imagens originais baixadas (dados brutos imutáveis)
│   └── dataset/           # Dataset organizado em train/valid/test (images/ e labels/)
├── docs/                  # Documentações técnicas e guias
├── notebooks/             # Notebooks de análise exploratória e treinamento
├── src/                   # Módulos Python reutilizáveis (utils.py, inference.py)
├── models/                # Pesos salvos dos modelos treinados (.pt / .onnx)
├── Dockerfile             # Configuração da imagem do container
├── docker-compose.yml     # Orquestração do ambiente conteinerizado
├── requirements.txt       # Lista de dependências Python
├── anotacoes.md           # Diário de bordo e anotações de aprendizado
└── .dockerignore / .gitignore
```

### 3.4. Racional Técnico e Aprendizados sobre a Estrutura de Pastas
Aprofundei os motivos técnicos pelos quais esta estrutura é a mais adequada para projetos de Visão Computacional e Machine Learning:

1. **Compatibilidade Nativa com YOLOv8:**
   * A biblioteca `ultralytics` exige que imagens (`images/`) e anotações em texto (`labels/`) estejam emparelhadas em subpastas correspondentes para `train`, `valid` e `test`. Essa padronização simplifica a configuração do `data.yaml`.

2. **Preservação de Dados Brutos (`data/raw/`):**
   * Manter os dados originais isolados garante a **imutabilidade**. Em caso de erros durante o pré-processamento ou re-rotulagem, os dados originais ficam salvos e disponíveis para reprocessamento sem necessidade de novo download.

3. **Separação de Escopos (`notebooks/` vs. `src/`):**
   * **`notebooks/`:** Reservado para prototipagem rápida, Análise Exploratória de Dados (EDA) e testes visuais interativos.
   * **`src/`:** Armazena o código modular, limpo e testado (ex: `inference.py`, `utils.py`), garantindo que o projeto em produção utilize scripts Python reutilizáveis sem dependência de notebooks.

4. **Gerenciamento de Artefatos (`models/`):**
   * Centraliza os checkpoints de treinamento (`.pt`, `.onnx`), isolando arquivos binários pesados da raiz do código.

5. **Estratégia de Controle de Versão (`.gitignore` e `.gitkeep`):**
   * **Motivo para ignorar a pasta `data/` no `.gitignore`:**
     * **Performance e Limites do Git:** Datasets de imagens ocupam centenas de MB ou GB. O Git não é otimizado para arquivos binários grandes, o que deixaria comandos como `git clone` lentos e violaria limites do GitHub (100 MB por arquivo).
     * **Privacidade e LGPD:** Fotos de trabalhadores em canteiros de obras envolvem direitos de imagem e privacidade, devendo ser mantidas fora de repositórios públicos.
     * **Separação de Ciclo de Vida:** O Git deve versionar o código e a arquitetura, enquanto os dados brutos são gerenciados por storages externos ou ferramentas apropriadas (Roboflow, DVC, S3).
   * **Uso Estratégico de `.gitkeep`:**
     * Utilizei arquivos `.gitkeep` para persistir a estrutura completa de pastas no repositório Git sem incluir o conteúdo pesado. Com as regras no `.gitignore` (`data/*`, `!data/*/`, `!data/**/.gitkeep`), o projeto mantém a arquitetura organizada para qualquer novo desenvolvedor.


---

## 4. Próximos Passos (Fase 2 em diante)

Com a Fase 1 concluída, minhas próximas ações serão:
1. **Fase 2 (Coleta e Rotulagem):** Coletar de 200 a 400 imagens representativas e realizar a anotação das classes `helmet` e `no_helmet` via Roboflow, exportando no formato YOLOv8 e dividindo em treino, validação e teste.
2. **Fase 3 (Pipeline & EDA):** Construir o notebook de análise exploratória utilizando Pandas e NumPy para auditar o balanceamento das classes, calcular a distribuição das áreas das caixas e gerar o arquivo `data.yaml`.
3. **Fase 4 (Treinamento):** Executar o ajuste fino do YOLOv8 Nano e acompanhar as curvas de perda e métricas de convergência.
