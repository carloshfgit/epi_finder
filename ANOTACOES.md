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

## 4. Fase 2: Coleta, Seleção e Rotulagem (Boas Práticas e Decisões)

Nesta fase, estudei o ecossistema de anotação de dados e as ferramentas da indústria de Visão Computacional, tomando decisões estratégicas para o avanço do projeto.

### 4.1. Decisão do Dataset (Roboflow Universe)
Optei por utilizar um dataset pré-rotulado do **Roboflow Universe**. Essa escolha permite acelerar a construção da solução e focar no domínio do pipeline completo: Análise Exploratória de Dados (EDA) com Pandas/NumPy, treinamento do YOLOv8 Nano, avaliação detalhada de métricas e construção do script de inferência.

### 4.2. Panorama das Ferramentas de Rotulagem na Indústria
Mapeei as principais ferramentas utilizadas no mercado para anotação de dados de Visão Computacional:
* **Roboflow:** Excelente para prototipagem rápida, gestão de dados na nuvem, data augmentation e exportação multiformato.
* **CVAT (Computer Vision Annotation Tool):** Padrão open-source da indústria mantido pela OpenCV/Intel. Destaca-se no processamento de vídeos (com rastreamento e interpolação automática entre quadros).
* **Label Studio:** Plataforma open-source multi-modal (imagem, áudio, texto), ideal para integração via API em pipelines Python de MLOps.
* **X-AnyLabeling:** Ferramenta desktop local integrada ao **SAM (Segment Anything Model)**, permitindo auto-rotulagem com 1 clique.
* **Plataformas Enterprise (Scale AI, Labelbox, V7 Labs):** Soluções corporativas focadas no gerenciamento de equipes de anotação em larga escala com fluxos rigorosos de controle de qualidade.

### 4.3. Boas Práticas Profissionais para Rotulagem de Imagens
Consolidei as diretrizes fundamentais para garantir datasets de alta qualidade (*"Garbage In, Garbage Out"*):
1. **Guia de Anotação (*Annotation Guidelines*):** Estabelecer regras claras antes de iniciar (limite mínimo de pixels para objetos distantes, regras para objetos cortados nas bordas e tratamento de oclusões).
2. **Caixas Delimitadoras Ajustadas (*Tight Bounding Boxes*):** As caixas devem envolver os objetos de forma rente, sem deixar sobra excessiva de fundo (o que ensinaria o modelo a associar o fundo ao objeto) e sem cortar partes visíveis da classe.
3. **Anotação Exaustiva:** Todas as ocorrências das classes visíveis na imagem devem ser obrigatoriamente rotuladas. Instâncias esquecidas são interpretadas pelo algoritmo de treino do YOLO como "fundo/negativo", penalizando severamente o aprendizado.
4. **Auto-Rotulagem Assistida por IA (*AI-Assisted Labeling / Human-in-the-Loop*):** Uso de modelos pré-treinados ou modelos de segmentação para gerar caixas prévias automaticamente, reduzindo até 80% do esforço manual, com o operador humano atuando na revisão e ajuste fino das caixas.

### 4.4. Conclusão da Fase 2 e Estrutura do Dataset Baixado
* **Dataset Selecionado:** `hard-hat-detection` (Roboflow Universe, versão em formato YOLOv8).
* **Localização no Projeto:** Baixado e descompactado com sucesso em [`data/dataset/`].
* **Particionamento Obtido:** Pastas `train/`, `valid/` e `test/` prontas com subpastas `images/` e `labels/`.
* **Classes Mapeadas no `data.yaml`:**
  * `0`: `with_helmet` (com capacete de proteção)
  * `1`: `without_helmet` (sem capacete / cabeça visível)

### 4.5. Entendimento sobre Extração e Divisão do Dataset (Train / Valid / Test)
* **Preservação da Hierarquia ao Descompactar:** Entendi que ao exportar um dataset no formato **YOLOv8** do Roboflow, o arquivo `.zip` já é montado na nuvem contendo a árvore interna de pastas (`train/`, `valid/`, `test/`, cada uma com subpastas `images/` e `labels/`). A ação de descompactar apenas extrai essa estrutura pré-existente para o diretório de destino.
* **Definição da Proporção do Split (Train/Valid/Test):** A divisão do conjunto de dados em Treino, Validação e Teste é definida na etapa de geração do dataset dentro da plataforma (ex: Roboflow Universe). Aprendi que a ferramenta realiza o sorteio aleatório das imagens (respeitando o balanceamento de classes) e distribui as amostras antes de disponibilizar o pacote compactado para download.

---


## 5. Próximos Passos (Fase 3: Pipeline & EDA)

Com a Fase 2 (Coleta e Rotulagem) totalmente concluída, o foco agora é a **Fase 3**:
1. **Configuração do `data/data.yaml`:** Criar/ajustar o arquivo principal de dados na raiz de `data/` apontando os caminhos absolutos ou relativos para o dataset.
2. **Análise Exploratória de Dados (EDA com Pandas & NumPy):** Criar o notebook `notebooks/01_eda_dataset.ipynb` para:
   * Ler os arquivos `.txt` de anotações em um `pandas.DataFrame`.
   * Verificar a distribuição de contagem de cada classe (`with_helmet` vs `without_helmet`).
   * Calcular estatísticas das caixas (largura, altura, área, aspect ratio) usando NumPy.
3. **Fase 4 (Treinamento):** Iniciar o ajuste fino do modelo `yolov8n.pt`.


