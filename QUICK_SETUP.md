# ⚡ Quick Setup - EPI Finder

Guia rápido e objetivo para colocar o ambiente de desenvolvimento e execução do **EPI Finder** para rodar em poucos minutos.

---

## 🛠️ Requisitos Prévios

Escolha um dos dois caminhos abaixo para executar o projeto:
* **Opção A (Recomendado):** Docker e Docker Compose instalados.
* **Opção B:** Python 3.10+ e gerenciador de pacotes pip instalados localmente.

---

## 🚀 Método A: Execução Rápida via Docker (Recomendado)

O Docker automatiza todo o setup de bibliotecas nativas de visão computacional (como OpenCV e FFmpeg) e dependências de Deep Learning (PyTorch).

### 1. Inicializar o Ambiente e Jupyter Lab
Na raiz do repositório, execute o comando abaixo para compilar a imagem e iniciar os serviços:
```bash
docker compose up --build
```
> 💡 *Dica:* Se quiser rodar em segundo plano, adicione a flag `-d` ao final do comando.

### 2. Acessar os Serviços
* **Jupyter Lab (Notebooks de Estudo):** Acesse diretamente pelo navegador em **[http://localhost:8888](http://localhost:8888)**.
* **Dashboard Streamlit (Monitoramento SST):** Com o container rodando, ative a interface web com o comando:
  ```bash
  docker compose exec epi-finder streamlit run app.py --server.port 8501 --server.address 0.0.0.0
  ```
  E acesse em: **[http://localhost:8501](http://localhost:8501)**.

### 3. Comandos Úteis do Container
* **Ver logs em tempo real:** `docker compose logs -f`
* **Parar a execução:** `docker compose down`
* **Excluir volumes e cache órfãos:** `docker compose down -v`

---

## 💻 Método B: Execução Local (Sem Docker)

Ideal se você deseja rodar os scripts e o painel web diretamente no interpretador Python do seu sistema host.

### 1. Configurar o Ambiente Virtual (`venv`)
No terminal, execute conforme o seu sistema operacional:

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell / Prompt):**
```bash
python -m venv venv
.\venv\Scripts\activate
```

### 2. Instalar Dependências e OpenCV
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Iniciar a Aplicação e Notebooks
* **Dashboard Web:** `streamlit run app.py`
* **Interface Jupyter Lab:** `jupyter lab`

---

## 💻 Cheat Sheet: Comandos Operacionais (CLI)

Você pode executar o pipeline modular via terminal (local ou através de `docker compose exec epi-finder <comando>`):

* **Treinamento (Fine-Tuning YOLOv8):**
  ```bash
  python src/train.py --epochs 25 --batch 8
  ```
* **Avaliação de Métricas (Split de Teste):**
  ```bash
  python src/evaluate.py --weights models/best.pt --split test
  ```
* **Inferência Operacional (Imagens estáticas + Recorte de evidências):**
  ```bash
  python src/inference.py --source data/dataset/test/images/ --weights models/best.pt --save-crops
  ```
* **Inferência em Vídeo com Rastreamento (MOT ByteTrack + Evidências):**
  ```bash
  python src/inference.py --source caminho/do/video.mp4 --weights models/best.pt --track --save-crops
  ```

---
*Para informações detalhadas sobre a arquitetura e fundamentos teóricos do projeto, consulte o [README.md](README.md).*
