# Imagem base oficial do Python
FROM python:3.12-slim

# Evita criação de arquivos .pyc e força o buffer de saída do Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Instala dependências do sistema operacional necessárias para OpenCV, PyTorch e manipulação de vídeo
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Define o diretório de trabalho no container
WORKDIR /app

# Copia e instala as dependências do Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia o restante do código do projeto para o container
COPY . .

# Expõe portas para Jupyter Lab (8888) e Streamlit (8501)
EXPOSE 8888 8501

# Comando padrão: Inicia o Jupyter Lab acessível pelo navegador do host
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
