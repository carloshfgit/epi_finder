# 🚀 Guia Completo de Deploy: EPI Finder no Streamlit Community Cloud

Este guia descreve o passo a passo detalhado para colocar a aplicação **EPI Finder** no ar gratuitamente utilizando o **Streamlit Community Cloud** conectado ao seu repositório no **GitHub**.

---

## 📋 Sumário
1. [Visão Geral e Arquitetura de Deploy](#1-visão-geral-e-arquitetura-de-deploy)
2. [Arquivos e Configurações Essenciais](#2-arquivos-e-configurações-essenciais)
3. [Preparação do Repositório Local](#3-preparação-do-repositório-local)
4. [Passo a Passo no Streamlit Cloud](#4-passo-a-passo-no-streamlit-cloud)
5. [Considerações de Performance e Recursos](#5-considerações-de-performance-e-recursos)
6. [Resolução de Problemas (Troubleshooting)](#6-resolução-de-problemas-troubleshooting)
7. [Atualizações Contínuas (CI/CD Automático)](#7-atualizações-contínuas-cicd-automático)

---

## 1. Visão Geral e Arquitetura de Deploy

O **Streamlit Community Cloud** executa contêineres Debian Linux gerenciados que:
- Clonam seu repositório GitHub automaticamente.
- Instalam dependências de sistema via `packages.txt` (`apt-get`).
- Instalam bibliotecas Python via `requirements.txt` (`pip`).
- Executam o script principal `app.py` sob conexão segura (**HTTPS**), permitindo o uso de webcam e uploads de mídia.

```mermaid
flowchart TD
    A[Repositório GitHub\n(Branch: main)] -->|Webhook automático| B[Streamlit Community Cloud]
    B --> C[Instala pacotes do SO:\npackages.txt]
    C --> D[Instala libs Python:\nrequirements.txt]
    D --> E[Executa streamlit run app.py]
    E --> F[Aplicação Online HTTPS:\nhttps://<seu-app>.streamlit.app]
```

---

## 2. Arquivos e Configurações Essenciais

Para que o deploy funcione sem falhas de bibliotecas ou caminhos, certifique-se de que os seguintes arquivos estão presentes na raiz do projeto:

### 2.1. `packages.txt` (Dependências do Sistema Operacional)
Codecs e utilitários nativos de manipulação de vídeo no Linux.
Arquivo: [`packages.txt`]
```text
ffmpeg
```

### 2.2. `requirements.txt` (Dependências Python)
Utilizamos `opencv-python-headless` para evitar dependências de drivers de janela gráfica em servidores e nuvem.
Arquivo: [`requirements.txt`]
```text
ultralytics>=8.0.0
opencv-python-headless>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
albumentations>=1.3.0
pyyaml>=6.0
streamlit>=1.30.0
plotly>=5.18.0
```

### 2.3. `.streamlit/config.toml` (Parâmetros do Servidor e Tema)
Configura o limite de upload de vídeos (200MB) e tema escuro customizado.
Arquivo: [`.streamlit/config.toml`]
```toml
[server]
headless = true
enableCORS = false
enableXsrfProtection = true
maxUploadSize = 200

[theme]
primaryColor = "#3b82f6"
backgroundColor = "#0e1726"
secondaryBackgroundColor = "#1e293b"
textColor = "#f8fafc"
font = "sans serif"
```

### 2.4. Pesos do Modelo Treinado (`models/best.pt`)
Por padrão, arquivos `.pt` podem estar no `.gitignore`. Para o modelo treinado estar disponível no deploy sem depender apenas do download em tempo de execução:
- Certifique-se de versionar o modelo `models/best.pt` (~6.2 MB):
```bash
git add -f models/best.pt
```
*(GitHub aceita arquivos individuais de até 100MB sem necessidade de Git LFS).*

---

## 3. Preparação do Repositório Local

Antes de abrir o navegador, execute os seguintes passos no seu terminal local:

### Passo 3.1: Verificar o status dos arquivos
```bash
git status
```
Verifique se `packages.txt`, `.streamlit/config.toml`, `app.py`, `src/` e `models/best.pt` estão preparados.

### Passo 3.2: Adicionar e Commitar as alterações
```bash
git add app.py requirements.txt packages.txt .streamlit/config.toml src/ models/best.pt DEPLOY.md
git commit -m "feat: configuracoes para deploy no Streamlit Community Cloud"
```

### Passo 3.3: Enviar para o GitHub
```bash
git push origin main
```
*(Substitua `main` pelo nome da sua branch padrão, caso utilize `master`).*

---

## 4. Passo a Passo no Streamlit Cloud

### Passo 4.1: Acessar a Plataforma
1. Acesse **[share.streamlit.io](https://share.streamlit.io/)** ou **[streamlit.io/cloud](https://streamlit.io/cloud)**.
2. Faça login com a sua conta do **GitHub**.

### Passo 4.2: Criar Nova Aplicação
1. No dashboard principal, clique no botão **"New app"** (ou **"Create app"**).
2. Na tela de configuração, preencha:
   - **Repository:** Selecione o repositório do projeto (ex.: `seu-usuario/epi_finder`).
   - **Branch:** `main` (ou a branch onde subiu o código).
   - **Main file path:** `app.py`.
   - **App URL (Opcional):** Personalize o subdomínio desejado (ex.: `epi-finder-sst.streamlit.app`).

### Passo 4.3: Configurações Avançadas (Opcional)
1. Clique em **"Advanced settings"**.
2. **Python Version:** Selecione `3.11` ou `3.12`.
3. **Secrets:** Caso venha a integrar com banco de dados externo ou APIs de notificação (Slack/Telegram), insira as chaves aqui. (Para o funcionamento padrão com detecção local e modelos YOLO, nenhuma chave de API é necessária).
4. Clique em **"Save"**.

### Passo 4.4: Iniciar o Deploy
1. Clique no botão **"Deploy!"**.
2. O terminal de build lateral será aberto exibindo a instalação dos pacotes Linux e Python.
3. Aguarde cerca de 2 a 3 minutos na primeira inicialização.

---

## 5. Considerações de Performance e Recursos

O Streamlit Community Cloud gratuito oferece **~1 GB de memória RAM** e CPUs compartilhadas. Para garantir máxima estabilidade:

| Funcionalidade | Comportamento no Cloud | Recomendação |
| :--- | :--- | :--- |
| **Detecção em Imagens** | Ultrarrápida (~30-80ms por frame) | Uso padrão com `models/best.pt` ou `yolov8n.pt`. |
| **Webcam Estática (`st.camera_input`)** | Funciona nativamente via HTTPS do navegador | Excelente para testes e demonstrações em tempo real. |
| **Vídeos Curtos (< 30s / 1080p)** | Processamento fluido | Ideal para demonstração de conformidade e MOT tracking. |
| **Vídeos Longos (> 5 min)** | Alto consumo de CPU/RAM | Recomenda-se processar trechos menores para evitar estouro de memória (OOM). |

---

## 6. Resolução de Problemas (Troubleshooting)

### 🔴 1. Erro `ImportError: libGL.so.1: cannot open shared object file`
- **Causa:** OpenCV precisa de bibliotecas gráficas do Linux que não vêm na imagem base padrão do Python.
- **Solução:** Certifique-se de que o arquivo [`packages.txt`](packages.txt) está na raiz do repositório no GitHub com `libgl1` e `libglib2.0-0`. Em seguida, reinicie o app no Streamlit Cloud (*"Reboot app"*).

### 🔴 2. `FileNotFoundError: models/best.pt`
- **Causa:** O arquivo `.pt` foi bloqueado pelo `.gitignore` e não subiu para o GitHub.
- **Solução:** Force a inclusão do arquivo com `git add -f models/best.pt`, faça o commit e dê `git push`. Alternativamente, a aplicação fará fallback automático para `yolov8n.pt` ou permitirá o upload manual via sidebar.

### 🔴 3. "App is out of memory (OOM)" ao processar vídeos pesados
- **Causa:** Processar vídeos longos em resolução 4K na CPU gratuita pode exceder 1 GB de RAM.
- **Solução:**
  1. No painel lateral do app, mantenha a opção de vídeo em resolução padrão.
  2. Ajuste o frame skip ou envie arquivos de vídeo com menor duração (MP4/H.264).

### 🔴 4. Como Reiniciar o App ou Limpar Cache
- Clique no menu **"Manage app"** (canto inferior direito) $\rightarrow$ **"..."** $\rightarrow$ **"Reboot app"** ou **"Clear cache and deploy"**.

---

## 7. Atualizações Contínuas (CI/CD Automático)

O Streamlit Community Cloud possui integração contínua automática:
1. Sempre que você fizer um novo `git push origin main`, o Streamlit detecta as alterações e atualiza o app em produção em poucos segundos.
2. Você pode acompanhar o status da aplicação e logs de acesso diretamente pelo botão **"Manage app"** no canto inferior da tela.

---

> ✨ **Dica para Apresentações e Portfólio:**
> Adicione o link do seu app em produção (ex: `https://epi-finder-sst.streamlit.app`) no topo do seu [`README.md`](README.md) e no seu perfil do LinkedIn / currículo como demonstração prática ao vivo!
