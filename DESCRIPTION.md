### Detector de Uso de Capacete de Segurança (EPI Finder - Detecção de Objetos)

Cobre: rotulagem, PyTorch / YOLOv8, NumPy, Pandas, métricas de detecção e pipeline de imagens.

- **O Problema:** Identificar automaticamente se trabalhadores/pessoas em áreas de risco estão utilizando ou não capacete de segurança (`head` - sem capacete vs `helmet` - com capacete) a partir de imagens e câmeras de monitoramento.
    
- **O que devo fazer:**
    
    1. **Coleta e Rotulagem:** Obter dataset no Roboflow (com as classes `head` para sem capacete e `helmet` para com capacete) no formato YOLOv8.
        
    2. **Pipeline de Dados & EDA:** Usar OpenCV, NumPy e Pandas para redimensionamento, normalização, auditoria do dataset (*balanceamento e áreas*) e _data augmentation_.
        
    3. **Treinamento e Experimentos:** Treinar um modelo leve de detecção (ex: YOLOv8 nano via Ultralytics) no Jupyter Notebook ou Google Colab.
        
    4. **Avaliação e Relatórios:** Comparar épocas de treino com Pandas e avaliar com métricas reais ($mAP@50$, precisão, recall), gerando um script de inferência com alertas visuais e logs de conformidade.