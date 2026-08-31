# 🛠️ Guia de Resolução de Problemas e Diagnósticos (TROUBLESHOOTING.md)

Este documento registra incidentes, anomalias de predição, falhas operacionais e comportamentos inesperados identificados no ecossistema **EPI Finder**. O objetivo é manter um histórico estruturado de diagnósticos de causa raiz (*Root Cause Analysis* - RCA), hipóteses investigativas e planos de ação para apoiar o ciclo de MLOps.

---

## 📋 Sumário de Incidentes Registrados

| ID | Data | Componente | Descrição Resumida | Severidade | Status |
|:---|:---:|:---|:---|:---:|:---:|
| [INC-001](#inc-001-modelo-não-identifica-ausência-de-capacete-em-imagem-de-teste-externa) | 2026-08-30 | Modelo / Visão | Falha na detecção da classe `head` (sem capacete) em foto de motociclistas | 🔴 Alta | ✅ Mitigado |
| [INC-002](#inc-002-falso-positivo-pessoa-calva-garupa-classificada-como-helmet-com-capacete) | 2026-08-30 | Modelo / Visão | Falso positivo: garupa calvo sem capacete detectado como 'helmet' | 🟡 Média | 🔍 Sob Investigação |

---

## 🔍 Incidentes em Detalhe

### INC-001: Modelo não identifica ausência de capacete em imagem de teste externa

- **Data de Abertura:** 30/08/2026
- **Ambiente:** Dashboard Streamlit ([`app.py`](app.py)) / Módulo de Inferência ([`src/inference.py`](src/inference.py))
- **Status:** ✅ Mitigado (Treinamento completo de 50 épocas executado no Colab)
- **Severidade:** 🔴 Alta (Inoperância na detecção de infração)

---

#### 1. Descrição do Sintoma
Ao submeter no Dashboard Streamlit uma imagem externa com dois indivíduos em uma motocicleta sem capacete, o sistema não exibiu nenhuma *bounding box* para a classe `0: head` (sem capacete / alerta de infração) e não contabilizou infrações no sumário de telemetria.

---

#### 2. Investigação Técnica e Evidências
A análise revelou que o modelo original em produção (`models/best.pt`) era fruto de um *smoke test* rápido de apenas 1 época. A rede não tinha pesos convergidos e operava com mAP@50 de 0.00025. 

---

#### 3. Resolução e Validação
- **Ação:** O modelo foi treinado por 50 épocas no Google Colab usando a GPU T4, atingindo mAP@50 global de **76.5%** (Recall de `head` subiu para **68.7%**).
- **Resultado:** Os novos pesos foram implantados em [`models/best.pt`](models/best.pt). O piloto da motocicleta (Bolsonaro) agora é detectado perfeitamente sob a classe **`ALERTA: SEM CAPACETE` com 85% de confiança**.
- **Observação:** O problema da não detecção global de infrações na imagem foi mitigado com sucesso, contudo revelou um falso positivo no garupa da mesma imagem (registrado no incidente [INC-002](#inc-002-falso-positivo-pessoa-calva-garupa-classificada-como-helmet-com-capacete)).

---

### INC-002: Falso positivo: pessoa calva (garupa) classificada como `helmet` (com capacete)

- **Data de Abertura:** 30/08/2026
- **Ambiente:** Dashboard Streamlit ([`app.py`](app.py)) / Módulo de Inferência ([`src/inference.py`](src/inference.py))
- **Status:** 🔍 Sob Investigação
- **Severidade:** 🟡 Média (Falha de falso positivo que classifica pessoa desprotegida como segura)

---

#### 1. Descrição do Sintoma
Após o deploy dos pesos de 50 épocas do Colab, o garupa da motocicleta (Luciano Hang), que é calvo e está sem capacete, foi classificado incorretamente como **`Capacete 0.72`** (classe `1: helmet`) com 72% de confiança pelo modelo de visão.

---

#### 2. Investigação Técnica e Evidências
A análise morfológica e visual do erro demonstrou os seguintes fatores:
- **Geometria e Brilho:** A cabeça calva possui formato perfeitamente arredondado e textura lisa e reflexiva. Sob iluminação de ambiente externo, o topo da cabeça gera um brilho (reflexão de luz) altamente similar ao topo de um capacete branco, amarelo ou cinza claro.
- **Viés do Dataset de Treino:** O dataset original `hard-hat-detection-v2` possui um viés de canteiro de obras, onde a grande maioria das pessoas sem capacete rotuladas como `head` possui cabelo visível. O modelo associou incorretamente a "pele lisa + formato circular reflexivo" diretamente com a classe `helmet` por não ter visto exemplos suficientes de calvos desprotegidos.

---

#### 3. Diagnóstico Conclusivo
Trata-se de uma limitação da distribuição dos dados de treinamento do dataset (*Dataset Bias* / *Under-representation*). O modelo aprendeu um atalho de textura/brilho para capacetes que falha com cabeças calvas. O problema **não** pode ser solucionado aumentando a quantidade de épocas de treinamento atuais, pois isso causaria sobreajuste (*overfitting*) e aumentaria a confiança no erro.

---

#### 4. Recomendações e Plano de Ação
- [ ] **Enriquecimento do Dataset:** Coletar e rotular novas amostras de pessoas calvas ou com cabeça raspada sem capacete, inserindo-as no split de treino sob a classe `0: head`.
- [ ] **Aumentos de Dados de Brilho/Contraste:** Implementar técnicas de *Data Augmentation* focadas em *Color Jitter*, brilho e contraste para ensinar o modelo a ignorar reflexos de iluminação como marcador decisivo de capacetes.

---

## 📝 Template para Novos Registros de Incidentes

Para registrar novos incidentes neste documento, utilize o modelo padronizado abaixo:

```markdown
### INC-[NÚMERO]: [Título Resumido do Problema]

- **Data de Abertura:** AAAA-MM-DD
- **Ambiente:** [Ex: Dashboard Streamlit, Docker, Módulo de Inferência, Pipeline de Treino]
- **Status:** [🔍 Sob Investigação | 🛠️ Em Correção | ✅ Resolvido | ⏸️ Pausado]
- **Severidade:** [🔴 Alta | 🟡 Média | 🟢 Baixa]

#### 1. Descrição do Sintoma
[Descrever o comportamento anômalo observado, prints, mensagens de erro ou logs]

#### 2. Investigação Técnica e Evidências
- **Passos para reprodução:**
- **Logs relevantes:**
- **Análise de código / dados:**

#### 3. Diagnóstico Conclusivo (Causa Raiz)
[Explicação técnica fundamentada da causa do problema]

#### 4. Recomendações e Plano de Ação
- [ ] Ação 1
- [ ] Ação 2
- [ ] Validação dos resultados
```
