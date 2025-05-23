
# Tech Challenge - Previsão de Custos de Seguro Saúde

Este projeto utiliza técnicas de Ciência de Dados e Machine Learning para **prever o valor de planos de seguro saúde** com base em dados demográficos e de saúde dos clientes.

---

## 📋 Descrição

O script `tech_challenge.py` executa um pipeline completo de análise e modelagem preditiva, incluindo:

- Carregamento e exploração do dataset `insurance.csv`
- Visualização de relações entre variáveis (idade, IMC, filhos) e o valor do seguro
- Pré-processamento dos dados com codificação *one-hot* e padronização
- Divisão dos dados em treino e teste, estratificando por faixa etária
- Treinamento de modelos de **Regressão Linear** e **Random Forest**
- Avaliação dos modelos com métricas **R²** e **RMSE**
- Previsão do valor do seguro para novos clientes

---

## 📦 Requisitos

- Python 3.x  
- pandas  
- matplotlib  
- seaborn  
- scikit-learn  
- numpy  

Instale as dependências com:

```bash
pip install -r requirements.txt
```

> Crie um `requirements.txt` com os pacotes usados, se desejar facilitar a instalação.

---

## 🚀 Como usar

1. Certifique-se de que o arquivo `insurance.csv` está no mesmo diretório do script.
2. Execute o script com:

```bash
python tech_challenge.py
```

O script exibirá gráficos, métricas de avaliação e previsões para novos planos de seguro.

---

## 🔍 Etapas do Projeto

1. **Exploração de Dados**  
   Visualização de dispersão entre idade, IMC, filhos e o valor do seguro.

2. **Pré-processamento**  
   Codificação de variáveis categóricas (*One-Hot Encoding*) e padronização de variáveis numéricas.

3. **Divisão dos Dados**  
   Separação em treino e teste, estratificando por faixa etária.

4. **Modelagem**  
   Treinamento de modelos de **Regressão Linear** e **Random Forest**.

5. **Avaliação**  
   Cálculo de métricas como **R²** e **RMSE**, além de análise de resíduos.

6. **Previsão**  
   Estimativa do valor do seguro para novos perfis de clientes.

---

## 📈 Resultados

- O script imprime as métricas de avaliação dos modelos (**R²** e **RMSE**).
- Exibe gráficos de dispersão, matriz de correlação e previsões.
- Realiza previsões para novos clientes, exibindo o valor estimado do plano.

---

## 📁 Estrutura dos Arquivos

- `tech_challenge.py` – Script principal com todo o pipeline de análise e modelagem.
- `insurance.csv` – Base de dados utilizada (não incluída neste repositório).

---

## 📝 Observações

- O pipeline de pré-processamento garante que os dados estejam prontos para os modelos de machine learning.
- O modelo **Random Forest** é utilizado para estimar o preço de novos planos de seguro.
- O projeto pode ser expandido com ajuste de hiperparâmetros e validação cruzada.
- Desenvolvido para fins de aprendizado e demonstração de técnicas de Ciência de Dados e Machine Learning.

---

## 👨‍💻 Autores

- Élcio Jesus Conceição
- Jéssica Santana dos Santos  
- Laércio Aparecido Pedroso  
- Lorrane Aparecida Pedroso
- Rafael Jordão Jardim