Tech Challenge - Insurance Charges Prediction
Este projeto utiliza aprendizado de máquina para prever o custo de planos de seguro de saúde com base em dados demográficos e de saúde dos clientes.

Descrição
O script tech_challenge.py realiza as seguintes etapas:

Carrega e explora o dataset insurance.csv
Visualiza relações entre variáveis (idade, IMC, filhos) e o valor do seguro
Pré-processa os dados usando codificação one-hot e padronização
Divide os dados em conjuntos de treino e teste, estratificando por faixa etária
Treina modelos de regressão linear e Random Forest para prever o valor do seguro
Avalia os modelos usando R² e RMSE
Realiza previsões para novos clientes
Requisitos
Python 3.x
pandas
matplotlib
seaborn
scikit-learn
numpy
Instale as dependências com:

Como usar
Certifique-se de que o arquivo insurance.csv está no mesmo diretório do script.
Execute o script:
O script exibirá gráficos, métricas de avaliação e previsões para novos planos.
Estrutura
tech_challenge.py: Script principal com todo o pipeline de análise e modelagem.
Observações
O script inclui visualizações para análise exploratória e avaliação dos resíduos dos modelos.
O pipeline de pré-processamento garante que os dados estejam prontos para os modelos de machine learning.
O modelo Random Forest é utilizado para estimar o preço de novos planos de seguro.
Desenvolvido para fins de aprendizado e demonstração de técnicas de ciência de dados e machine learning.