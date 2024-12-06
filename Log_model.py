#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt # visualização gráfica
#%%
# Carregar os dados
df = pd.read_excel('diabetes_kaggle.xlsx', engine='openpyxl')  # Use o argumento 'engine' para compatibilidade

# Verificar as primeiras linhas
print(df.head())
#%%
# Separar variáveis independentes e a variável dependente
X = df.drop('Outcome', axis=1)
y = df['Outcome']
#%%
# Dividir os dados em 70% para treino e 30% para teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#%%
# Padronizar as variáveis para melhorar o desempenho do modelo
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#%%
# Inicializar e treinar o modelo de regressão logística
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
#%%
# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)
#%%
# Avaliar o desempenho do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy:.2f}')

# Relatório de classificação
print(classification_report(y_test, y_pred))

# Matriz de confusão
print(confusion_matrix(y_test, y_pred))


# In[3.7]: Construção de função para a definição da matriz de confusão
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, recall_score

def matriz_confusao(predicts, observado, cutoff):
    # Transformar as probabilidades em predições binárias com base no cutoff
    predicao_binaria = (predicts >= cutoff).astype(int)

    # Gerar a matriz de confusão
    cm = confusion_matrix(observado, predicao_binaria)
    
    # Plotar a matriz de confusão
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap='Blues', values_format='d')
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.title('Matriz de Confusão')
    plt.show()

    # Calcular as métricas de sensibilidade, especificidade e acurácia
    sensitividade = recall_score(observado, predicao_binaria, pos_label=1)
    especificidade = recall_score(observado, predicao_binaria, pos_label=0)
    acuracia = accuracy_score(observado, predicao_binaria)

    # Visualizar os principais indicadores desta matriz de confusão
    indicadores = pd.DataFrame({'Sensitividade': [sensitividade],
                                'Especificidade': [especificidade],
                                'Acurácia': [acuracia]})
    return indicadores

# Usar a função com o modelo treinado
# Fazer previsões de probabilidade
y_probs = model.predict_proba(X_test)[:, 1]

# Plotar a matriz de confusão com o cutoff de 0.5
indicadores = matriz_confusao(y_probs, y_test, cutoff=0.40)
print(indicadores)

#%%
import pandas as pd

# Suponha que y_test são os valores reais e y_pred são as previsões do modelo
# Convertendo para binário, se necessário (usando 0.5 como cutoff)
y_pred_binary = (y_pred >= 0.5).astype(int)

# Criando a tabela com pandas
df_comparison = pd.DataFrame({
    'Valor Real': y_test,
    'Valor Previsto': y_pred_binary
})

# Exibindo as primeiras linhas da tabela
print(df_comparison.head(10))

#%%
# Obter as probabilidades contínuas
y_pred_probs = model.predict_proba(X_test)[:, 1]  # Apenas a probabilidade da classe 1

# Criando a tabela com pandas
df_comparison = pd.DataFrame({
    'Valor Real': y_test,
    'Valor da Sigmoide': y_pred_probs
})

# Exibindo as primeiras linhas da tabela
print(df_comparison.head(10))

#%%
from sklearn.metrics import roc_curve, roc_auc_score

# Calcular a curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)

# Calcular a AUC
auc_score = roc_auc_score(y_test, y_pred_probs)

# Plotar a curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Classificador Aleatório')
plt.xlabel('Taxa de Falsos Positivos (FPR)')
plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
