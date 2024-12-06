### Modelo logístico de pré-diagnóstico de diabetes: um estudo de caso com dados do Kaggle

### Introdução

De forma simplificada e acessível, desenvolvi um código que fornece uma prévia de diagnósticos de diabetes com base em variáveis de saúde de mulheres acima de 21 anos. Minha intenção é explorar como a **regressão logística** pode auxiliar em predições binárias, determinando a classificação do modelo de acordo com a distribuição de Bernoulli (probabilidade do evento e não evento; P e 1 - P).

Neste trabalho, direcionei os esforços para mantê-lo realmente simples, com o objetivo de que ele permaneça no LinkedIn como parte de um portfólio mais amplo de modelos aplicáveis em soluções de automatização de decisões.

Vale ressaltar que o tema do estudo apenas representa um exemplo prático de aplicação e que não configura ou substitui nenhuma forma de avaliação médica real. Os dados foram obtidos no site Kaggle, onde há uma extensa oferta gratuita de bases de dados.

![](https://media.licdn.com/dms/image/v2/D4D12AQFPnwVK6APzGw/article-inline_image-shrink_1500_2232/article-inline_image-shrink_1500_2232/0/1730305643736?e=1738800000&v=beta&t=8KmFgUUzDD3qSqloQN4Feb0byR9fxgEup9NH1Pwo8Y8)

Cabeçalho com os primeiros registros

O trabalho foi iniciado com as seguintes etapas:

1. **Avaliação e Tratamento dos Dados:** Realizou-se uma análise inicial para identificar dados faltantes, lacunas ou inconsistências, ajustando tudo o que era necessário, mesmo que os dados já estivessem relativamente bem estruturados na origem.
2. **Divisão e Preparação das Variáveis:** As variáveis foram separadas em variáveis explicativas (X) e variável-alvo (Y), seguidas de uma padronização das escalas e definição de parâmetros iniciais para otimizar o desempenho do modelo.

---

### Desenvolvimento

Para o desenvolvimento do modelo, foi escolhida a plataforma Python, sob o interpretador Spyder, para conduzir todas as etapas das análises. As bibliotecas utilizadas no Python foram:

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

Essas bibliotecas permitiram a construção e análise do modelo de maneira eficiente, facilitando a implementação dos métodos estatísticos necessários.

Logo, o modelo é então estruturado matematicamente da seguinte forma:

É calculada a função de regressão linear em **z** (1) chamada de **logito**, para depois ser aplicada na função **y** **sigmoide** (2), que por fim é aplicada na maximização por verossimilhança na função **log-likelihood** (3), que é a função final de construção do modelo.

![](https://media.licdn.com/dms/image/v2/D4D12AQEZkfH88O4b1A/article-inline_image-shrink_1500_2232/article-inline_image-shrink_1500_2232/0/1730297892972?e=1738800000&v=beta&t=SMzrK3cU_X1yx63L4_vQ2o0k0oSmtXTKZjcZtY_pmxo)

Essa é a construção básica do modelo, há ainda outras funções que otimizam e reduzem os erros para aperfeiçoá-lo. Mas, por ora, essas satisfazem o objetivo do trabalho.

Essas funções todas estão implícitas no comando de treinamento do modelo:

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

---

### Definição de Cutoff e Matriz de Confusão

Após a construção inicial do modelo, foram definidos os parâmetros de avaliação.

No momento de desenho da matriz de confusão, que é a tabela que apresenta os acertos e os erros, é definido um valor de **cutoff** para eleger em qual parte da curva do modelo serão divididos os dados. Lembrando que esse modelo prediz valores binários.

Foram testados diversos valores de cutoff até se alcançar um ponto ideial sobre o melhor equilíbrio entre sensibilidade e especificidade. O valor de **0,4** foi identificado como o mais adequado, permitindo maximizar a identificação de resultados positivos, sem comprometer excessivamente a taxa de falsos positivos.

Na avaliação inicial de indicadores com esse valor **0,4** de cutoff chegou-se em:

**Sensitividade** - Modelo identifica corretamente **73%** dos casos positivos de diabetes.

**Especificidade** - Modelo identifica corretamente **72%** dos casos negativos de diabetes.

**Acurácia** - Modelo acerta globalmente **72%** das previsões totais.

Foi adotada uma abordagem para tornar o modelo mais equilibrado em termos de taxa de acerto tanto para verdadeiros positivos quanto para verdadeiros negativos. Em modelos aplicados à área da saúde, segundo a literatura do assunto, é comum priorizar a identificação mais clara dos casos positivos, já que um cutoff mais elevado poderia reduzir o número de diagnósticos positivos detectados, mas ao custo de aumentar o risco de falsos negativos, o que pode comprometer a eficácia na detecção de casos críticos.

A matriz de confusão abaixo simplesmente apresenta quantos resultados **verdadeiros** **negativos** de fato foram diagnosticados corretamente (105), diagnósticos incorretos como **falso negativos** (23), além dos **verdadeiros positivos** (62) e **falsos positivos** (41).

![](https://media.licdn.com/dms/image/v2/D4D12AQHi4VVeFRB85A/article-inline_image-shrink_1500_2232/article-inline_image-shrink_1500_2232/0/1730312803186?e=1738800000&v=beta&t=9Zz6TmRO_SEoEVhLL4viDW4aOI6LiWMuPKI-Zw7FMCg)

Matriz de Confusão com cutoff de 0,4.

O valor de cutoff é responsável por estabelecer a classificação do valor da sigmoide como 0 ou 1, de acordo com o ponto de corte definido; ou seja, se o valor de output do modelo for **0,33**, iremos considerar como um caso **negativo** para diabetes atribuindo-lhe **0**, pois está abaixo do cutoff de **0,4**. Se for maior que **0,4**, então classificaremos o caso como **positivo** atribuindo-lhe **1.**

Na tabela a seguir, são apresentados os resultados finais do modelo, com o valor real e o valor da sigmoide [aplicação do **logito z (1)** em **sigmoide (2)**] para cada observação, conforme descrito na etapa de desenvolvimento. Isso demonstra como o modelo converte probabilidades em classes binárias, baseando-se no cutoff estabelecido.

![](https://media.licdn.com/dms/image/v2/D4D12AQFlZEHXoGqhlA/article-inline_image-shrink_1500_2232/article-inline_image-shrink_1500_2232/0/1730314441366?e=1738800000&v=beta&t=GAogqIrp75SFEOrXCdDsW7ZNuuq5s4l3Zl3PdzjwXDc)

O valor da sigmoide é convertido a 0 ou 1 a depender do critério estabelecido do cutoff.

---

### Curva ROC

A **Curva ROC** (Receiver Operating Characteristic) é uma ferramenta visual usada para avaliar o desempenho de modelos de classificação binária. Ela mostra a relação entre a **Taxa de Verdadeiros Positivos (Sensibilidade)** e a **Taxa de Falsos Positivos** em diferentes limiares de decisão.

Quanto mais próxima a curva estiver do canto superior esquerdo, melhor é a capacidade do modelo de separar as classes, indicando uma menor taxa de falsos positivos e uma maior taxa de verdadeiros positivos.

### Área Sob a Curva (AUC)

A **Área Sob a Curva** (Area Under Curve) quantifica o desempenho geral do modelo, variando de 0 a 1, onde valores mais altos indicam maior capacidade de discriminação entre classes.

**Interpretação da AUC:**

**AUC = 1,0:** Perfeita separação entre classes; o modelo acerta todas as classificações.

**AUC = 0,5:** Classificação aleatória; o modelo não tem capacidade de discriminar entre as classes.

**AUC < 0,5:** Desempenho pior que o acaso; indica que o modelo está fazendo previsões opostas.

**AUC entre 0,7 e 0,8:** Modelo razoável.

**AUC entre 0,8 e 0,9:** Modelo bom.

**AUC acima de 0,9:** Modelo excelente.

Neste trabalho a AUC ficou em **0,8**.

Na curva ROC a seguir, é possível observar como ela se aproxima do canto superior esquerdo, indicando que o modelo possui uma boa capacidade de separação entre classes. Quanto mais esticada a curva estiver para esse canto, melhor o desempenho do modelo, pois representa uma alta sensibilidade combinada com uma baixa taxa de falsos positivos, conforme destacado acima.

![](https://media.licdn.com/dms/image/v2/D4D12AQFL-0NeWm5oZQ/article-inline_image-shrink_1500_2232/article-inline_image-shrink_1500_2232/0/1730312891610?e=1738800000&v=beta&t=rCvXkGE_XWlnl1f5KuZCPiK4N4XnzZoe_TXSHfVmor8)

A partir deste ponto, o modelo está preparado para ser aplicado a novos dados e integrado em um ambiente de produção. Isso permitirá que ele realize previsões em tempo real, ajudando na tomada de decisões automatizadas com base nos padrões aprendidos durante o treinamento.

### Conclusão

O modelo de regressão logística apresentado demonstrou um desempenho consistente na classificação binária, evidenciado pela análise das métricas de sensibilidade, especificidade, acurácia e pela interpretação da Curva ROC. O uso de Python como plataforma permitiu explorar com clareza as etapas de construção, ajuste de parâmetros e avaliação do modelo, mantendo um equilíbrio entre identificação de resultados positivos e controle de falsos positivos.

Embora o foco deste estudo tenha sido um exemplo de pré-diagnóstico de diabetes, a metodologia e as técnicas aplicadas são amplamente adaptáveis a outros problemas de classificação binária na área de saúde e em diversos setores, como finanças, marketing, segurança, entre outros.

Espero que este artigo tenha sido útil para entender como a regressão logística pode ser utilizada para automatizar decisões e fornecer insights valiosos em diferentes contextos. O objetivo era demonstrar de maneira mais simples possível (tomara que ficou!) como é possível se utilizar dos algoritmos de machine learning para desafios concretos. Sinta-se à vontade para deixar seus comentários, compartilhar suas experiências ou sugerir novos temas para exploração.
