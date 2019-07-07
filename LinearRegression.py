import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
data = pd.read_csv("./iris.csv", header = 0)
sns.pairplot(data,hue = "species")
plt.show()
formula = "petal_length ~ sepal_width + sepal_length + petal_width + C(species)"
lrModel = smf.ols(formula, data = data)
reg01 = lrModel.fit()
print(reg01.summary())