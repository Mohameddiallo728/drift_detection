import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Data for the table
data = {
    "Concept": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Définition": [
        "La proportion de diagnostics corrects parmi tous les diagnostics effectués.",
        "La proportion de vrais positifs (diagnostics corrects de la maladie) parmi tous les diagnostics positifs (diagnostics de la maladie).",
        "La proportion de vrais positifs parmi tous les vrais cas (patients réellement malades).",
        "La moyenne harmonique de la précision et du rappel."
    ],
    "Exemple": [
        "Si tu diagnostiques correctement 90 patients sur 100, ton accuracy est de 90%.",
        "Si tu diagnostiques 30 patients comme malades et 25 d'entre eux le sont vraiment, ta précision est de 83.3%.",
        "Si 40 patients sont réellement malades et tu diagnostiques correctement 25 d'entre eux, ton rappel est de 62.5%.",
        "Si ta précision est de 83.3% et ton rappel est de 62.5%, ton F1 score est de 71.4%."
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Plot the table
plt.figure(figsize=(12, 4))
sns.set(font_scale=1.2)
plt.axis('off')
table = plt.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center', colColours=['#f2f2f2']*len(df.columns))
table.auto_set_font_size(False)
table.set_fontsize(12)
table.auto_set_column_width(col=list(range(len(df.columns))))
table.scale(5, 5)

# Save the image
plt.savefig('accuracy_precision_recall_f1_score.png', bbox_inches='tight', dpi=300)
plt.show()
