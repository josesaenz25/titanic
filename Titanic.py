# app.py
# Streamlit Titanic Survival Prediction App
# Muestra métricas, tablas, modelos y gráficas interactivas

import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats

from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA




st.markdown("""
    <style>
        /* Oculta la barra superior (Deploy y menú) */
        [data-testid="stDecoration"] {
            display: none;
        }

        /* Opcional: elimina espacio superior extra */
        header {
            visibility: hidden;
        }
    </style>
""", unsafe_allow_html=True)


# Configuración de la página
st.set_page_config(page_title="Titanic Survival Prediction", layout="wide")



# Subtítulo centrado y estilizado
st.markdown("""
        <div style='
            background-color: #F2F2F2;
            padding: 10px;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            text-align: center;
        '>
            <h2 style='color: #062C5F; font-size: 50px;'>🚢 Titanic Survival Prediction</h2>
            <p style='color: #0033A0; font-size: 15px;'>Modelado con KNN, Logistic Regression y Decision Tree usando diferentes esquemas de preprocesamiento.</p>
            <h2 style='color: #0033A0; font-size: 20px;'>Alumno: Granados Saenz José de Jesús</h2>
        </div>
    """, unsafe_allow_html=True)



"\n"
"\n"
"\n"
"\n"
"\n"
# 1) Cargar datos
train = pd.read_csv("Datos de Entrenamiento del Conjunto Titanic.csv")
test  = pd.read_csv("Datos de Prueba del Conjunto Titanic.csv")

st.markdown("""
        <div style='
            background-color: #F2F2F2;
            padding: 10px;
            border-radius: 12px;
            margin-top: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            text-align: center;
            '>
            <h4 style='color: #0033A0;'>📊 Datos de Entrenamiento (muestra)</h4>
            </div>
        """, unsafe_allow_html=True)

st.dataframe(train.head())



"\n"
"\n"
"\n"
st.markdown("""
        <div style='
            background-color: #F2F2F2;
            padding: 10px;
            border-radius: 12px;
            margin-top: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            text-align: center;
            '>
            <h4 style='color: #0033A0;'>📊 Datos de Prueba (muestra)</h4>
            </div>
        """, unsafe_allow_html=True)

st.dataframe(test.head())


"\n"
"\n"
"\n"
# 🔎 Diccionario de variables
st.markdown("""
        <div style='
            background-color: #F2F2F2;
            padding: 10px;
            border-radius: 12px;
            margin-top: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            text-align: center;
            '>
            <h4 style='color: #0033A0;'>📘 Diccionario de Variables del Dataset Titanic</h4>
            </div>
        """, unsafe_allow_html=True)

variable_info = pd.DataFrame({
    "Variable": ["survival", "pclass", "sex", "age", "sibsp", "parch", "ticket", "fare", "cabin", "embarked"],
    "Definición": [
        "Supervivencia",
        "Clase del boleto",
        "Sexo",
        "Edad en años",
        "Hermanos/esposos a bordo",
        "Padres/hijos a bordo",
        "Número de boleto",
        "Tarifa del pasajero",
        "Número de cabina",
        "Puerto de embarque"
    ],
    "Clave": [
        "0 = No, 1 = Sí",
        "1 = 1ra, 2 = 2da, 3 = 3ra",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "C = Cherbourg, Q = Queenstown, S = Southampton"
    ]
})

st.dataframe(variable_info.style.set_properties(**{
    'text-align': 'left',
    'font-size': '14px'
}).set_table_styles([{
    'selector': 'th',
    'props': [('font-size', '15px'), ('text-align', 'left')]
}]))

# 2) Definir variables
target_col = "survival"
X = train.drop(columns=[target_col])
y = train[target_col].astype(int)
X_test = test.copy()

numeric_features = ["age", "fare", "pclass", "sibsp", "parch"]
categorical_features = ["embarked", "sex"]



"\n"
"\n"
# Preprocesamiento
basic_numeric = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
basic_categorical = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
preprocess_basic = ColumnTransformer([("num", basic_numeric, numeric_features), ("cat", basic_categorical, categorical_features)], remainder="drop")

preprocess_advanced = ColumnTransformer([("num", basic_numeric, numeric_features), ("cat", basic_categorical, categorical_features)], remainder="drop")

"\n"
"\n"
# Modelos
models = {
    "KNN": KNeighborsClassifier(n_neighbors=7, weights="distance"),
    "LR": LogisticRegression(max_iter=500, solver="lbfgs"),
    "DT": DecisionTreeClassifier(max_depth=6, random_state=42)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_with_cv(X_data, y_data, pipeline, name):
    y_pred_cv = cross_val_predict(pipeline, X_data, y_data, cv=cv)
    prec = precision_score(y_data, y_pred_cv, zero_division=0)
    rec = recall_score(y_data, y_pred_cv, zero_division=0)
    f1  = f1_score(y_data, y_pred_cv, zero_division=0)
    return {"pipeline_name": name, "precision": prec, "recall": rec, "f1": f1, "y_pred_cv": y_pred_cv}

results = []


"\n"
"\n"
# Evaluar básico y avanzado
for mname, m in models.items():
    pipe_basic = Pipeline([("prep", preprocess_basic), ("model", m)])
    results.append(evaluate_with_cv(X, y, pipe_basic, f"basic_{mname}"))

    pipe_adv = Pipeline([("prep", preprocess_advanced), ("select", SelectKBest(score_func=f_classif, k="all")), ("model", m)])
    results.append(evaluate_with_cv(X, y, pipe_adv, f"advanced_{mname}"))

metrics_df = pd.DataFrame(results)[["pipeline_name", "precision", "recall", "f1"]].sort_values(by="f1", ascending=False)

st.markdown("""
        <div style='
            background-color: #F2F2F2;
            padding: 10px;
            border-radius: 12px;
            margin-top: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            text-align: center;
            '>
            <h4 style='color: #0033A0;'>📈 Tabla de Métricas (ordenada por F1)</h4>
            </div>
        """, unsafe_allow_html=True)

st.dataframe(metrics_df.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1": "{:.3f}"}))



# Selección del mejor modelo
best_row = metrics_df.iloc[0]
best_name = best_row["pipeline_name"]
st.success(f"✅ Mejor pipeline: {best_name}")

"\n"
"\n"
# Reconstruir pipeline
def make_pipeline_by_name(name):
    kind, algo = name.split("_")
    if algo == "KNN":
        model = KNeighborsClassifier(n_neighbors=7, weights="distance")
    elif algo == "LR":
        model = LogisticRegression(max_iter=500, solver="lbfgs")
    else:
        model = DecisionTreeClassifier(max_depth=6, random_state=42)

    if kind == "basic":
        pipeline = Pipeline([("prep", preprocess_basic), ("model", model)])
    else:
        pipeline = Pipeline([("prep", preprocess_advanced), ("select", SelectKBest(score_func=f_classif, k='all')), ("model", model)])
    return pipeline, kind

best_pipeline, best_kind = make_pipeline_by_name(best_name)
best_pipeline.fit(X, y)



"\n"
"\n"
# Predicciones en X_test
y_hat = best_pipeline.predict(X_test)

st.markdown("""
        <div style='
            background-color: #F2F2F2;
            padding: 10px;
            border-radius: 12px;
            margin-top: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            text-align: center;
            '>
            <h4 style='color: #0033A0;'>🔮 Vector de Predicciones (y_hat)</h4>
            </div>
        """, unsafe_allow_html=True)

# Usar 'ticket' como identificador del pasajero
identificador = X_test["ticket"] if "ticket" in X_test.columns else pd.Series([f"Pasajero {i}" for i in range(len(y_hat))])

# Construir tabla con nombre, predicción e interpretación
# Copia X_test para no modificar el original
tabla_predicciones = X_test.copy()

# Añade columna de predicción
tabla_predicciones["Predicción"] = y_hat

# Añade interpretación legible
tabla_predicciones["Interpretación"] = ["Sobrevivió" if val == 1 else "No sobrevivió" for val in y_hat]

# Convertir 'age' y 'fare' a enteros (redondeando)
tabla_predicciones["age"] = tabla_predicciones["age"].round(0).astype("Int64")
tabla_predicciones["fare"] = tabla_predicciones["fare"].round(0).astype("Int64")

# Reordenar columnas para que 'ticket', 'Predicción' e 'Interpretación' estén al inicio
cols = ["ticket", "Predicción", "Interpretación"] + [col for col in tabla_predicciones.columns if col not in ["ticket", "Predicción", "Interpretación"]]
tabla_predicciones = tabla_predicciones[cols]

# Mostrar tabla con estilo
st.dataframe(
    tabla_predicciones.style.set_properties(**{
        'text-align': 'center',
        'font-size': '14px'
    }).set_table_styles([{
        'selector': 'th',
        'props': [('font-size', '15px'), ('text-align', 'center')]
    }])
)






# Estadísticas de predicción
total = len(y_hat)
sobrevivientes = np.sum(y_hat)
no_sobrevivientes = total - sobrevivientes



st.markdown(f"""
            <div style="background-color:#F2F2F2; padding:5px; border-left:10px solid #0033A0; margin-top:0px;">
                <h4 style="color:#0033A0;">📊 Resumen de Predicciones:</h4>
                <p style="color:#0033A0;"> * 0 → El modelo predice que el pasajero <b>no sobrevivió</b>.</p>
                <p style="color:#0033A0;">* 1 → El modelo predice que el pasajero <b>sí sobrevivió</b>.</p>
                <p style="color:#0033A0;">* El campo <b>Pasajero</b> usa el número de boleto como identificador.</p>
                <p style="color:#0033A0;">* Total de pasajeros: <b>{total}</b></p>
                <p style="color:#0033A0;">* Predicción: <b>{sobrevivientes} sobrevivieron</b>, <b>{no_sobrevivientes} no sobrevivieron</b>.</p> 
            </div>
            """, unsafe_allow_html=True)


"\n"
"\n"
"\n"
"\n"
"\n"
"\n"
"\n"


st.markdown("""
        <div style='
            background-color: #F2F2F2;
            padding: 10px;
            border-radius: 12px;
            margin-top: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            text-align: center;
            '>
            <h4 style='color: #0033A0;'>"📈 Matriz CV"</h4>
            </div>
        """, unsafe_allow_html=True)

# Matriz de confusión
y_pred_cv_best = [r for r in results if r["pipeline_name"] == best_name][0]["y_pred_cv"]
cm = confusion_matrix(y, y_pred_cv_best)

fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_title("Matriz de Confusión (CV)")
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
st.pyplot(fig)




"\n"
"\n"
"\n"
st.markdown("""
        <div style='
            background-color: #F2F2F2;
            padding: 10px;
            border-radius: 12px;
            margin-top: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            text-align: center;
            '>
            <h4 style='color: #0033A0;'>"📈 Análisis de Distribución: Gráficas Q-Q"</h4>
            </div>
        """, unsafe_allow_html=True)

# Selección de variables numéricas para análisis
variables_qq = ["age", "fare"]

for var in variables_qq:
    fig_qq, ax_qq = plt.subplots(figsize=(5, 4))
    # Eliminar nulos antes de graficar
    datos = train[var].dropna()
    stats.probplot(datos, dist="norm", plot=ax_qq)
    ax_qq.set_title(f"Gráfica Q-Q para '{var}'")
    ax_qq.set_xlabel("Cuantiles teóricos")
    ax_qq.set_ylabel("Cuantiles de los datos")
    st.pyplot(fig_qq)



"\n"
"\n"
"\n"
"\n"
"\n"

st.markdown("""
        <div style='
            background-color: #F2F2F2;
            padding: 10px;
            border-radius: 12px;
            margin-top: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            text-align: center;
            '>
            <h4 style='color: #0033A0;'>"📈 Distribución 2D"</h4>
            </div>
        """, unsafe_allow_html=True)

# Scatter PCA
X_transformed = best_pipeline.named_steps["prep"].fit_transform(X)
if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()
pca = PCA(n_components=2, random_state=42)
X2 = pca.fit_transform(X_transformed)

fig2, ax2 = plt.subplots(figsize=(6,5))
scatter = ax2.scatter(X2[:,0], X2[:,1], c=y, cmap="coolwarm", s=40, alpha=0.6, label="true")
ax2.scatter(X2[:,0], X2[:,1], c=y_pred_cv_best, cmap="coolwarm", marker="x", s=25, alpha=0.6, label="prediction")
ax2.set_title("Distribución 2D (PCA) de clases verdaderas y predichas")
ax2.set_xlabel("x1")
ax2.set_ylabel("x2")
ax2.legend(loc="upper right")
st.pyplot(fig2)




st.markdown("""
        <div style='
            background-color: #F2F2F2;
            padding: 10px;
            border-radius: 12px;
            margin-top: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            text-align: center;
            '>
            <h4 style='color: #0033A0;'>✅ "Este dashboard muestra los datos, métricas, predicciones y gráficas de los modelos entrenados."</h4>
            </div>
        """, unsafe_allow_html=True)
