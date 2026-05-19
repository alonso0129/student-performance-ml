
import streamlit as st
import joblib
import numpy as np

# cargar modelos
modelo_lr = joblib.load("modelos/logistic_regression_model.pkl")
modelo_rf = joblib.load("modelos/random_forest_model.pkl")

# titulo
st.title("Predicción de Rendimiento Académico")

st.write("""
Esta aplicación utiliza Machine Learning para predecir
si un estudiante probablemente aprobará o desaprobará
según sus hábitos de estudio y comportamiento.
""")

# inputs

horas_estudio = st.slider(
    "Tiempo de estudio semanal",
    1, 4, 2
)

cursos_desaprobados = st.slider(
    "Número de cursos desaprobados anteriormente",
    0, 4, 0
)

faltas = st.slider(
    "Número de faltas",
    0, 100, 5
)

salidas_amigos = st.slider(
    "Frecuencia de salidas con amigos",
    1, 5, 3
)

alcohol_semana = st.slider(
    "Consumo de alcohol entre semana",
    1, 5, 1
)

alcohol_fin = st.slider(
    "Consumo de alcohol fines de semana",
    1, 5, 2
)

# selector modelo

modelo_seleccionado = st.selectbox(
    "Seleccione el modelo de Machine Learning",
    ["Logistic Regression", "Random Forest"]
)

# boton prediccion

if st.button("Predecir rendimiento académico"):

    datos = np.array([[
        horas_estudio,
        cursos_desaprobados,
        faltas,
        salidas_amigos,
        alcohol_semana,
        alcohol_fin
    ]])

    if modelo_seleccionado == "Logistic Regression":
        pred = modelo_lr.predict(datos)
    else:
        pred = modelo_rf.predict(datos)

    if pred[0] == 1:
        st.success("El estudiante probablemente APROBARÁ")
    else:
        st.error("El estudiante probablemente DESAPROBARÁ")

# informacion alumno

st.write("---")
st.write("Alumno: Alonso Javier Gaitán Manrique")
st.write("Código ISIL: 70389863")

st.write("Cuaderno Google Colab:")
st.write("https://colab.research.google.com/drive/1q7IYbQzSVu2m5way78BxXynI6-FcQ2Ed?usp=sharing")
