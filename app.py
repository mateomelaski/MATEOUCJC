import streamlit as st
import pandas as pd
import plotly.express as px
import time
import io

# Configuración de la página
st.set_page_config(page_title="Analizador de Clima Laboral (Demo)", page_icon="🏢", layout="wide")

st.title("HR Analytics: Analizador de Clima Laboral 🏢 (Versión Demo)")
st.markdown("""
Esta es una **versión de demostración** que utiliza un clasificador basado en palabras clave para emular el comportamiento de una IA. 
La herramienta real (como se describe en el diseño) se conectaría a la API de Gemini para procesar el lenguaje natural.

**⚠️ Aviso de Privacidad y RGPD:** Asegúrate de que el archivo contiene comentarios anónimos sin datos personales identificables. Esta aplicación no almacena ningún dato en disco; todo se procesa en la memoria de la sesión temporal.
""")

# Barra lateral para configuración
with st.sidebar:
    st.header("Configuración ⚙️")
    
    # Lista cerrada de temas (Configurable en la demo)
    temas_defecto = "Liderazgo, Compensación, Cultura, Carga de trabajo, Instalaciones, Comunicación, Desarrollo, Evaluación"
    temas_input = st.text_input("Temas de interés (separados por coma):", value=temas_defecto)
    st.caption("Nota de la demo: El clasificador heurístico tiene palabras clave mapeadas para los temas por defecto. Si cambias los nombres (ej. 'Salario' en vez de 'Compensación'), puede que no los reconozca bien.")
    temas_validos = [t.strip().lower() for t in temas_input.split(',')]
    
    uploaded_file = st.file_uploader("Sube los resultados de la encuesta (CSV o Excel)", type=["csv", "xlsx", "xls"])

# Función de clasificación heurística (Demo sin IA)
def keyword_classifier(comment, valid_topics):
    comment_lower = str(comment).lower()
    
    # 1. Determinar Sentimiento por palabras clave
    positivas = ["excelente", "buen", "me encanta", "feliz", "bien", "genial", "apoyo", "disponible", "guiar", "guía"]
    negativas = ["mal", "terrible", "poco", "estrés", "falta", "injusto", "excesiv", "debajo", "necesitan mejorar", "no son"]
    
    score = 0
    for w in positivas:
        if w in comment_lower: score += 1
    for w in negativas:
        if w in comment_lower: score -= 1
        
    if score > 0:
        sentimiento = "Positivo"
    elif score < 0:
        sentimiento = "Negativo"
    else:
        sentimiento = "Neutral"

    # 2. Determinar Tema por palabras clave
    tema = "Otro / No clasificado"
    diccionario_temas = {
        "liderazgo": ["jefe", "líder", "manager", "direct", "guía", "jefatura"],
        "compensación": ["sueldo", "salario", "pago", "beneficio", "dinero", "bono"],
        "cultura": ["ambiente", "equipo", "amigables", "cultura", "compañer", "apoyo"],
        "carga de trabajo": ["horas", "estrés", "carga", "excesiva", "equilibrio"],
        "instalaciones": ["oficina", "silla", "baño", "instalaciones", "espacio", "sillas"],
        "comunicación": ["comunicación", "objetivos", "información", "largo plazo"],
        "desarrollo": ["curso", "desarrollo", "profesional", "crecimiento", "oportunidades"],
        "evaluación": ["evaluación", "desempeño", "feedback"]
    }
    
    for t_valido in valid_topics:
        if t_valido in diccionario_temas:
            if any(palabra in comment_lower for palabra in diccionario_temas[t_valido]):
                tema = next((t.strip() for t in temas_input.split(',') if t.strip().lower() == t_valido), t_valido.title())
                break
                
    if tema == "Otro / No clasificado":
        for t_valido in valid_topics:
            if t_valido in comment_lower:
                tema = next((t.strip() for t in temas_input.split(',') if t.strip().lower() == t_valido), t_valido.title())
                break
                
    return sentimiento, tema

if uploaded_file is not None:
    try:
        # Cache reset
        if (
            'df_procesado' not in st.session_state 
            or st.session_state.get('last_uploaded') != uploaded_file.name
            or st.session_state.get('last_temas') != temas_input
        ):
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            comentario_col = None
            for col in df.columns:
                if any(x in col.lower() for x in ["comentario", "respuesta", "feedback"]):
                    comentario_col = col
                    break
                    
            if comentario_col is None:
                st.error("No se encontró una columna llamada 'Comentario', 'Respuesta' o 'Feedback' en el archivo.")
                st.stop()
                
            with st.spinner('Analizando mediante palabras clave (Modo Demo)...'):
                time.sleep(1) # Simular latencia
                sentimientos = []
                temas = []
                for comment in df[comentario_col]:
                    s, t = keyword_classifier(comment, temas_validos)
                    sentimientos.append(s)
                    temas.append(t)
                
                df['Sentimiento'] = sentimientos
                df['Tema Principal'] = temas
                
                st.session_state['df_procesado'] = df
                st.session_state['last_uploaded'] = uploaded_file.name
                st.session_state['last_temas'] = temas_input

        df = st.session_state['df_procesado']
        
        st.success("¡Análisis completado exitosamente!")
        
        # --- DASHBOARD ---
        st.header("📊 Dashboard de Resultados")
        
        col1, col2, col3 = st.columns(3)
        total_comentarios = len(df)
        positivos = len(df[df['Sentimiento'] == 'Positivo'])
        negativos = len(df[df['Sentimiento'] == 'Negativo'])
        
        pct_pos = (positivos/total_comentarios)*100 if total_comentarios > 0 else 0
        pct_neg = (negativos/total_comentarios)*100 if total_comentarios > 0 else 0
        
        col1.metric("Total de Comentarios", total_comentarios)
        col2.metric("Comentarios Positivos", f"{positivos} ({pct_pos:.1f}%)")
        col3.metric("Comentarios Negativos", f"{negativos} ({pct_neg:.1f}%)")
        
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.subheader("Distribución de Sentimiento")
            fig_sentiment = px.pie(df, names='Sentimiento', color='Sentimiento',
                                 color_discrete_map={'Positivo':'#2ecc71', 'Neutral':'#95a5a6', 'Negativo':'#e74c3c'})
            st.plotly_chart(fig_sentiment)
            
        with col_chart2:
            st.subheader("Temas Principales")
            tema_counts = df['Tema Principal'].value_counts().reset_index()
            tema_counts.columns = ['Tema Principal', 'Cantidad']
            fig_temas = px.bar(tema_counts, x='Tema Principal', y='Cantidad', color='Tema Principal')
            st.plotly_chart(fig_temas)
        
        # Resumen Ejecutivo Dinámico
        st.subheader("🤖 Resumen Ejecutivo (Construido Dinámicamente)")
        tema_principal_negativo = None
        if negativos > 0:
            tema_principal_negativo = df[df['Sentimiento'] == 'Negativo']['Tema Principal'].mode()[0]
        tema_principal_positivo = None
        if positivos > 0:
            tema_principal_positivo = df[df['Sentimiento'] == 'Positivo']['Tema Principal'].mode()[0]

        resumen_texto = f"Se han analizado **{total_comentarios} comentarios**. "
        if pct_pos > pct_neg:
            resumen_texto += f"El clima general es mayormente **positivo ({pct_pos:.1f}%)**. "
        else:
            resumen_texto += f"Existe una proporción significativa de comentarios **negativos ({pct_neg:.1f}%)**. "
            
        if tema_principal_positivo:
            resumen_texto += f"Los empleados valoran especialmente aspectos relacionados con **{tema_principal_positivo}**. "
        if tema_principal_negativo:
            resumen_texto += f"Sin embargo, el área más crítica que requiere atención es **{tema_principal_negativo}**, ya que concentra la mayor cantidad de feedback negativo."
            
        st.info(resumen_texto)
        
        # Tabla
        st.subheader("📝 Datos Detallados")
        col_filtro1, col_filtro2 = st.columns(2)
        with col_filtro1:
            filtro_sentimiento = st.selectbox("Filtrar por Sentimiento:", ["Todos"] + list(df['Sentimiento'].unique()))
        with col_filtro2:
            filtro_tema = st.selectbox("Filtrar por Tema:", ["Todos"] + list(df['Tema Principal'].unique()))
            
        df_filtrado = df.copy()
        if filtro_sentimiento != "Todos":
            df_filtrado = df_filtrado[df_filtrado['Sentimiento'] == filtro_sentimiento]
        if filtro_tema != "Todos":
            df_filtrado = df_filtrado[df_filtrado['Tema Principal'] == filtro_tema]
            
        st.dataframe(df_filtrado)
            
        # Descargas
        st.markdown("### Descargar Resultados")
        col_dl1, col_dl2 = st.columns(2)
        
        csv_export = df_filtrado.to_csv(index=False).encode('utf-8')
        with col_dl1:
            st.download_button("📥 Descargar CSV", csv_export, 'analisis_clima_hr.csv', 'text/csv')
            
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df_filtrado.to_excel(writer, index=False, sheet_name='Resultados')
        with col_dl2:
            st.download_button("📥 Descargar Excel", buffer.getvalue(), 'analisis_clima_hr.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")

elif uploaded_file is None:
    st.info("Por favor, sube un archivo CSV o Excel con las encuestas para comenzar.")
