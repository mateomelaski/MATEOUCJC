import streamlit as st
import pandas as pd
import plotly.express as px
import time
import io

# ---------------------------------------------------------
# Configuración global
# ---------------------------------------------------------
st.set_page_config(page_title="HR Analytics · Clima laboral", page_icon="🏢", layout="wide")

COLOR_MAP = {'Positivo': '#2ecc71', 'Neutral': '#95a5a6', 'Negativo': '#e74c3c'}

# ---------------------------------------------------------
# Funciones de backend (Simulación local sin coste)
# ---------------------------------------------------------
def keyword_classifier(comment, valid_topics):
    """Clasifica sentimientos y temas localmente mediante heurística."""
    comment_lower = str(comment).lower()
    
    # 1. Determinar Sentimiento
    positivas = ["excelente", "buen", "me encanta", "feliz", "bien", "genial", "apoyo", "disponible", "guiar", "guía"]
    negativas = ["mal", "terrible", "poco", "estrés", "falta", "injusto", "excesiv", "debajo", "necesitan mejorar", "no son"]
    
    score = sum(1 for w in positivas if w in comment_lower) - sum(1 for w in negativas if w in comment_lower)
    sentimiento = "Positivo" if score > 0 else "Negativo" if score < 0 else "Neutral"

    # 2. Determinar Tema
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
    
    for t_original in valid_topics:
        t_lower = t_original.lower()
        if t_lower in diccionario_temas:
            if any(p in comment_lower for p in diccionario_temas[t_lower]):
                return sentimiento, t_original
                
    # Fallback por coincidencia exacta
    for t_original in valid_topics:
        if t_original.lower() in comment_lower:
            return sentimiento, t_original
                
    return sentimiento, "Otro / No clasificado"

def classify_comments_mock(df_comments, temas):
    """Procesa el dataframe y actualiza la barra de progreso."""
    results = []
    total = len(df_comments)
    progress_bar = st.progress(0, text="Iniciando análisis...")
    
    for i, row in df_comments.iterrows():
        # Actualización de progreso fluida
        if i % max(1, total // 10) == 0:
            progress_bar.progress(i / total, text=f"Procesando comentario {i+1} de {total}...")
            time.sleep(0.05) # Pequeño retardo visual
            
        s, t = keyword_classifier(row['text'], temas)
        results.append({"id": row['id'], "sentimiento": s, "tema": t})
        
    progress_bar.progress(1.0, text="Análisis completado.")
    time.sleep(0.5)
    progress_bar.empty()
    return results

def generate_summary_mock(df_results):
    """Redacta un resumen ejecutivo dinámicamente basado en los datos."""
    total = len(df_results)
    if total == 0:
        return "No hay datos para resumir."
        
    pos = len(df_results[df_results['Sentimiento'] == 'Positivo'])
    neg = len(df_results[df_results['Sentimiento'] == 'Negativo'])
    
    pct_pos = (pos/total)*100
    pct_neg = (neg/total)*100
    
    tema_principal_negativo = None
    if neg > 0:
        tema_principal_negativo = df_results[df_results['Sentimiento'] == 'Negativo']['Tema'].mode()[0]
    tema_principal_positivo = None
    if pos > 0:
        tema_principal_positivo = df_results[df_results['Sentimiento'] == 'Positivo']['Tema'].mode()[0]

    resumen = f"El análisis revela que de un total de {total} comentarios procesados, "
    if pct_pos > pct_neg:
        resumen += f"el clima general es predominantemente positivo ({pct_pos:.1f}%). "
    else:
        resumen += f"existe una proporción significativa de insatisfacción ({pct_neg:.1f}% de comentarios negativos). "
        
    if tema_principal_positivo:
        resumen += f"Como puntos fuertes, los empleados valoran especialmente los aspectos relacionados con **{tema_principal_positivo}**. "
    if tema_principal_negativo:
        resumen += f"Sin embargo, el principal foco de mejora detectado es **{tema_principal_negativo}**, área que concentra la mayor parte del feedback crítico. "
        
    resumen += "\n\n**Acciones recomendadas:**\n1. Realizar sesiones de escucha activa sobre las áreas más criticadas.\n2. Reforzar y comunicar las políticas de bienestar.\n3. Monitorear la evolución del sentimiento en el próximo trimestre."
    
    return resumen


# ---------------------------------------------------------
# Interfaz de Usuario (UI)
# ---------------------------------------------------------
st.title("HR Analytics · Clima laboral")
st.markdown("Análisis automatizado de comentarios abiertos mediante Inteligencia Artificial.")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Configuración")
    
    temas_defecto = "Liderazgo, Compensación, Cultura, Carga de trabajo, Instalaciones, Comunicación, Desarrollo, Evaluación"
    temas_input = st.text_area("Temas de interés (separados por coma)", value=temas_defecto)
    temas_validos = [t.strip() for t in temas_input.split(',') if t.strip()]
    
    uploaded_file = st.file_uploader("Archivo de encuestas (CSV/Excel)", type=["csv", "xlsx", "xls"])
    analizar_btn = st.button("Analizar datos")

# --- MAIN ÁREA ---
if uploaded_file is None:
    st.info("Sube un archivo de encuestas en la barra lateral para comenzar.")
else:
    # 1. Leer archivo
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")
        st.stop()

    # 2. Detección automática de la columna
    comentario_col = None
    cols_lower = [c.lower() for c in df.columns]
    for orig, lower in zip(df.columns, cols_lower):
        if any(keyword in lower for keyword in ["comentario", "respuesta", "feedback"]):
            comentario_col = orig
            break
            
    if not comentario_col:
        comentario_col = st.selectbox("Selecciona la columna que contiene los comentarios:", df.columns)
        
    # Lógica de botón y estado
    if analizar_btn:
        df_target = df.copy()
        
        # Limitar a 300 filas
        if len(df_target) > 300:
            st.warning("El archivo tiene más de 300 filas. Se procesarán solo las primeras 300.")
            df_target = df_target.head(300)
            
        # Preparación de datos (asignación de ID temporal para correlación)
        df_target = df_target.reset_index(drop=True)
        df_target['id_interno'] = df_target.index
        
        df_to_process = pd.DataFrame({
            'id': df_target['id_interno'],
            'text': df_target[comentario_col].astype(str)
        })
        
        with st.spinner("Clasificando comentarios..."):
            resultados_json = classify_comments_mock(df_to_process, temas_validos)
            
        # Correlación de resultados
        res_dict = {item['id']: item for item in resultados_json}
        df_target['Sentimiento'] = df_target['id_interno'].map(lambda x: res_dict.get(x, {}).get('sentimiento', 'Neutral'))
        df_target['Tema'] = df_target['id_interno'].map(lambda x: res_dict.get(x, {}).get('tema', 'Otro / No clasificado'))
        df_target = df_target.drop(columns=['id_interno'])
        
        with st.spinner("Redactando resumen ejecutivo..."):
            resumen = generate_summary_mock(df_target)
            
        # Guardar en estado para persistir la vista
        st.session_state['df_analizado'] = df_target
        st.session_state['resumen_ejecutivo'] = resumen
        st.session_state['last_file'] = uploaded_file.name
        st.session_state['last_temas'] = temas_input
        
    # Comprobar si se ha modificado la configuración desde el último análisis
    needs_recalc = False
    if 'df_analizado' in st.session_state:
        if st.session_state.get('last_file') != uploaded_file.name or st.session_state.get('last_temas') != temas_input:
            needs_recalc = True
            
    if needs_recalc:
        st.warning("La configuración o el archivo han cambiado. Pulsa 'Analizar datos' para actualizar.")

    # --- DASHBOARD DE RESULTADOS ---
    if 'df_analizado' in st.session_state and not needs_recalc:
        df_result = st.session_state['df_analizado']
        resumen = st.session_state['resumen_ejecutivo']
        
        st.success("Análisis completado con éxito.")
        
        # 1. KPIs
        total = len(df_result)
        pos = len(df_result[df_result['Sentimiento'] == 'Positivo'])
        neg = len(df_result[df_result['Sentimiento'] == 'Negativo'])
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Comentarios", total)
        col2.metric("Positivos", f"{(pos/total)*100:.1f}%" if total > 0 else "0%")
        col3.metric("Negativos", f"{(neg/total)*100:.1f}%" if total > 0 else "0%")
        
        # 2. Resumen Ejecutivo
        st.subheader("Resumen Ejecutivo")
        st.info(resumen)
        
        # 3. Gráficos
        st.subheader("Análisis Visual")
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            fig_pie = px.pie(df_result, names='Sentimiento', color='Sentimiento', 
                             color_discrete_map=COLOR_MAP, title="Distribución Global del Clima")
            st.plotly_chart(fig_pie)
            
        with col_c2:
            tema_counts = df_result['Tema'].value_counts().reset_index()
            fig_bar = px.bar(tema_counts, x='Tema', y='count', title="Frecuencia de Temas")
            st.plotly_chart(fig_bar)
            
        tema_sent = df_result.groupby(['Tema', 'Sentimiento']).size().reset_index(name='Cantidad')
        fig_stack = px.bar(tema_sent, x='Tema', y='Cantidad', color='Sentimiento', 
                           color_discrete_map=COLOR_MAP, title="Impacto del Sentimiento por Tema", barmode='stack')
        st.plotly_chart(fig_stack)
        
        # 4. Datos y Exportación
        st.subheader("Desglose Detallado")
        
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            filtro_sent = st.selectbox("Sentimiento", ["Todos"] + list(df_result['Sentimiento'].unique()))
        with f_col2:
            filtro_tema = st.selectbox("Tema", ["Todos"] + list(df_result['Tema'].unique()))
            
        df_filtered = df_result.copy()
        if filtro_sent != "Todos":
            df_filtered = df_filtered[df_filtered['Sentimiento'] == filtro_sent]
        if filtro_tema != "Todos":
            df_filtered = df_filtered[df_filtered['Tema'] == filtro_tema]
            
        st.dataframe(df_filtered)
        
        # Botones de descarga
        dl_col1, dl_col2 = st.columns(2)
        csv_data = df_filtered.to_csv(index=False).encode('utf-8')
        dl_col1.download_button("Descargar CSV", csv_data, 'clima_laboral.csv', 'text/csv')
        
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df_filtered.to_excel(writer, index=False, sheet_name='Resultados')
        dl_col2.download_button("Descargar Excel", buffer.getvalue(), 'clima_laboral.xlsx', 
                                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
