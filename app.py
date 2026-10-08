import streamlit as st
import pandas as pd
import plotly.express as px
import time
import io
import json

from google import genai
from google.genai import types

# ---------------------------------------------------------
# Configuración global
# ---------------------------------------------------------
st.set_page_config(page_title="HR Analytics · Clima laboral", page_icon="🏢", layout="wide")

MODEL = "gemini-2.5-flash"
COLOR_MAP = {'Positivo': '#2ecc71', 'Neutral': '#95a5a6', 'Negativo': '#e74c3c'}

# ---------------------------------------------------------
# Funciones de backend e IA
# ---------------------------------------------------------
def get_client(api_key):
    """Inicializa el cliente de Gemini API."""
    return genai.Client(api_key=api_key)

def _call_api_with_retries(client, batch_df, temas, max_retries=3):
    """
    Envía un lote de comentarios a la API de Gemini pidiendo JSON estructurado.
    Implementa reintentos con espera exponencial en caso de fallos.
    """
    batch_data = batch_df.to_dict('records')
    prompt_data = json.dumps(batch_data, ensure_ascii=False)
    
    temas_str = ", ".join(temas)
    
    # Instrucciones detalladas para forzar el comportamiento analítico
    system_instruction = f"""
    Eres un analista experto en Recursos Humanos. Tu tarea es analizar la siguiente lista de comentarios extraídos de una encuesta de clima laboral.
    Debes comprender el significado completo del comentario, detectando ironías, negaciones y contexto, sin dejarte llevar por palabras sueltas.
    Si un comentario es mixto (ej. "el salario es bajo pero el equipo es genial"), evalúa cuál es el sentimiento dominante o, si están equilibrados, clasifícalo como Neutral.
    
    Reglas estrictas para el JSON de salida:
    1. 'sentimiento' debe ser EXACTAMENTE uno de estos tres: Positivo, Negativo, Neutral.
    2. 'tema' debe ser EXACTAMENTE uno de los temas permitidos, o 'Otro' si ninguno encaja.
    Temas permitidos: {temas_str}.
    """
    
    # Definición de esquema para Structured Output en google-genai
    response_schema = types.Schema(
        type=types.Type.ARRAY,
        items=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "id": types.Schema(type=types.Type.INTEGER),
                "sentimiento": types.Schema(
                    type=types.Type.STRING, 
                    enum=["Positivo", "Negativo", "Neutral"]
                ),
                "tema": types.Schema(type=types.Type.STRING)
            },
            required=["id", "sentimiento", "tema"]
        )
    )
    
    config = types.GenerateContentConfig(
        temperature=0.0,
        response_mime_type="application/json",
        response_schema=response_schema,
        system_instruction=system_instruction
    )
    
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt_data,
                config=config
            )
            
            data = json.loads(response.text)
            
            # Validación de integridad de los IDs devueltos
            expected_ids = set(batch_df['id'])
            received_ids = set([item.get('id') for item in data if 'id' in item])
            
            if expected_ids.issubset(received_ids):
                return data
            else:
                raise ValueError("Faltan IDs en la respuesta del modelo.")
                
        except Exception as e:
            if attempt < max_retries:
                time.sleep((2 ** attempt) + 2)  # Backoff exponencial para sortear límite de cuota
            else:
                st.error(f"Error procesando lote tras {max_retries} intentos: {str(e)}")
                # Retorno de seguridad para no romper todo el proceso
                return [{"id": row['id'], "sentimiento": "Neutral", "tema": "Otro"} for _, row in batch_df.iterrows()]

def classify_comments(df_comments, temas, client):
    """
    Divide los comentarios en lotes de 20 para su análisis iterativo.
    """
    results = []
    batch_size = 20
    total_batches = (len(df_comments) + batch_size - 1) // batch_size
    
    progress_bar = st.progress(0, text="Iniciando análisis...")
    
    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(df_comments))
        batch = df_comments.iloc[start_idx:end_idx]
        
        progress_text = f"Procesando lote {i+1} de {total_batches} ({start_idx+1} al {end_idx})..."
        progress_bar.progress(i / total_batches, text=progress_text)
        
        batch_results = _call_api_with_retries(client, batch, temas)
        results.extend(batch_results)
        
        # Respetar rate limits de la capa gratuita entre lotes
        if i < total_batches - 1:
            time.sleep(4)
            
    progress_bar.progress(1.0, text="Análisis completado.")
    time.sleep(1)
    progress_bar.empty()
    
    return results

def generate_summary(client, df_results, comentario_col):
    """
    Genera un resumen ejecutivo extrayendo conclusiones globales.
    """
    total = len(df_results)
    if total == 0:
        return "No hay datos para resumir."
        
    pos = len(df_results[df_results['Sentimiento'] == 'Positivo'])
    neg = len(df_results[df_results['Sentimiento'] == 'Negativo'])
    neu = len(df_results[df_results['Sentimiento'] == 'Neutral'])
    
    temas_count = df_results['Tema'].value_counts().to_dict()
    
    # Extraer muestra representativa balanceada (máx 30)
    sample_size = min(30, total)
    sample_df = df_results.sample(n=sample_size, random_state=42)
    sample_comments = sample_df[comentario_col].tolist()
    
    prompt = f"""
    Eres el Director de Recursos Humanos. Basándote en los resultados de la última encuesta de clima, redacta un resumen ejecutivo en español de máximo 120 palabras.
    
    Estadísticas globales ({total} comentarios):
    - Positivos: {pos}
    - Negativos: {neg}
    - Neutrales: {neu}
    - Frecuencia por temas: {json.dumps(temas_count, ensure_ascii=False)}
    
    Muestra de comentarios (para aportar contexto cualitativo):
    {json.dumps(sample_comments, ensure_ascii=False)}
    
    El resumen debe incluir de forma concisa:
    1. El clima general.
    2. Los puntos fuertes principales.
    3. Los principales focos de mejora.
    4. 2-3 acciones recomendadas.
    """
    
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.3)
        )
        return response.text
    except Exception as e:
        return f"No se pudo generar el resumen ejecutivo: {str(e)}"


# ---------------------------------------------------------
# Interfaz de Usuario (UI)
# ---------------------------------------------------------
st.title("HR Analytics · Clima laboral")
st.markdown("Análisis automatizado de comentarios abiertos mediante Inteligencia Artificial.")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Configuración")
    
    # Lectura de API Key
    api_key = None
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    else:
        api_key = st.text_input("Gemini API Key", type="password")
        
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
        if not api_key:
            st.error("Es necesaria la API Key de Gemini para analizar.")
            st.stop()
            
        df_target = df.copy()
        
        # Limitar a 300 filas
        if len(df_target) > 300:
            st.warning("El archivo tiene más de 300 filas. Por seguridad y límites de cuota, solo se procesarán las primeras 300.")
            df_target = df_target.head(300)
            
        # Preparación de datos (asignación de ID temporal para correlación)
        df_target = df_target.reset_index(drop=True)
        df_target['id_interno'] = df_target.index
        
        df_to_process = pd.DataFrame({
            'id': df_target['id_interno'],
            'text': df_target[comentario_col].astype(str)
        })
        
        # Ejecución
        client = get_client(api_key)
        
        with st.spinner("Clasificando comentarios..."):
            resultados_json = classify_comments(df_to_process, temas_validos, client)
            
        # Correlación de resultados
        res_dict = {item['id']: item for item in resultados_json}
        df_target['Sentimiento'] = df_target['id_interno'].map(lambda x: res_dict.get(x, {}).get('sentimiento', 'Neutral'))
        df_target['Tema'] = df_target['id_interno'].map(lambda x: res_dict.get(x, {}).get('tema', 'Otro'))
        df_target = df_target.drop(columns=['id_interno'])
        
        with st.spinner("Redactando resumen ejecutivo..."):
            resumen = generate_summary(client, df_target, comentario_col)
            
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
