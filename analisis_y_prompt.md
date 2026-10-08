# Trabajo: Herramienta de IA para Recursos Humanos

*Nota para el evaluador: La aplicación adjunta (`app.py`) es una versión demostrativa funcional que clasifica los textos mediante heurística (palabras clave) para mostrar el comportamiento de la interfaz y la lógica de la herramienta sin incurrir en costes de API reales.*

## 1. Análisis Funcional: Analizador de Encuestas de Clima Laboral

### Problema que se quiere resolver
Analizar manualmente respuestas abiertas de encuestas de clima requiere mucho tiempo y está sujeto a sesgos. Se necesita una herramienta automatizada y objetiva.

### Usuario objetivo
Profesionales de Recursos Humanos y HR Business Partners (HRBP).

### Datos de entrada (Input)
1.  **Archivo de datos:** CSV o Excel con comentarios anónimos.
2.  **Configuración de Análisis:** Temas de interés (ej. liderazgo, compensación).
3.  **API Key:** Clave de Gemini API configurada en secretos para el LLM.

### Funcionamiento paso a paso
1.  **Carga e Inicialización:** El usuario define temas clave y sube el archivo.
2.  **Análisis IA (Procesamiento):** La app agrupa los comentarios y los envía a Gemini (LLM). El modelo clasifica el sentimiento (Positivo/Negativo/Neutral) y asigna un tema de la lista predefinida, devolviendo un JSON estructurado.
3.  **Generación de visualizaciones:** Se almacenan los datos procesados y se generan un resumen global dinámico y estadísticas.
4.  **Presentación:** Dashboard con gráficos interactivos y tabla filtrable.

### Resultados esperados (Output)
1.  **KPIs y Gráficos:** Distribución de sentimientos y temas principales.
2.  **Resumen Ejecutivo:** Párrafo generado dinámicamente con las conclusiones.
3.  **Tabla de datos:** Comentarios clasificados, filtrables y exportables a CSV/Excel.

### Privacidad y Limitaciones
- **Privacidad y RGPD:** Solo se procesan comentarios anónimos. Los datos se mantienen en la memoria de la sesión (no se guardan en disco).
- **Limitaciones de la IA:** La herramienta es un apoyo analítico; las conclusiones estratégicas deben validarlas profesionales de HR.

---

## 2. Instrucción (Prompt) para generar el código

*Prompt para ChatGPT o Claude:*

```text
Actúa como desarrollador experto en Python y Streamlit. Escribe una app web para Recursos Humanos que analice respuestas abiertas de encuestas de clima usando IA.

### Contexto y Privacidad
- **Objetivo:** Clasificar el sentimiento y extraer los temas de los comentarios de los empleados.
- **Privacidad:** Muestra un aviso de RGPD (los datos deben ser anónimos y no se guardan en disco).

### Inputs
- Carga de archivo (CSV/Excel).
- Input de texto para definir temas de interés separados por comas.
- La API Key debe leerse de `st.secrets` (nunca en el código).

### Proceso (LLM)
- Usa la librería `google-genai` para llamar a la API de Gemini (gemini-2.5-flash).
- Llama a la API por lotes (ej. 20 filas por llamada) para procesar todo el CSV sin superar límites, con temperatura 0 para mayor consistencia.
- Pide al LLM que clasifique cada comentario devolviendo estrictamente un JSON con: Sentimiento (Positivo/Negativo/Neutral) y Tema (debe ser estrictamente uno de los definidos por el usuario).
- Guarda el dataframe resultante en `st.session_state` para no repetir la llamada a la API al cambiar filtros.

### Outputs e Interfaz
- **Sidebar:** Configuración de temas y subida de archivo.
- **Main Area:** Título, aviso de privacidad, y spinner durante el análisis.
- **Dashboard:** 
  - `st.columns` para métricas (Total, % Positivos, % Negativos).
  - Dos gráficos de `plotly.express`: pie chart de sentimientos y bar chart de temas.
  - Resumen Ejecutivo (calculado a partir de los datos procesados, destacando lo más positivo y el mayor problema) en un `st.info`.
- **Tabla y Exportación:** Muestra el dataframe con filtros interactivos de Sentimiento y Tema. Incluye botones de descarga en CSV y Excel (`application/vnd.openxmlformats-officedocument.spreadsheetml.sheet`).

Devuelve todo el código en un único archivo `app.py` e incluye `requirements.txt` con streamlit, pandas, plotly, google-genai y openpyxl.
```
