import streamlit as st
from datetime import datetime

st.set_page_config(
    page_title='VoiceWise AI',
    page_icon='logo_wise.png',
    layout='wide',
    initial_sidebar_state='expanded'
)

# CSS personalizado para mejorar el diseño
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #4472C4 0%, #3B5998 100%);  ← NUEVA LÍNEA
        color: white;
        padding: 3rem 2rem;
        border-radius: 15px;
        margin-bottom: 3rem;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .tool-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        border: 1px solid #e1e8ed;
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .tool-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.15);
        border-color: #667eea;
    }
    
    .tool-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 2rem;
        border: none;
        border-radius: 10px;
        font-size: 1.1rem;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
        text-decoration: none;
        display: inline-block;
        width: 100%;
        text-align: center;
        margin-top: 1rem;
    }
    
    .tool-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        color: white;
        text-decoration: none;
    }
    
    .stats-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);  
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .feature-list {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .footer-info {
        background: linear-gradient(135deg, #333 0%, #555 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-top: 3rem;
        text-align: center;
    }
    
    .icon-large {
        font-size: 4rem;
        margin-bottom: 1rem;
        display: block;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    # Header principal simplificado
    st.markdown("""
    <div class="main-header">
        <h1><span style="color: white;">Instituto Universitario Rumiñahui</h1>
        <h2><span style="color: white;">VoiceWise AI</h2>
        <p style="font-size: 1.2rem; margin-top: 1rem; opacity: 0.9; color: white;">
            | Departamento de Investigación | 
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Descripción del proyecto
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 🚀 Plataforma de Procesamiento de Audio
        
        Esta plataforma integra **tecnologías de inteligencia artificial** de última generación 
        para el procesamiento, análisis y transcripción de contenido de audio. Diseñada para el 
        análisis de contenido y procesamiento de grandes volúmenes de audio.
        
        ### 🎯 Características principales:
        - **IA Avanzada**: Tecnología OpenAI Whisper de última generación
        - **Procesamiento por lotes**: Manejo eficiente de múltiples archivos
        - **Análisis inteligente**: Detección y resaltado de palabras clave
        - **Interfaz intuitiva**: Diseño optimizado con Streamlit
        """)
    
    with col2:
        st.markdown("""
        <div class="stats-container">
        <h3><span style="color: white;">📊 Estadísticas del Sistema</h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1.5rem 0;">
            <div>
                <h2 style="margin: 0; font-size: 2.5rem;color: white;">3</h2>
                <p style="margin: 0;">Herramientas</p>
            </div>
            <div>
                <h2 style="margin: 0; font-size: 2.5rem;color: white;">AI</h2>
                <p style="margin: 0;">Powered by Whisper</p>
            </div>
            <div style="grid-column: 1 / -1;">
                <h2 style="margin: 0; font-size: 2.5rem;color: white;">24/7</h2>
                <p style="margin: 0;">Disponibilidad</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Herramientas disponibles con botones atractivos
    st.markdown("## 🛠️ Herramientas Disponibles")
    #st.markdown("*Selecciona la herramienta que necesitas para tu proyecto*")
    st.markdown("")
    
    # Grid de herramientas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="tool-card">
            <div class="icon-large">🎙️</div>
            <h3 style="text-align: center; color: #333; margin-bottom: 1rem;">Transcripción Individual</h3>
            <p style="color: #666; text-align: center; line-height: 1.6;">
                Convierte archivos de audio individuales a texto con análisis avanzado de palabras clave 
                y marcas de tiempo precisas.
            </p>
            <div class="feature-list">
                <strong>✨ Características:</strong>
                <ul style="margin: 0.5rem 0; padding-left: 1rem;">
                    <li>Transcripción con IA Whisper</li>
                    <li>Búsqueda de palabras clave</li>
                    <li>Marcas de tiempo precisas</li>
                    <li>Filtrado inteligente</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.page_link(
            "pages/1_🎙️_Audio_Texto.py", 
            label="🚀 Iniciar Transcripción Individual",
            icon="🎙️",
            use_container_width=True
        )
    
    with col2:
        st.markdown("""
        <div class="tool-card">
            <div class="icon-large">📦</div>
            <h3 style="text-align: center; color: #333; margin-bottom: 1rem;">Procesamiento Masivo</h3>
            <p style="color: #666; text-align: center; line-height: 1.6;">
                Procesa múltiples archivos de audio desde un ZIP. Ideal para análisis de grandes 
                volúmenes de contenido de forma automatizada.
            </p>
            <div class="feature-list">
                <strong>✨ Características:</strong>
                <ul style="margin: 0.5rem 0; padding-left: 1rem;">
                    <li>Procesamiento por lotes</li>
                    <li>Ordenamiento automático</li>
                    <li>Análisis estadístico</li>
                    <li>Descarga organizada</li>
                    <li>Reportes detallados</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.page_link(
            "pages/2_🎙️_Audio_Texto_Extenso.py", 
            label="🚀 Iniciar Procesamiento Masivo",
            icon="📦",
            use_container_width=True
        )
    
    with col3:
        st.markdown("""
        <div class="tool-card">
            <div class="icon-large">✂️</div>
            <h3 style="text-align: center; color: #333; margin-bottom: 1rem;">División de Audio</h3>
            <p style="color: #666; text-align: center; line-height: 1.6;">
                Divide archivos de audio largos en segmentos optimizados con detección automática 
                de silencios y control de calidad.
            </p>
            <div class="feature-list">
                <strong>✨ Características:</strong>
                <ul style="margin: 0.5rem 0; padding-left: 1rem;">
                    <li>Detección de silencios</li>
                    <li>Control de calidad</li>
                    <li>Efectos profesionales</li>
                    <li>Numeración automática</li>
                    <li>Metadata completa</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.page_link(
            "pages/3_✂️_Recortar_Audio.py", 
            label="🚀 Iniciar División de Audio",
            icon="✂️",
            use_container_width=True
        )
    
    st.markdown("---")
    
    # Sección de tecnologías
    st.markdown("## 🔬 Tecnologías Utilizadas")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🤖 Inteligencia Artificial
        - **OpenAI Whisper**: Transcripción de última generación
        - **Procesamiento NLP**: Análisis avanzado de texto
        - **Machine Learning**: Optimización continua
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Procesamiento de Datos
        - **Análisis de patrones**: Detección automática
        - **Estadísticas avanzadas**: Métricas detalladas
        - **Visualización**: Interfaces interactivas
        """)
    
    with col3:
        st.markdown("""
        ### 🛡️ Calidad y Seguridad
        - **Procesamiento Streamlit Cloud**: Datos seguros
        - **Validación robusta**: Control de errores
        - **Estándares**: Calidad garantizada
        """)
    
    # Footer institucional
    st.markdown(f"""
    <div class="footer-info">
        <h1><span style="color: white;">🏛️ Instituto Universitario Rumiñahui</span></h1>
        <p><strong>Departamento de Investigación</strong> | Innovación y Desarrollo Tecnológico</p>
        <p>Comprometidos con la excelencia académica y el avance científico</p>
        <p style="opacity: 0.8; font-size: 0.9rem; margin-top: 1.5rem;">
            Última actualización: {datetime.now().strftime("%d/%m/%Y")} | 
            Versión: 2.0 | 
            Desarrollado por: PDMN
        </p>
    </div>
    """, unsafe_allow_html=True)

# ✅ Mostrar el logo en el sidebar
with st.sidebar:
    st.image("logo_wise_2.png", width=150)
    
