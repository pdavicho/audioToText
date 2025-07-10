import streamlit as st
from pydub import AudioSegment, silence
import os
import zipfile
import tempfile
import shutil
import time
from typing import List, Tuple, Dict
from dataclasses import dataclass
import io

st.set_page_config(
    page_title="Recortar Audios Extensos", 
    page_icon="✂️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicializar session state
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'segments_info' not in st.session_state:
    st.session_state.segments_info = []
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = None

@dataclass
class SegmentInfo:
    filename: str
    filepath: str
    duration_seconds: float
    start_time: float
    end_time: float
    file_size_mb: float

def get_audio_info(file_path: str) -> Dict:
    """Obtener información básica del archivo de audio"""
    try:
        audio = AudioSegment.from_file(file_path)
        duration_seconds = len(audio) / 1000
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        
        return {
            'duration_seconds': duration_seconds,
            'duration_formatted': format_duration(duration_seconds),
            'file_size_mb': file_size_mb,
            'sample_rate': audio.frame_rate,
            'channels': audio.channels,
            'format': audio.sample_width * 8,  # bits per sample
            'success': True
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def format_duration(seconds: float) -> str:
    """Formatear duración en formato legible"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

def calculate_estimated_segments(duration_seconds: float, interval_minutes: int) -> int:
    """Calcular número estimado de segmentos"""
    interval_seconds = interval_minutes * 60
    return int(duration_seconds / interval_seconds) + (1 if duration_seconds % interval_seconds > 0 else 0)

def divide_audio_advanced(
    file_path: str, 
    interval_minutes: int = 2,
    silence_detection: bool = True,
    min_silence_len: int = 1000,
    silence_thresh_adjustment: int = 16,
    fade_duration: int = 100,
    output_format: str = "mp3",
    output_quality: str = "medium"
) -> Tuple[List[SegmentInfo], str]:
    """
    Función avanzada para dividir audio con múltiples opciones
    """
    try:
        # Crear directorio temporal único
        temp_dir = tempfile.mkdtemp(prefix="audio_segments_")
        
        # Cargar el archivo de audio
        audio = AudioSegment.from_file(file_path)
        
        # Configurar calidad de salida
        bitrate_map = {
            "low": "64k",
            "medium": "128k", 
            "high": "192k",
            "very_high": "320k"
        }
        export_bitrate = bitrate_map.get(output_quality, "128k")
        
        # Convertir minutos a milisegundos
        interval_ms = interval_minutes * 60 * 1000
        
        # Configurar umbral de silencio
        silence_thresh = audio.dBFS - silence_thresh_adjustment
        
        # Variables para segmentación
        start = 0
        segment_count = 1
        segments_info = []
        
        total_duration = len(audio)
        
        while start < total_duration:
            end = min(start + interval_ms, total_duration)
            segment = audio[start:end]
            
            # Detectar silencios si está habilitado
            if silence_detection and end < total_duration:
                try:
                    silent_ranges = silence.detect_silence(
                        segment, 
                        min_silence_len=min_silence_len, 
                        silence_thresh=silence_thresh
                    )
                    
                    # Ajustar el corte al último silencio si existe
                    if silent_ranges:
                        # Buscar silencio cerca del final del segmento
                        for silence_start, silence_end in reversed(silent_ranges):
                            if silence_start >= (len(segment) - 30 * 1000):  # Últimos 30 segundos
                                end = start + silence_start
                                segment = audio[start:end]
                                break
                except Exception:
                    # Si falla la detección de silencio, continuar sin ella
                    pass
            
            # Aplicar fade in/out si está configurado
            if fade_duration > 0:
                segment = segment.fade_in(fade_duration).fade_out(fade_duration)
            
            # Generar nombre del archivo
            segment_filename = f"audio_seg_{segment_count:03d}.{output_format}"
            segment_filepath = os.path.join(temp_dir, segment_filename)
            
            # Exportar segmento
            segment.export(
                segment_filepath, 
                format=output_format,
                bitrate=export_bitrate
            )
            
            # Crear información del segmento
            segment_info = SegmentInfo(
                filename=segment_filename,
                filepath=segment_filepath,
                duration_seconds=len(segment) / 1000,
                start_time=start / 1000,
                end_time=end / 1000,
                file_size_mb=os.path.getsize(segment_filepath) / (1024 * 1024)
            )
            
            segments_info.append(segment_info)
            
            # Actualizar contadores
            start = end
            segment_count += 1
            
            # Yield progress para el progress bar
            progress = min(start / total_duration, 1.0)
            yield progress, segments_info
        
        yield 1.0, segments_info  # Completado
        
    except Exception as e:
        raise Exception(f"Error procesando audio: {str(e)}")

def create_zip_advanced(segments: List[SegmentInfo], include_metadata: bool = True) -> bytes:
    """Crear ZIP con los segmentos y metadata opcional"""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Agregar archivos de audio
        for segment in segments:
            if os.path.exists(segment.filepath):
                zipf.write(segment.filepath, segment.filename)
        
        # Agregar metadata si está habilitado
        if include_metadata:
            metadata_content = create_metadata_file(segments)
            zipf.writestr("SEGMENTOS_INFO.txt", metadata_content)
    
    zip_buffer.seek(0)
    return zip_buffer.read()

def create_metadata_file(segments: List[SegmentInfo]) -> str:
    """Crear archivo de metadata con información de los segmentos"""
    content = "📊 INFORMACIÓN DE SEGMENTOS DE AUDIO\n"
    content += "=" * 50 + "\n\n"
    
    total_duration = sum(seg.duration_seconds for seg in segments)
    total_size = sum(seg.file_size_mb for seg in segments)
    
    content += f"📈 RESUMEN GENERAL:\n"
    content += f"• Total de segmentos: {len(segments)}\n"
    content += f"• Duración total: {format_duration(total_duration)}\n"
    content += f"• Tamaño total: {total_size:.2f} MB\n"
    content += f"• Duración promedio por segmento: {format_duration(total_duration/len(segments))}\n\n"
    
    content += "📋 DETALLE POR SEGMENTO:\n"
    content += "-" * 50 + "\n"
    
    for i, segment in enumerate(segments, 1):
        content += f"{i:2d}. {segment.filename}\n"
        content += f"    ⏱️  Duración: {format_duration(segment.duration_seconds)}\n"
        content += f"    📏 Tiempo: {format_duration(segment.start_time)} → {format_duration(segment.end_time)}\n"
        content += f"    💾 Tamaño: {segment.file_size_mb:.2f} MB\n"
        content += "\n"
    
    return content

def cleanup_temp_files():
    """Limpiar archivos temporales"""
    if st.session_state.temp_dir and os.path.exists(st.session_state.temp_dir):
        try:
            shutil.rmtree(st.session_state.temp_dir)
            st.session_state.temp_dir = None
            return True
        except:
            return False
    return True

def main():
    st.title("✂️ Recortar Audios Extensos")
    st.markdown("*Divide archivos de audio extensos*")
    st.markdown("---")
    
    # Sidebar con configuraciones
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # Configuración básica
        st.subheader("📏 Segmentación")
        interval = st.slider(
            "Duración por segmento (minutos):",
            min_value=1,
            max_value=30,
            value=2,
            step=1
        )
        
        # Configuración de detección de silencio
        st.subheader("🔇 Detección de Silencio")
        silence_detection = st.checkbox("Activar detección inteligente", value=True)
        
        if silence_detection:
            min_silence_len = st.slider(
                "Duración mínima de silencio (ms):",
                min_value=500,
                max_value=3000,
                value=1000,
                step=250
            )
            
            silence_thresh_adj = st.slider(
                "Sensibilidad de detección:",
                min_value=10,
                max_value=30,
                value=16,
                step=2,
                help="Valores menores = más sensible"
            )
        else:
            min_silence_len = 1000
            silence_thresh_adj = 16
        
        # Configuración de calidad
        st.subheader("🎵 Calidad de Salida")
        output_format = st.selectbox(
            "Formato:",
            ["mp3", "wav", "m4a"],
            index=0
        )
        
        output_quality = st.selectbox(
            "Calidad:",
            ["low", "medium", "high", "very_high"],
            index=1,
            format_func=lambda x: {
                "low": "Baja (64kbps)",
                "medium": "Media (128kbps)", 
                "high": "Alta (192kbps)",
                "very_high": "Muy Alta (320kbps)"
            }[x]
        )
        
        # Configuración adicional
        #st.subheader("🎛️ Efectos")
        #fade_duration = st.slider(
        #    "Fade in/out (ms):",
        #    min_value=0,
        #    max_value=1000,
        #    value=100,
        #    step=50
        #)
        
        include_metadata = st.checkbox("Incluir archivo de información", value=True)
        
        # Botón de limpieza
        if st.button("🗑️ Limpiar archivos temporales"):
            if cleanup_temp_files():
                st.success("Archivos limpiados")
                st.session_state.processing_complete = False
                st.session_state.segments_info = []
    
    # Área principal
    uploaded_file = st.file_uploader(
        "📁 Sube tu archivo de audio",
        type=["mp3", "wav", "m4a", "flac", "aac", "ogg"],
        help="Formatos soportados: MP3, WAV, M4A, FLAC, AAC, OGG"
    )
    
    if uploaded_file is not None:
        # Guardar archivo temporal
        temp_file_path = f"temp_{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Obtener información del archivo
        audio_info = get_audio_info(temp_file_path)
        
        if not audio_info['success']:
            st.error(f"❌ Error al cargar el archivo: {audio_info['error']}")
            os.remove(temp_file_path)
            return
        
        # Mostrar información del archivo
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Información del Archivo")
            st.write(f"**📄 Nombre:** {uploaded_file.name}")
            st.write(f"**⏱️ Duración:** {audio_info['duration_formatted']}")
            st.write(f"**💾 Tamaño:** {audio_info['file_size_mb']:.2f} MB")
            st.write(f"**🎵 Calidad:** {audio_info['sample_rate']} Hz, {audio_info['channels']} canal(es), {audio_info['format']} bits")
        
        with col2:
            st.markdown("### 📈 Estimación de Resultado")
            estimated_segments = calculate_estimated_segments(audio_info['duration_seconds'], interval)
            est_size_per_segment = audio_info['file_size_mb'] / estimated_segments
            
            st.write(f"**🧩 Segmentos estimados:** {estimated_segments}")
            st.write(f"**📏 Duración por segmento:** ~{interval} minutos")
            st.write(f"**💾 Tamaño por segmento:** ~{est_size_per_segment:.2f} MB")
            st.write(f"**📦 Tamaño ZIP estimado:** ~{audio_info['file_size_mb'] * 0.9:.2f} MB")
        
        # Botón de procesamiento
        if st.button("✂️ Dividir Audio", type="primary", use_container_width=True):
            try:
                # Configurar contenedores para el progreso
                progress_container = st.container()
                info_container = st.container()
                
                with progress_container:
                    st.markdown("### 🔄 Procesando Audio...")
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Procesar audio
                    start_time = time.time()
                    segments_info = []
                    
                    processor = divide_audio_advanced(
                        temp_file_path,
                        interval_minutes=interval,
                        silence_detection=silence_detection,
                        min_silence_len=min_silence_len,
                        silence_thresh_adjustment=silence_thresh_adj,
                        #fade_duration=fade_duration,
                        output_format=output_format,
                        output_quality=output_quality
                    )
                    
                    # Actualizar progreso
                    for progress, current_segments in processor:
                        progress_bar.progress(progress)
                        status_text.text(f"Procesando segmento {len(current_segments)}/{estimated_segments}...")
                        segments_info = current_segments
                    
                    processing_time = time.time() - start_time
                    
                    # Guardar información en session state
                    st.session_state.segments_info = segments_info
                    st.session_state.processing_complete = True
                    st.session_state.temp_dir = os.path.dirname(segments_info[0].filepath) if segments_info else None
                    
                    # Completar progreso
                    progress_bar.progress(1.0)
                    status_text.text(f"✅ Completado en {processing_time:.2f} segundos")
                
                # Mostrar resultados
                with info_container:
                    st.markdown("### 🎉 Procesamiento Completado")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Segmentos creados", len(segments_info))
                    with col2:
                        total_duration = sum(seg.duration_seconds for seg in segments_info)
                        st.metric("Duración total", format_duration(total_duration))
                    with col3:
                        total_size = sum(seg.file_size_mb for seg in segments_info)
                        st.metric("Tamaño total", f"{total_size:.2f} MB")
                    with col4:
                        st.metric("Tiempo de proceso", f"{processing_time:.2f}s")
            
            except Exception as e:
                st.error(f"❌ Error durante el procesamiento: {str(e)}")
            finally:
                # Limpiar archivo temporal original
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
    
    # Mostrar resultados si hay procesamiento completado
    if st.session_state.processing_complete and st.session_state.segments_info:
        st.markdown("---")
        st.markdown("### 📦 Resultados")
        
        segments_info = st.session_state.segments_info
        
        # Botón de descarga
        col1, col2 = st.columns([3, 1])
        
        with col1:
            try:
                zip_data = create_zip_advanced(segments_info, include_metadata)
                
                st.download_button(
                    label="📥 Descargar Todos los Segmentos (ZIP)",
                    data=zip_data,
                    file_name=f"audio_segments_{int(time.time())}.zip",
                    mime="application/zip",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error generando ZIP: {e}")
        
        with col2:
            if st.button("🔄 Nuevo Procesamiento"):
                st.session_state.processing_complete = False
                st.session_state.segments_info = []
                cleanup_temp_files()
                st.rerun()
        
        # Lista detallada de segmentos
        with st.expander("📋 Detalle de Segmentos", expanded=False):
            for i, segment in enumerate(segments_info, 1):
                col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
                
                with col1:
                    st.write(f"**{i}. {segment.filename}**")
                with col2:
                    st.write(f"⏱️ {format_duration(segment.duration_seconds)}")
                with col3:
                    st.write(f"📏 {format_duration(segment.start_time)} → {format_duration(segment.end_time)}")
                with col4:
                    st.write(f"💾 {segment.file_size_mb:.2f} MB")
    
    # Información y ayuda
    if not uploaded_file:
        st.markdown("---")
        with st.expander("ℹ️ Información y Ayuda", expanded=False):
            st.markdown("""
            ### 🚀 Características principales:
            
            - **🎯 División inteligente**: Detecta silencios para cortes naturales
            - **⚙️ Múltiples formatos**: MP3, WAV, M4A, FLAC, AAC, OGG
            - **🎵 Control de calidad**: Desde 64kbps hasta 320kbps
            - **🔇 Detección de silencio**: Cortes en momentos apropiados
            - **📊 Metadata completa**: Información detallada de cada segmento
            - **💾 Descarga organizada**: ZIP con todos los archivos
            
            ### 📋 Consejos de uso:
            
            - **Duración recomendada**: 2-5 minutos por segmento
            - **Detección de silencio**: Actívala para cortes más naturales
            - **Calidad media**: Equilibrio perfecto entre tamaño y calidad
            
            
            ### 🔧 Configuraciones avanzadas:
            
            - **Sensibilidad de silencio**: Ajusta según el tipo de audio
            - **Duración mínima**: Evita cortes en pausas muy breves
            - **Formatos de salida**: Elige según tu necesidad final
            """)

if __name__ == "__main__":
    main()
