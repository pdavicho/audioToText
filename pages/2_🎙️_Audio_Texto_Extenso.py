import streamlit as st
from streamlit_tags import st_tags
import warnings
warnings.filterwarnings('ignore')

import whisper
from whisper.utils import get_writer
import tempfile
import os
import time
import re
import zipfile
import shutil
from dataclasses import dataclass
from typing import List, Set, Tuple, Dict
from reportlab.lib.pagesizes import A4, letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.lib.colors import HexColor, black, blue, red
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas
from reportlab.platypus import Image
from datetime import datetime
import io


st.set_page_config(page_title='Audio Texto Extenso', page_icon=':studio_microphone:', layout="wide")

# Inicializar session state
if 'processing_results' not in st.session_state:
    st.session_state.processing_results = []
if 'current_temp_dir' not in st.session_state:
    st.session_state.current_temp_dir = None
if 'transcription_complete' not in st.session_state:
    st.session_state.transcription_complete = False
if 'keywords' not in st.session_state:
    st.session_state.keywords = []
if 'processing_keywords' not in st.session_state:
    st.session_state.processing_keywords = []
if 'total_processing_time' not in st.session_state:
    st.session_state.total_processing_time = 0
if 'show_results' not in st.session_state:
    st.session_state.show_results = False

@dataclass
class TranscriptionResult:
    filename: str
    filepath: str
    transcription: str
    duration: float
    processing_time: float
    found_keywords: List[str]
    word_count: int
    srt_path: str = None

@dataclass
class SRTSegment:
    index: int
    start_time: str
    end_time: str
    text: str
    contains_keywords: bool = False

@st.cache_resource
def load_whisper_model():
    """Load Whisper model with caching"""
    try:
        return whisper.load_model('base')
    except Exception as e:
        st.error(f"Error cargando modelo Whisper: {e}")
        return None

model = load_whisper_model()

def natural_sort_key(filename: str) -> tuple:
    """Genera una clave de ordenamiento natural para archivos con números"""
    import re
    parts = re.split(r'(\d+)', filename.lower())
    result = []
    for part in parts:
        if part.isdigit():
            result.append(int(part))
        else:
            result.append(part)
    return tuple(result)

def sort_audio_files(audio_files: List[str]) -> List[str]:
    """Ordena archivos de audio de manera inteligente"""
    files_with_names = [(os.path.basename(f), f) for f in audio_files]
    sorted_files = sorted(files_with_names, key=lambda x: natural_sort_key(x[0]))
    return [full_path for _, full_path in sorted_files]

def get_audio_files_from_zip(zip_file) -> Tuple[List[str], str]:
    """Extract and validate audio files from ZIP"""
    try:
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, "uploaded.zip")
        
        with open(zip_path, "wb") as f:
            f.write(zip_file.read())
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(temp_dir)
        
        audio_extensions = ('.wav', '.mp3', '.wave', '.m4a', '.flac', '.aac')
        audio_files = []
        
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.lower().endswith(audio_extensions):
                    full_path = os.path.join(root, file)
                    audio_files.append(full_path)
        
        audio_files = sort_audio_files(audio_files)
        return audio_files, temp_dir
        
    except zipfile.BadZipFile:
        st.error("El archivo no es un ZIP válido")
        return [], None
    except Exception as e:
        st.error(f"Error procesando ZIP: {e}")
        return [], None

def validate_audio_file(filepath: str) -> bool:
    """Validate if audio file can be processed"""
    try:
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            return False
        valid_extensions = ('.wav', '.mp3', '.wave', '.m4a', '.flac', '.aac')
        return filepath.lower().endswith(valid_extensions)
    except:
        return False

def get_transcribe_safe(audio_path: str, language: str = 'es') -> Dict:
    """Safe transcription with error handling"""
    try:
        if model is None:
            return {"error": "Modelo Whisper no disponible"}
        
        start_time = time.time()
        result = model.transcribe(audio=audio_path, language=language, verbose=False)
        processing_time = time.time() - start_time
        
        return {
            "text": result.get("text", ""),
            "segments": result.get("segments", []),
            "processing_time": processing_time,
            "error": None
        }
    except Exception as e:
        return {"error": f"Error transcribiendo: {str(e)}"}

def find_keywords_in_text(text: str, keywords: List[str]) -> List[str]:
    """Find which keywords are present in text"""
    found = []
    text_lower = text.lower()
    for keyword in keywords:
        if keyword and keyword.lower().strip() in text_lower:
            found.append(keyword)
    return found

def highlight_keywords(text: str, keywords: List[str]) -> str:
    """Highlight keywords in text"""
    highlighted = text
    for keyword in keywords:
        if keyword and keyword.strip():
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            highlighted = pattern.sub(
                lambda m: f'<mark style="background-color: #ffeb3b; color: #d32f2f; font-weight: bold;">{m.group()}</mark>',
                highlighted
            )
    return highlighted

def save_individual_files(result: Dict, filename: str, output_dir: str) -> Dict[str, str]:
    """Save transcription files for individual audio"""
    base_name = os.path.splitext(filename)[0]
    saved_files = {}
    
    try:
        txt_path = os.path.join(output_dir, f"{base_name}.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(result.get('text', ''))
        saved_files['txt'] = txt_path
        
        if result.get('segments'):
            srt_path = os.path.join(output_dir, f"{base_name}.srt")
            srt_content = []
            for i, segment in enumerate(result['segments'], 1):
                start = format_timestamp(segment['start'])
                end = format_timestamp(segment['end'])
                text = segment['text'].strip()
                srt_content.extend([str(i), f"{start} --> {end}", text, ""])
            
            with open(srt_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(srt_content))
            saved_files['srt'] = srt_path
        
    except Exception as e:
        st.warning(f"Error guardando archivos para {filename}: {e}")
    
    return saved_files

def format_timestamp(seconds):
    """Convert seconds to SRT timestamp format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def create_pdf_report(results: List[TranscriptionResult], keywords: List[str], processing_summary: Dict) -> bytes:
    """Genera un reporte PDF profesional"""
    buffer = io.BytesIO()
    
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=20*mm,
        leftMargin=20*mm,
        topMargin=25*mm,
        bottomMargin=20*mm,
        title="Reporte de Transcripción Masiva"
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=HexColor('#1e3c72'),
        fontName='Helvetica-Bold'
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=15,
        spaceBefore=20,
        textColor=HexColor('#2a5298'),
        fontName='Helvetica-Bold'
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=10,
        alignment=TA_JUSTIFY,
        fontName='Helvetica'
    )
    
    meta_style = ParagraphStyle(
        'MetaStyle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=HexColor('#666666'),
        fontName='Helvetica'
    )
    
    story = []
    
    # Header institucional con logo
    try:
        if os.path.exists('logo_instituto.png'):
            # Crear imagen del logo
            logo_img = Image('logo_instituto.png', width=1.5*inch, height=0.75*inch)
            
            # Crear párrafo con el texto institucional
            header_text = Paragraph("""
                <b>INSTITUTO UNIVERSITARIO RUMIÑAHUI</b><br/>
                <font size="12">Departamento de Investigación</font><br/>
                <font size="10" color="#666666">VoiceWise AI</font>
            """, ParagraphStyle(
                'HeaderText',
                parent=normal_style,
                fontSize=14,
                alignment=TA_LEFT,
                fontName='Helvetica-Bold'
            ))
            
            # Tabla con logo a la izquierda y texto a la derecha
            header_data = [[logo_img, header_text]]
            
            header_table = Table(header_data, colWidths=[2*inch, 5.5*inch])
            header_table.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('ALIGN', (0, 0), (0, 0), 'CENTER'),  # Logo centrado en su celda
                ('ALIGN', (1, 0), (1, 0), 'LEFT'),    # Texto alineado a la izquierda
                ('LEFTPADDING', (0, 0), (-1, -1), 10),
                ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                ('TOPPADDING', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ]))
            
            story.append(header_table)
        
        else:
            # Fallback sin logo
            header_content = """
            <para align="center">
                <b>🏛️ INSTITUTO UNIVERSITARIO RUMIÑAHUI</b><br/>
                <font size="12">Departamento de Investigación</font><br/>
                <font size="10" color="#666666">VoiceWise AI</font>
            </para>
            """
            story.append(Paragraph(header_content, normal_style))
        
    except Exception as e:
        # Fallback en caso de error con la imagen
        header_content = """
        <para align="center">
            <b>🏛️ INSTITUTO UNIVERSITARIO RUMIÑAHUI</b><br/>
            <font size="12">Departamento de Investigación</font><br/>
            <font size="10" color="#666666">VoiceWise AI</font>
        </para>
        """
        story.append(Paragraph(header_content, normal_style))

    story.append(Spacer(1, 20))
    
    story.append(Paragraph("📊 REPORTE DE TRANSCRIPCIÓN MASIVA", title_style))
    story.append(Spacer(1, 10))
    
    # Resumen ejecutivo
    story.append(Paragraph("📈 INFORMACIÓN DE LOS ARCHIVOS", subtitle_style))
    
    total_files = len(results)
    successful = len([r for r in results if r.transcription])
    total_duration = sum(r.duration for r in results)
    total_processing = sum(r.processing_time for r in results)
    total_words = sum(r.word_count for r in results)
    files_with_keywords = len([r for r in results if r.found_keywords])
    
    summary_data = [
        ['📁 Total de archivos procesados:', f"{total_files}"],
        ['✅ Transcripciones exitosas:', f"{successful}"],
        ['📅 Fecha de procesamiento:', datetime.now().strftime("%d/%m/%Y %H:%M:%S")],
        ['⏱️ Duración total de audio:', f"{total_duration:.1f} segundos ({total_duration/60:.1f} minutos)"],
        ['⚡ Tiempo total de procesamiento:', f"{total_processing:.1f} segundos"],
        ['📝 Total de palabras transcritas:', f"{total_words:,}"],
        ['🎯 Archivos con palabras clave:', f"{files_with_keywords}"],
        ['🔍 Palabras clave buscadas:', ", ".join(keywords) if keywords else "Ninguna"]
    ]
    
    summary_table = Table(summary_data, colWidths=[4.5*inch, 3*inch])
    summary_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0, 0), (0, -1), HexColor('#1e3c72')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    
    story.append(summary_table)
    story.append(Spacer(1, 20))
    
    # Análisis de palabras clave global
    if keywords:
        story.append(Paragraph("🎯 ANÁLISIS GLOBAL DE PALABRAS CLAVE", subtitle_style))
        
        keyword_stats = {}
        for keyword in keywords:
            keyword_stats[keyword] = len([r for r in results if keyword in r.found_keywords])
        
        if any(count > 0 for count in keyword_stats.values()):
            story.append(Paragraph("📊 <b>Estadísticas de aparición:</b>", normal_style))
            for keyword, count in keyword_stats.items():
                percentage = (count / total_files * 100) if total_files > 0 else 0
                color = "#2a5298" if count > 0 else "#666666"
                story.append(Paragraph(f"• <font color='{color}'><b>{keyword}:</b> {count} archivos ({percentage:.1f}%)</font>", normal_style))
        else:
            story.append(Paragraph("❌ <b>No se encontraron las palabras clave en ningún archivo</b>", normal_style))
        
        story.append(Spacer(1, 20))
    
    # Detalle por archivo
    story.append(Paragraph("📄 DETALLE POR ARCHIVO", subtitle_style))
    
    for i, result in enumerate(results, 1):
        file_header = f"📁 {i}. {result.filename}"
        story.append(Paragraph(file_header, ParagraphStyle(
            'FileHeader',
            parent=subtitle_style,
            fontSize=14,
            textColor=HexColor('#1e3c72'),
            spaceBefore=15,
            spaceAfter=10
        )))
        
        file_data = [
            ['⏱️ Duración:', f"{result.duration:.1f}s"],
            ['⚡ Tiempo de procesamiento:', f"{result.processing_time:.1f}s"],
            ['📝 Palabras transcritas:', f"{result.word_count}"],
            ['🎯 Palabras clave encontradas:', ", ".join(result.found_keywords) if result.found_keywords else "Ninguna"]
        ]
        
        file_table = Table(file_data, colWidths=[2.5*inch, 4.5*inch])
        file_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('TEXTCOLOR', (0, 0), (0, -1), HexColor('#2a5298')),
            ('BACKGROUND', (0, 0), (-1, -1), HexColor('#f8f9fa')),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#e1e5e9')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        story.append(file_table)
        
        if result.transcription:
            story.append(Spacer(1, 10))
            story.append(Paragraph("📝 <b>Transcripción:</b>", normal_style))
            
            highlighted_text = result.transcription
            for keyword in result.found_keywords:
                highlighted_text = highlighted_text.replace(
                    keyword, 
                    f'<font color="#d32f2f"><b>{keyword}</b></font>'
                )
            
            if len(highlighted_text) > 500:
                highlighted_text = highlighted_text[:500] + "..."
            
            story.append(Paragraph(highlighted_text, ParagraphStyle(
                'TranscriptionText',
                parent=normal_style,
                fontSize=10,
                leftIndent=15,
                rightIndent=15,
                borderWidth=1,
                borderColor=HexColor('#e1e8ed'),
                borderPadding=10,
                backColor=HexColor('#ffffff')
            )))
        
        story.append(Spacer(1, 15))
        
        if i % 3 == 0 and i < len(results):
            story.append(Paragraph("─" * 80, ParagraphStyle(
                'Separator',
                parent=meta_style,
                alignment=TA_CENTER,
                textColor=HexColor('#cccccc')
            )))
            story.append(Spacer(1, 10))
    
    # Footer
    # Footer con logo - al final de la página actual
    story.append(Spacer(1, 30))

    try:
        if os.path.exists('logo_wise_2.png'):
            # Crear imagen del logo para el footer
            logo_footer = Image('logo_wise_2.png', width=0.8*inch, height=0.4*inch)
            
            # Crear párrafo con el texto del footer
            footer_text = Paragraph(f"""
                <font size="8" color="#666666">
                    Reporte generado automáticamente por el Sistema VoiceWise AI<br/>
                    Instituto Universitario Rumiñahui - Departamento de Investigación<br/>
                    {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}
                </font>
            """, ParagraphStyle(
                'FooterText',
                parent=meta_style,
                fontSize=8,
                alignment=TA_LEFT,
                textColor=HexColor('#666666')
            ))
            
            # Tabla del footer: logo a la izquierda, texto a la derecha
            footer_data = [[logo_footer, footer_text]]
            
            footer_table = Table(footer_data, colWidths=[1.5*inch, 6*inch])
            footer_table.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('ALIGN', (0, 0), (0, 0), 'CENTER'),  # Logo centrado en su celda
                ('ALIGN', (1, 0), (1, 0), 'LEFT'),    # Texto alineado a la izquierda
                ('LEFTPADDING', (0, 0), (-1, -1), 10),
                ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                ('TOPPADDING', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                ('LINEABOVE', (0, 0), (-1, 0), 1, HexColor('#cccccc')),
            ]))
            
            story.append(footer_table)
        
        else:
            # Debug: mostrar que no encuentra el archivo
            st.warning("No se encontró el archivo logo_sistema.png en la raíz del proyecto")
            footer_content = f"""
            <para align="center">
                <font size="8" color="#666666">
                    Reporte generado automáticamente por el Sistema VoiceWise AI<br/>
                    Instituto Universitario Rumiñahui - Departamento de Investigación<br/>
                    {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}
                </font>
            </para>
            """
            story.append(Paragraph(footer_content, meta_style))
        
    except Exception as e:
        # Debug: mostrar el error específico
        st.error(f"Error cargando logo del sistema: {e}")
        footer_content = f"""
        <para align="center">
            <font size="8" color="#666666">
                Reporte generado automáticamente por el Sistema VoiceWise AI<br/>
                Instituto Universitario Rumiñahui - Departamento de Investigación<br/>
                {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}
            </font>
        </para>
        """
        story.append(Paragraph(footer_content, meta_style))
    
    doc.build(story)
    
    buffer.seek(0)
    pdf_content = buffer.read()
    buffer.close()
    
    return pdf_content

def create_download_zip(results: List[TranscriptionResult], keywords: List[str]) -> bytes:
    """Create ZIP file with all transcription results"""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        report = create_summary_report_md(results, keywords)
        zip_file.writestr("REPORTE_TRANSCRIPCION.md", report.encode('utf-8'))
        
        for result in results:
            if result.transcription:
                base_name = os.path.splitext(result.filename)[0]
                
                zip_file.writestr(f"transcripciones/{base_name}.txt", result.transcription.encode('utf-8'))
                
                if result.srt_path and os.path.exists(result.srt_path):
                    with open(result.srt_path, 'r', encoding='utf-8') as srt_file:
                        srt_content = srt_file.read()
                    zip_file.writestr(f"transcripciones_srt/{base_name}.srt", srt_content.encode('utf-8'))
                
                highlighted = highlight_keywords(result.transcription, keywords)
                zip_file.writestr(f"resaltados/{base_name}_resaltado.html", 
                                f"<html><body><pre>{highlighted}</pre></body></html>".encode('utf-8'))
    
    zip_buffer.seek(0)
    return zip_buffer.read()

def create_summary_report_md(results: List[TranscriptionResult], keywords: List[str]) -> str:
    """Create summary report of all transcriptions in markdown"""
    total_files = len(results)
    successful = len([r for r in results if r.transcription])
    total_duration = sum(r.duration for r in results)
    total_processing = sum(r.processing_time for r in results)
    total_words = sum(r.word_count for r in results)
    
    files_with_keywords = len([r for r in results if r.found_keywords])
    
    report = f"""# 📊 Reporte de Transcripción Masiva

## 📈 Estadísticas Generales
- **Total de archivos procesados:** {total_files}
- **Transcripciones exitosas:** {successful}
- **Duración total de audio:** {total_duration:.1f} segundos ({total_duration/60:.1f} minutos)
- **Tiempo total de procesamiento:** {total_processing:.1f} segundos
- **Total de palabras transcritas:** {total_words:,}
- **Archivos con palabras clave:** {files_with_keywords}

## 🔍 Palabras Clave Buscadas
{', '.join(keywords) if keywords else 'Ninguna'}

## 📄 Detalle por Archivo
"""
    
    for result in results:
        status = "✅" if result.transcription else "❌"
        keywords_found = ", ".join(result.found_keywords) if result.found_keywords else "Ninguna"
        
        report += f"""
### {status} {result.filename}
- **Duración:** {result.duration:.1f}s
- **Tiempo de procesamiento:** {result.processing_time:.1f}s
- **Palabras:** {result.word_count}
- **Palabras clave encontradas:** {keywords_found}
"""
    
    return report

def opciones():
    """Keyword selection interface"""
    keywords = st_tags(
        label='🏷️ Palabras clave para buscar en todas las transcripciones:',
        text='Presiona Enter o añade más términos',
        value=['emergencia', 'robo', 'drogas'],
        suggestions=['extorsión', 'robos', 'rescate', 'auxilio', 'accidente', 'violencia', 'ayuda'],
        maxtags=15,
        key="batch_keywords"
    )
    return keywords

def cleanup_temp_directory():
    """Clean up temporary directory"""
    if st.session_state.current_temp_dir and os.path.exists(st.session_state.current_temp_dir):
        try:
            shutil.rmtree(st.session_state.current_temp_dir)
            st.session_state.current_temp_dir = None
        except:
            pass

def parse_srt_file(srt_file_path: str) -> List[SRTSegment]:
    """Parse SRT file into structured segments"""
    segments = []
    
    try:
        if not os.path.exists(srt_file_path):
            return segments
            
        with open(srt_file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
    except Exception as e:
        return segments
    
    if not content:
        return segments
    
    blocks = re.split(r'\n\s*\n', content)
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            try:
                index = int(lines[0])
                time_line = lines[1]
                text = '\n'.join(lines[2:])
                
                time_match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})', time_line)
                if time_match:
                    start_time, end_time = time_match.groups()
                    segments.append(SRTSegment(index, start_time, end_time, text))
            except (ValueError, IndexError):
                continue
    
    return segments

def check_segment_for_keywords(segment: SRTSegment, keywords: List[str]) -> bool:
    """Check if segment contains any keywords"""
    if not segment or not segment.text or not keywords:
        return False
        
    text_lower = segment.text.lower()
    return any(keyword.lower().strip() in text_lower for keyword in keywords if keyword and keyword.strip())

def display_enhanced_srt_for_file(srt_file_path: str, keywords: List[str], filename: str):
    """Display SRT file with enhanced formatting and keyword highlighting"""
    try:
        if not srt_file_path or not os.path.exists(srt_file_path):
            st.warning(f"Archivo SRT no encontrado para {filename}")
            return
        
        segments = parse_srt_file(srt_file_path)
        
        if not segments:
            st.warning(f"No se encontraron segmentos en el archivo SRT de {filename}")
            return
        
        segments_with_keywords = []
        
        for segment in segments:
            try:
                if check_segment_for_keywords(segment, keywords):
                    segment.contains_keywords = True
                    segments_with_keywords.append(segment)
            except Exception as e:
                continue
        
        total_segments = len(segments)
        keyword_segments = len(segments_with_keywords)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total de segmentos", total_segments)
        with col2:
            st.metric("Con palabras clave", keyword_segments)
        with col3:
            st.metric("Porcentaje", f"{(keyword_segments/total_segments*100):.1f}%" if total_segments > 0 else "0%")
        
        if keyword_segments > 0:
            segments_to_display = segments_with_keywords
            st.markdown("#### 🎯 Segmentos con palabras clave")
        else:
            segments_to_display = segments[:10]
            st.markdown("#### 📋 Muestra de segmentos (sin palabras clave)")
            st.info(f"No se encontraron palabras clave. Mostrando los primeros 10 segmentos de {total_segments} total.")
        
        if not segments_to_display:
            st.info("No hay segmentos para mostrar.")
            return
        
        max_segments = 20
        if len(segments_to_display) > max_segments:
            st.warning(f"Mostrando los primeros {max_segments} segmentos relevantes de {len(segments_to_display)} encontrados.")
            segments_to_display = segments_to_display[:max_segments]
        
        for segment in segments_to_display:
            try:
                highlighted_text, _ = highlight_keywords_in_text(segment.text, keywords)
                
                st.markdown(f"""
                <div style="padding: 12px; margin: 10px 0; border-left: 4px solid #d32f2f; background-color: #fff3e0; border-radius: 4px;">
                    <div style="font-size: 12px; color: #666; margin-bottom: 8px; font-weight: bold;">
                        🎯 Segmento #{segment.index} | ⏱️ {segment.start_time} → {segment.end_time}
                    </div>
                    <div style="font-size: 14px; line-height: 1.5;">
                        {highlighted_text}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.write(f"**🎯 #{segment.index}** | ⏱️ {segment.start_time} → {segment.end_time}")
                st.write(segment.text)
                st.divider()
                
    except Exception as e:
        st.error(f"Error procesando marcas de tiempo para {filename}")
        st.info("Los archivos se procesaron correctamente. Puedes usar los archivos SRT descargados.")

def highlight_keywords_in_text(text: str, keywords: List[str]) -> Tuple[str, List[str]]:
    """Highlight keywords in text and return found terms"""
    if not text or not keywords:
        return text, []
        
    highlighted_text = text
    found_terms = []
    
    for keyword in keywords:
        if keyword and keyword.strip():
            try:
                pattern = re.compile(re.escape(keyword.strip()), re.IGNORECASE)
                if pattern.search(text):
                    found_terms.append(keyword.strip())
                    highlighted_text = pattern.sub(
                        lambda m: f'<mark style="background-color: #ffeb3b; color: #d32f2f; font-weight: bold;">{m.group()}</mark>',
                        highlighted_text
                    )
            except re.error:
                continue
    
    return highlighted_text, found_terms

def display_results_section():
    """Función para mostrar los resultados de manera consistente"""
    if not st.session_state.processing_results:
        return
        
    results = st.session_state.processing_results
    keywords = st.session_state.get('processing_keywords', [])
    total_time = st.session_state.get('total_processing_time', 0)
    
    # Mostrar resultados procesados
    st.markdown("### 📋 Resultados del Procesamiento")
    for j, res in enumerate(results):
        emoji = "🎯" if res.found_keywords else "📄"
        
        with st.expander(f"{emoji} {res.filename}", expanded=bool(res.found_keywords)):
            if res.found_keywords:
                st.success(f"Palabras encontradas: **{', '.join(res.found_keywords)}**")
                
                tab1, tab2 = st.tabs(["📝 Texto resaltado", "⏱️ Marcas de tiempo"])
                
                with tab1:
                    highlighted_text = highlight_keywords(res.transcription, keywords)
                    st.markdown(highlighted_text, unsafe_allow_html=True)
                
                with tab2:
                    if res.srt_path and os.path.exists(res.srt_path):
                        display_enhanced_srt_for_file(res.srt_path, keywords, res.filename)
                    else:
                        st.info("No hay archivo SRT disponible")
            else:
                st.write("❌ No se encontraron palabras clave")
                
                tab1, tab2 = st.tabs(["📝 Texto completo", "⏱️ Marcas de tiempo"])
                
                with tab1:
                    preview_text = res.transcription[:500] + "..." if len(res.transcription) > 500 else res.transcription
                    st.write(preview_text)
                
                with tab2:
                    if res.srt_path and os.path.exists(res.srt_path):
                        display_enhanced_srt_for_file(res.srt_path, keywords, res.filename)
                    else:
                        st.info("No hay archivo SRT disponible")
    
    # Mostrar resumen estadístico
    st.markdown("---")
    st.markdown("## 📊 Resumen del Procesamiento Masivo")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Archivos procesados", len(results))
    with col2:
        st.metric("Con palabras clave", len([r for r in results if r.found_keywords]))
    with col3:
        st.metric("Total palabras", f"{sum(r.word_count for r in results):,}")
    with col4:
        st.metric("Tiempo total", f"{total_time:.1f}s")
    
    # Sección de reportes y descargas
    st.markdown("### 📄 Generar Reportes y Descargas")
    
    col_report1, col_report2 = st.columns([1, 1])
    
    with col_report1:
        # Generar reporte PDF profesional
        try:
            processing_summary = {
                'total_time': total_time,
                'total_files': len(results),
                'successful_files': len([r for r in results if r.transcription])
            }
            
            pdf_content = create_pdf_report(results, keywords, processing_summary)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_filename = f"reporte_transcripcion_masiva_{timestamp}.pdf"
            
            st.download_button(
                label="📄 Descargar Reporte PDF Completo",
                data=pdf_content,
                file_name=pdf_filename,
                mime="application/pdf",
                help="Reporte con análisis completo de todas las transcripciones",
                use_container_width=True,
                type="primary"
            )
            
        except Exception as e:
            st.error(f"Error generando reporte PDF: {e}")
            #st.info("Asegúrate de tener instalado: pip install reportlab")
    
    with col_report2:
        # Descargar ZIP con todos los resultados
        try:
            zip_data = create_download_zip(results, keywords)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_filename = f"transcripciones_completas_{timestamp}.zip"
            
            st.download_button(
                label="📦 Descargar Todos los Resultados (ZIP)",
                data=zip_data,
                file_name=zip_filename,
                mime="application/zip",
                help="ZIP con transcripciones TXT, SRT, HTML resaltados y reporte MD",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Error creando ZIP: {e}")
    
    # Información sobre los reportes
    st.info("""
    📋 **Los reportes incluyen:**
    • 📊 **Reporte en PDF**: Análisis completo, estadísticas, marca institucional
    • 📦 **ZIP Completo**: Archivos TXT, SRT con marcas de tiempo, HTML resaltados
    • 📈 **Estadísticas globales**: Aparición de palabras clave por archivo
    • 🎯 **Análisis temporal**: Segmentos relevantes identificados
    • 🏛️ **Marca institucional**: Instituto Universitario Rumiñahui
    """)
    
    # Botón para limpiar resultados y empezar de nuevo
    st.markdown("---")
    if st.button("🔄 Procesar nuevos archivos", type="secondary"):
        st.session_state.processing_results = []
        st.session_state.transcription_complete = False
        st.session_state.processing_keywords = []
        st.session_state.total_processing_time = 0
        st.session_state.show_results = False
        cleanup_temp_directory()
        st.rerun()

# Interfaz principal
if __name__ == "__main__":
    st.title('🎙️ Audio Texto Extenso - Transcripción Masiva desde ZIP')
    st.markdown("*Sistema completo de transcripción masiva con análisis de palabras clave y reportes*")
    st.markdown("---")
    
    # Sidebar con información
    with st.sidebar:
        st.header("ℹ️ Información del Sistema")
        st.write("**🎯 Características principales:**")
        st.write("• Procesamiento masivo por lotes")
        st.write("• Ordenamiento automático inteligente")
        st.write("• Búsqueda de palabras clave")
        st.write("• Marcas de tiempo precisas")
        st.write("• Reportes PDF")
        st.write("• Descarga organizada de resultados")
        st.write("")
        st.write("")
        
        if st.button("🗑️ Limpiar archivos temporales"):
            cleanup_temp_directory()
            st.session_state.processing_results = []
            st.session_state.show_results = False
            st.success("Archivos limpiados")
    
    # Upload ZIP file (siempre visible)
    zip_file = st.file_uploader(
        '📦 Sube un archivo ZIP con múltiples audios',
        type=['zip'],
        help="El ZIP puede contener archivos en subdirectorios. Se ordenarán automáticamente por número."
    )
    
    # Mostrar resultados persistentes si existen
    if st.session_state.processing_results and st.session_state.show_results:
        st.success(f"✅ Resultados disponibles ({len(st.session_state.processing_results)} archivos procesados)")
        display_results_section()
    
    # Upload ZIP file
    elif zip_file is None:
        st.info('📁 Sube un archivo ZIP con múltiples audios para comenzar el procesamiento masivo')
        
        # Instrucciones detalladas
        with st.expander("📖 Instrucciones de uso del sistema masivo", expanded=True):
            st.markdown("""
            ### 🚀 Cómo usar Audio Texto Extenso (Modo Masivo):
            
            1. **📦 Prepara tu ZIP**: Coloca todos los archivos de audio en un ZIP
            2. **📁 Sube el archivo**: Usa el botón de arriba para subir tu ZIP
            3. **🔢 Verifica el orden**: Los archivos se procesarán en orden numérico automático
            4. **🏷️ Configura palabras clave**: Selecciona los términos que quieres encontrar
            5. **🚀 Procesa en lote**: Inicia el procesamiento masivo
            6. **📊 Revisa resultados**: Analiza cada archivo individualmente
            7. **📄 Descarga reportes**: Obtén PDF y ZIP completo
            
            ### 🔢 Ordenamiento inteligente automático:
            Los archivos se procesan en orden numérico, no alfabético:
            - ✅ `audio_seg_1.mp3` → `audio_seg_2.mp3` → `audio_seg_10.mp3`
            - ✅ `segment_01.wav` → `segment_02.wav` → `segment_03.wav`
            - ✅ `parte1.mp3` → `parte2.mp3` → `parte11.mp3`
            - ✅ También funciona con: `audio1`, `seg_001`, `recording_5`, etc.
            
            ### 📁 Formatos de audio soportados:
            - **WAV**: Recomendado para máxima calidad
            - **MP3**: Más común, buena compatibilidad  
            - **WAVE**: Alta fidelidad
            - **M4A, FLAC, AAC**: Formatos adicionales soportados
            
            ### 📊 Resultados del procesamiento masivo:
            - **📝 Transcripciones completas** en formato TXT
            - **⏱️ Archivos SRT** con marcas de tiempo precisas
            - **🎨 Archivos HTML** con palabras clave resaltadas
            - **📄 Reporte PDF** con análisis completo
            - **📈 Estadísticas globales** de todas las transcripciones
            - **🎯 Análisis de palabras clave** por archivo individual
            """)
    
    else:
        # Procesar ZIP file
        with st.spinner("🔍 Analizando archivo ZIP y ordenando archivos..."):
            audio_files, temp_dir = get_audio_files_from_zip(zip_file)
            st.session_state.current_temp_dir = temp_dir
        
        if not audio_files:
            st.error("❌ No se encontraron archivos de audio válidos en el ZIP")
        else:
            st.success(f"✅ {len(audio_files)} archivos de audio encontrados y ordenados automáticamente")
            
            # Mostrar lista de archivos encontrados (ahora ordenados)
            with st.expander("📋 Archivos encontrados (en orden de procesamiento)", expanded=False):
                for i, audio_file in enumerate(audio_files, 1):
                    filename = os.path.basename(audio_file)
                    file_size = os.path.getsize(audio_file) / (1024 * 1024)  # MB
                    is_valid = validate_audio_file(audio_file)
                    status = "✅" if is_valid else "❌"
                    st.write(f"**{i}.** {status} **{filename}** ({file_size:.2f} MB)")
            
            # Filtrar archivos válidos manteniendo el orden
            valid_files = [f for f in audio_files if validate_audio_file(f)]
            
            if not valid_files:
                st.error("❌ No hay archivos de audio válidos para procesar")
            else:
                # Configuración de procesamiento
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    keywords = opciones()
                    st.session_state.keywords = keywords
                    if keywords:
                        st.info(f"🔍 Buscando en todos los archivos: **{', '.join(keywords)}**")
                
                with col2:
                    st.metric("Archivos válidos", len(valid_files))
                    st.metric("Archivos inválidos", len(audio_files) - len(valid_files))
                
                # Procesamiento masivo
                if st.button('🚀 Procesar todos los archivos en lote', type="primary"):
                    if not keywords:
                        st.warning("⚠️ Agrega al menos una palabra clave para continuar")
                    else:
                        # Limpiar resultados anteriores
                        st.session_state.processing_results = []
                        st.session_state.show_results = False
                        
                        results = []
                        output_dir = tempfile.mkdtemp()
                        
                        # Progress bars
                        overall_progress = st.progress(0)
                        status_text = st.empty()
                        
                        # Placeholder para mostrar resultados en tiempo real
                        results_placeholder = st.empty()
                        
                        start_total = time.time()
                        
                        for i, audio_file in enumerate(valid_files):
                            filename = os.path.basename(audio_file)
                            
                            # Actualizar progreso
                            progress = (i + 1) / len(valid_files)
                            overall_progress.progress(progress)
                            status_text.text(f"🎵 Procesando archivo {i+1}/{len(valid_files)}: {filename}")
                            
                            # Transcribir archivo
                            transcription_result = get_transcribe_safe(audio_file)
                            
                            if transcription_result.get("error"):
                                st.error(f"❌ Error en archivo {i+1} ({filename}): {transcription_result['error']}")
                                continue
                            
                            # Procesar resultados
                            text = transcription_result.get("text", "")
                            found_keywords = find_keywords_in_text(text, keywords)
                            word_count = len(text.split()) if text else 0
                            
                            # Guardar archivos individuales
                            saved_files = save_individual_files(transcription_result, filename, output_dir)
                            
                            # Crear objeto resultado
                            result = TranscriptionResult(
                                filename=filename,
                                filepath=audio_file,
                                transcription=text,
                                duration=len(transcription_result.get("segments", [])) * 1.0,
                                processing_time=transcription_result.get("processing_time", 0),
                                found_keywords=found_keywords,
                                word_count=word_count,
                                srt_path=saved_files.get('srt')
                            )
                            
                            results.append(result)
                            
                            # Mostrar progreso con resultados acumulados
                            with results_placeholder.container():
                                st.markdown(f"### 📋 Progreso del Procesamiento ({len(results)}/{len(valid_files)})")
                                
                                for j, res in enumerate(results):
                                    emoji = "🎯" if res.found_keywords else "📄"
                                    
                                    with st.expander(f"{emoji} {res.filename}", expanded=bool(res.found_keywords)):
                                        if res.found_keywords:
                                            st.success(f"Palabras encontradas: **{', '.join(res.found_keywords)}**")
                                            
                                            tab1, tab2 = st.tabs(["📝 Texto resaltado", "⏱️ Marcas de tiempo"])
                                            
                                            with tab1:
                                                highlighted_text = highlight_keywords(res.transcription, keywords)
                                                st.markdown(highlighted_text, unsafe_allow_html=True)
                                            
                                            with tab2:
                                                if res.srt_path and os.path.exists(res.srt_path):
                                                    display_enhanced_srt_for_file(res.srt_path, keywords, res.filename)
                                                else:
                                                    st.info("No hay archivo SRT disponible")
                                        else:
                                            st.write("❌ No se encontraron palabras clave")
                                            
                                            tab1, tab2 = st.tabs(["📝 Texto completo", "⏱️ Marcas de tiempo"])
                                            
                                            with tab1:
                                                preview_text = res.transcription[:500] + "..." if len(res.transcription) > 500 else res.transcription
                                                st.write(preview_text)
                                            
                                            with tab2:
                                                if res.srt_path and os.path.exists(res.srt_path):
                                                    display_enhanced_srt_for_file(res.srt_path, keywords, res.filename)
                                                else:
                                                    st.info("No hay archivo SRT disponible")
                        
                        # Finalizar procesamiento
                        total_time = time.time() - start_total
                        
                        # Guardar en session_state para persistencia
                        st.session_state.processing_results = results
                        st.session_state.processing_keywords = keywords
                        st.session_state.total_processing_time = total_time
                        st.session_state.show_results = True
                        
                        overall_progress.progress(1.0)
                        status_text.text(f"✅ Procesamiento completado en {total_time:.2f} segundos")
                        
                        # Forzar actualización para mostrar la sección de descargas
                        st.rerun()    
