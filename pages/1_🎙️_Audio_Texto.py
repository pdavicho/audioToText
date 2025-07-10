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
from dataclasses import dataclass
from typing import List, Set, Tuple
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


st.set_page_config(page_title='Speech To Text', page_icon=':studio_microphone:', layout="wide")

# Inicializar session state
if 'transcription_complete' not in st.session_state:
    st.session_state.transcription_complete = False
if 'srt_path' not in st.session_state:
    st.session_state.srt_path = None
if 'keywords' not in st.session_state:
    st.session_state.keywords = []
if 'transcription_result' not in st.session_state:
    st.session_state.transcription_result = {}
if 'transcription_text' not in st.session_state:
    st.session_state.transcription_text = ""
if 'found_keywords' not in st.session_state:
    st.session_state.found_keywords = set()
if 'processing_time' not in st.session_state:
    st.session_state.processing_time = 0
if 'original_filename' not in st.session_state:
    st.session_state.original_filename = ""

@st.cache_resource
def load_model():
    return whisper.load_model('base')

model = load_model()

@dataclass
class SRTSegment:
    index: int
    start_time: str
    end_time: str
    text: str
    contains_keywords: bool = False

#_______________________Código para la página de reporte ________________________
# Función para crear el reporte PDF
def create_pdf_report(
    filename: str,
    transcription_text: str,
    keywords: List[str],
    found_keywords: Set[str],
    srt_segments: List = None,
    processing_time: float = 0,
    audio_duration: str = "N/A"
) -> bytes:
    """
    Genera un reporte PDF profesional con la transcripción y análisis
    """
    
    # Crear buffer en memoria para el PDF
    buffer = io.BytesIO()
    
    # Configurar documento
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=20*mm,
        leftMargin=20*mm,
        topMargin=25*mm,
        bottomMargin=20*mm,
        title=f"Reporte de Transcripción - {filename}"
    )
    
    # Estilos personalizados
    styles = getSampleStyleSheet()
    
    # Estilo para título principal
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=HexColor('#1e3c72'),
        fontName='Helvetica-Bold'
    )
    
    # Estilo para subtítulos
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=15,
        spaceBefore=20,
        textColor=HexColor('#2a5298'),
        fontName='Helvetica-Bold'
    )
    
    # Estilo para texto normal
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=10,
        alignment=TA_JUSTIFY,
        fontName='Helvetica'
    )
    
    # Estilo para metadata
    meta_style = ParagraphStyle(
        'MetaStyle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=HexColor('#666666'),
        fontName='Helvetica'
    )
    
    # Contenido del PDF
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
    
    # Título del reporte
    story.append(Paragraph("📄 REPORTE DE TRANSCRIPCIÓN DE AUDIO", title_style))
    story.append(Spacer(1, 10))
    
    # Información del archivo
    story.append(Paragraph("📊 INFORMACIÓN DEL ARCHIVO", subtitle_style))
    
    # Tabla de metadata
    metadata_data = [
        ['📁 Nombre del archivo:', filename],
        ['📅 Fecha de procesamiento:', datetime.now().strftime("%d/%m/%Y %H:%M:%S")],
        ['⏱️ Duración del audio:', audio_duration],
        ['⚡ Tiempo de procesamiento:', f"{processing_time:.2f} segundos"],
        ['🤖 Modelo utilizado:', "OpenAI Whisper (Base)"],
        ['🔍 Palabras clave buscadas:', ", ".join(keywords) if keywords else "Ninguna"]
    ]
    
    metadata_table = Table(metadata_data, colWidths=[4.5*inch, 3*inch])
    metadata_table.setStyle(TableStyle([
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
    
    story.append(metadata_table)
    story.append(Spacer(1, 20))
    
    # Análisis de palabras clave
    if keywords:
        story.append(Paragraph("🎯 ANÁLISIS DE PALABRAS CLAVE", subtitle_style))
        
        if found_keywords:
            story.append(Paragraph(f"✅ <b>Palabras encontradas ({len(found_keywords)}):</b>", normal_style))
            found_list = "<br/>".join([f"• <font color='#2a5298'><b>{word}</b></font>" for word in found_keywords])
            story.append(Paragraph(found_list, normal_style))
        else:
            story.append(Paragraph("❌ <b>No se encontraron las palabras clave especificadas</b>", normal_style))
        
        # Palabras no encontradas
        not_found = set(keywords) - found_keywords
        if not_found:
            story.append(Spacer(1, 10))
            story.append(Paragraph(f"⚠️ <b>Palabras no encontradas ({len(not_found)}):</b>", normal_style))
            not_found_list = "<br/>".join([f"• <font color='#666666'>{word}</font>" for word in not_found])
            story.append(Paragraph(not_found_list, normal_style))
        
        story.append(Spacer(1, 20))
    
    # Estadísticas del texto
    word_count = len(transcription_text.split()) if transcription_text else 0
    char_count = len(transcription_text) if transcription_text else 0
    
    story.append(Paragraph("📈 ESTADÍSTICAS DEL TEXTO", subtitle_style))
    
    stats_data = [
        ['📝 Total de palabras:', f"{word_count:,}"],
        ['🔤 Total de caracteres:', f"{char_count:,}"],
        ['📏 Promedio palabras/minuto:', f"{word_count/max(processing_time/60, 1):.0f}" if processing_time > 0 else "N/A"]
    ]
    
    if srt_segments:
        stats_data.extend([
            ['⏱️ Total de segmentos:', f"{len(srt_segments)}"],
            ['🎯 Segmentos con palabras clave:', f"{sum(1 for seg in srt_segments if seg.contains_keywords)}"]
        ])
    
    stats_table = Table(stats_data, colWidths=[4.5*inch, 3*inch])
    stats_table.setStyle(TableStyle([
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
    
    story.append(stats_table)
    story.append(Spacer(1, 20))
    
    # Transcripción completa
    story.append(Paragraph("📝 TRANSCRIPCIÓN COMPLETA", subtitle_style))
    
    if transcription_text:
        # Resaltar palabras clave en el texto
        highlighted_text = transcription_text
        for keyword in found_keywords:
            highlighted_text = highlighted_text.replace(
                keyword, 
                f'<font color="#d32f2f"><b>{keyword}</b></font>'
            )
        
        # Dividir texto en párrafos para mejor lectura
        paragraphs = highlighted_text.split('\n')
        for para in paragraphs:
            if para.strip():
                story.append(Paragraph(para.strip(), normal_style))
                story.append(Spacer(1, 8))
    else:
        story.append(Paragraph("No se pudo obtener la transcripción.", normal_style))
    
    # Nueva página para segmentos con marcas de tiempo
    if srt_segments and any(seg.contains_keywords for seg in srt_segments):
        story.append(PageBreak())
        story.append(Paragraph("⏱️ SEGMENTOS CON PALABRAS CLAVE", subtitle_style))
        story.append(Paragraph("Los siguientes segmentos contienen las palabras clave especificadas, organizados cronológicamente:", normal_style))
        story.append(Spacer(1, 15))
        
        # Mostrar solo segmentos relevantes
        relevant_segments = [seg for seg in srt_segments if seg.contains_keywords]
        
        # Agregar estadísticas de segmentos
        stats_segments_data = [
            ['📊 Total de segmentos analizados:', f"{len(srt_segments)}"],
            ['🎯 Segmentos con palabras clave:', f"{len(relevant_segments)}"],
            ['📈 Porcentaje de relevancia:', f"{(len(relevant_segments)/len(srt_segments)*100):.1f}%"]
        ]
        
        stats_segments_table = Table(stats_segments_data, colWidths=[4.5*inch, 3*inch])
        stats_segments_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('TEXTCOLOR', (0, 0), (0, -1), HexColor('#1e3c72')),
            ('BACKGROUND', (0, 0), (-1, -1), HexColor('#f8f9fa')),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#e1e5e9')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        story.append(stats_segments_table)
        story.append(Spacer(1, 20))
        
        # Agregar nota explicativa
        story.append(Paragraph("🔍 <b>Análisis Temporal:</b> Los segmentos se muestran en orden cronológico. Las palabras clave están resaltadas en <font color='#d32f2f'><b>rojo</b></font>.", normal_style))
        story.append(Spacer(1, 15))
        
        # Limitar número de segmentos para el PDF (más que en pantalla)
        max_segments_pdf = 30
        segments_to_show = relevant_segments[:max_segments_pdf]
        
        for i, segment in enumerate(segments_to_show, 1):
            # Crear una tabla para cada segmento para mejor formato
            segment_data = [
                [f"🎯 Segmento #{segment.index}", f"⏱️ {segment.start_time} → {segment.end_time}"]
            ]
            
            segment_header_table = Table(segment_data, colWidths=[4*inch, 3.5*inch])
            segment_header_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 11),
                ('TEXTCOLOR', (0, 0), (0, -1), HexColor('#1e3c72')),
                ('TEXTCOLOR', (1, 0), (1, -1), HexColor('#2a5298')),
                ('BACKGROUND', (0, 0), (-1, -1), HexColor('#f0f4f8')),
                ('GRID', (0, 0), (-1, -1), 1, HexColor('#d1d9e0')),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('LEFTPADDING', (0, 0), (-1, -1), 8),
                ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            
            story.append(segment_header_table)
            
            # Texto del segmento con palabras clave resaltadas
            segment_text = segment.text
            for keyword in found_keywords:
                # Reemplazar de manera case-insensitive pero preservando el caso original
                import re
                pattern = re.compile(re.escape(keyword), re.IGNORECASE)
                segment_text = pattern.sub(
                    lambda m: f'<font color="#d32f2f"><b>{m.group()}</b></font>', 
                    segment_text
                )
            
            # Crear párrafo con el texto del segmento
            segment_content_style = ParagraphStyle(
                'SegmentContent',
                parent=normal_style,
                fontSize=11,
                leftIndent=15,
                rightIndent=15,
                spaceAfter=15,
                spaceBefore=5,
                borderWidth=1,
                borderColor=HexColor('#e1e8ed'),
                borderPadding=10,
                backColor=HexColor('#ffffff')
            )
            
            story.append(Paragraph(segment_text, segment_content_style))
            
            # Agregar separador visual cada 3 segmentos
            if i % 3 == 0 and i < len(segments_to_show):
                story.append(Spacer(1, 5))
                story.append(Paragraph("─" * 80, ParagraphStyle(
                    'Separator',
                    parent=meta_style,
                    alignment=TA_CENTER,
                    textColor=HexColor('#cccccc')
                )))
                story.append(Spacer(1, 5))
        
        # Nota si hay más segmentos
        if len(relevant_segments) > max_segments_pdf:
            remaining = len(relevant_segments) - max_segments_pdf
            story.append(Spacer(1, 10))
            story.append(Paragraph(f"📋 <i>Se muestran los primeros {max_segments_pdf} segmentos relevantes. Hay {remaining} segmentos adicionales con palabras clave que pueden consultarse en la aplicación web.</i>", meta_style))
        
        # Resumen de timing
        if relevant_segments:
            first_occurrence = relevant_segments[0].start_time
            last_occurrence = relevant_segments[-1].end_time
            
            story.append(Spacer(1, 20))
            timing_summary = f"""
            <para align="center">
                <b>📍 RESUMEN TEMPORAL</b><br/>
                <font size="10">Primera aparición: {first_occurrence} | Última aparición: {last_occurrence}</font><br/>
                <font size="10">Distribución: {len(relevant_segments)} momentos relevantes identificados</font>
            </para>
            """
            story.append(Paragraph(timing_summary, ParagraphStyle(
                'TimingSummary',
                parent=normal_style,
                alignment=TA_CENTER,
                fontSize=11,
                textColor=HexColor('#2a5298'),
                borderWidth=1,
                borderColor=HexColor('#2a5298'),
                borderPadding=10,
                backColor=HexColor('#f8f9ff')
            )))
    
    elif srt_segments and not any(seg.contains_keywords for seg in srt_segments):
        # Si hay segmentos pero ninguno contiene palabras clave
        story.append(PageBreak())
        story.append(Paragraph("⏱️ ANÁLISIS TEMPORAL", subtitle_style))
        story.append(Paragraph("Se analizaron los segmentos temporales del audio, pero ninguno contiene las palabras clave especificadas.", normal_style))
        
        stats_data = [
            ['📊 Total de segmentos analizados:', f"{len(srt_segments)}"],
            ['🎯 Segmentos con palabras clave:', "0"],
            ['💡 Recomendación:', "Verificar ortografía de palabras clave o usar sinónimos"]
        ]
        
        no_results_table = Table(stats_data, colWidths=[4.5*inch, 3*inch])
        no_results_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('TEXTCOLOR', (0, 0), (0, -1), HexColor('#1e3c72')),
            ('BACKGROUND', (0, 0), (-1, -1), HexColor('#fff8e1')),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#ffcc02')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        story.append(no_results_table)
    
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
    
    # Construir PDF
    doc.build(story)
    
    # Obtener contenido del buffer
    buffer.seek(0)
    pdf_content = buffer.read()
    buffer.close()
    
    return pdf_content

def upload_audio():
    file = st.file_uploader('Subir un audio', type=['.wav', '.mp3', '.wave'])
    if file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(file.read())
            return tmp_file.name, file.name

def get_transcribe(audio: str, language: str = 'es'):
    return model.transcribe(audio=audio, language=language, verbose=True)

def save_file(results, format='tsv'):
    writer = get_writer(format, './')
    writer(results, f'transcribe.{format}')
    if format == 'srt':
        return f'transcribe.{format}'

def opciones():
    keywords = st_tags(
        label='Escoger las palabras que desea analizar:',
        text='Presionar enter o añadir más',
        value=['emergencia', 'robo', 'drogas'],
        suggestions=['extorsion', 'robos', 'rescate', 'auxilio'],
        maxtags=10,
        key="opciones"
    )
    return keywords

def parse_srt_file(srt_file_path: str) -> List[SRTSegment]:
    """Parse SRT file into structured segments"""
    segments = []
    
    try:
        if not os.path.exists(srt_file_path):
            return segments
            
        with open(srt_file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
    except Exception as e:
        st.error(f"Error leyendo archivo SRT: {e}")
        return segments
    
    if not content:
        return segments
    
    # Split by double newlines to get individual subtitle blocks
    blocks = re.split(r'\n\s*\n', content)
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            try:
                index = int(lines[0])
                time_line = lines[1]
                text = '\n'.join(lines[2:])
                
                # Parse time line
                time_match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})', time_line)
                if time_match:
                    start_time, end_time = time_match.groups()
                    segments.append(SRTSegment(index, start_time, end_time, text))
            except (ValueError, IndexError):
                continue
    
    return segments

def highlight_keywords_in_text(text: str, keywords: List[str]) -> Tuple[str, Set[str]]:
    """Highlight keywords in text and return found terms"""
    if not text or not keywords:
        return text, set()
        
    highlighted_text = text
    found_terms = set()
    
    for keyword in keywords:
        if keyword and keyword.strip():  # Verificar que el keyword no esté vacío
            # Create a case-insensitive pattern that preserves original case
            try:
                pattern = re.compile(re.escape(keyword.strip()), re.IGNORECASE)
                if pattern.search(text):
                    found_terms.add(keyword.strip())
                    # Reemplazar de manera más segura
                    highlighted_text = pattern.sub(
                        lambda m: f'<mark style="background-color: #ffeb3b; color: #d32f2f; font-weight: bold;">{m.group()}</mark>',
                        highlighted_text
                    )
            except re.error:
                # Si hay error en regex, continuar con el siguiente keyword
                continue
    
    return highlighted_text, found_terms

def check_segment_for_keywords(segment: SRTSegment, keywords: List[str]) -> bool:
    """Check if segment contains any keywords"""
    if not segment or not segment.text or not keywords:
        return False
        
    text_lower = segment.text.lower()
    return any(keyword.lower().strip() in text_lower for keyword in keywords if keyword and keyword.strip())

def format_srt_segment_html(segment: SRTSegment, keywords: List[str]) -> str:
    """Format a single SRT segment as HTML"""
    try:
        highlighted_text, _ = highlight_keywords_in_text(segment.text, keywords)
        
        # Style the time stamp based on whether it contains keywords
        time_style = "color: #d32f2f; font-weight: bold;" if segment.contains_keywords else "color: #666;"
        
        return f"""
        <div style="margin: 15px 0; padding: 10px; border-left: 3px solid {'#d32f2f' if segment.contains_keywords else '#ccc'}; background-color: {'#fff3e0' if segment.contains_keywords else '#f9f9f9'};">
            <div style="font-size: 12px; {time_style} margin-bottom: 5px;">
                <strong>{segment.index}</strong> | {segment.start_time} → {segment.end_time}
            </div>
            <div style="font-size: 14px; line-height: 1.4;">
                {highlighted_text}
            </div>
        </div>
        """
    except Exception as e:
        # Si hay error formateando, devolver versión básica
        return f"""
        <div style="margin: 15px 0; padding: 10px; border-left: 3px solid #ccc; background-color: #f9f9f9;">
            <div style="font-size: 12px; color: #666; margin-bottom: 5px;">
                <strong>{segment.index}</strong> | {segment.start_time} → {segment.end_time}
            </div>
            <div style="font-size: 14px; line-height: 1.4;">
                {segment.text}
            </div>
        </div>
        """

def display_enhanced_srt(srt_file_path: str, keywords: List[str]):
    """Display SRT file with enhanced formatting and keyword highlighting"""
    try:
        if not srt_file_path or not os.path.exists(srt_file_path):
            st.error("Archivo SRT no encontrado")
            return
        
        segments = parse_srt_file(srt_file_path)
        
        if not segments:
            st.warning("No se encontraron segmentos en el archivo SRT")
            return
        
        # Check which segments contain keywords
        segments_with_keywords = []
        segments_without_keywords = []
        
        for segment in segments:
            try:
                if check_segment_for_keywords(segment, keywords):
                    segment.contains_keywords = True
                    segments_with_keywords.append(segment)
                else:
                    segments_without_keywords.append(segment)
            except Exception as e:
                # Si hay error procesando un segmento, agregarlo sin keywords
                segments_without_keywords.append(segment)
        
        # Display statistics
        total_segments = len(segments)
        keyword_segments = len(segments_with_keywords)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total de segmentos", total_segments)
        with col2:
            st.metric("Con palabras clave", keyword_segments)
        with col3:
            st.metric("Porcentaje", f"{(keyword_segments/total_segments*100):.1f}%" if total_segments > 0 else "0%")
        
        # Display options con key única
        display_option = st.radio(
            "Mostrar:",
            ["Solo segmentos con palabras clave", "Todos los segmentos", "Solo segmentos sin palabras clave"],
            horizontal=True,
            key="display_filter_option"
        )
        
        # Select segments to display
        if display_option == "Solo segmentos con palabras clave":
            segments_to_display = segments_with_keywords
        elif display_option == "Solo segmentos sin palabras clave":
            segments_to_display = segments_without_keywords
        else:
            segments_to_display = segments
        
        if not segments_to_display:
            st.info("No hay segmentos para mostrar con la selección actual.")
            return
        
        # Display segments
        st.markdown("### Transcripción con marcas de tiempo")
        
        # Limitar número de segmentos mostrados para evitar problemas de rendimiento
        max_segments = 50
        if len(segments_to_display) > max_segments:
            st.warning(f"Mostrando los primeros {max_segments} segmentos de {len(segments_to_display)} total.")
            segments_to_display = segments_to_display[:max_segments]
        
        # Display segments one by one para mejor manejo de errores
        for segment in segments_to_display:
            try:
                html_content = format_srt_segment_html(segment, keywords)
                st.markdown(html_content, unsafe_allow_html=True)
            except Exception as e:
                # Si hay error con un segmento específico, mostrar versión simple
                st.write(f"**{segment.index}** | {segment.start_time} → {segment.end_time}")
                st.write(segment.text)
                st.write("---")
                
    except Exception as e:
        st.error(f"Error procesando archivo SRT: {e}")

def highlight_text_simple(text: str, keywords: List[str]) -> Tuple[str, Set[str]]:
    """Simple highlighting for the main text display"""
    if not text or not keywords:
        return text, set()
        
    highlighted_text = text
    found_terms = set()
    
    for keyword in keywords:
        if keyword and keyword.strip() and keyword.lower() in text.lower():
            found_terms.add(keyword)
            # Use case-insensitive replacement
            try:
                highlighted_text = re.sub(
                    re.escape(keyword.strip()), 
                    f'<mark style="background-color: #ffeb3b; color: #d32f2f; font-weight: bold;">{keyword.strip()}</mark>', 
                    highlighted_text, 
                    flags=re.IGNORECASE
                )
            except re.error:
                continue
    
    return highlighted_text, found_terms

def display_results():
    """Función para mostrar los resultados guardados en session_state"""
    texto = st.session_state.transcription_text
    found_terms = st.session_state.found_keywords
    original_filename = st.session_state.original_filename
    opciones_elegidas = st.session_state.keywords
    processing_time = st.session_state.processing_time
    srt_path = st.session_state.srt_path
    
    if found_terms:
        st.success(f"🎯 Encontradas las palabras: **{', '.join(found_terms)}**")
        
        # Main text display
        st.markdown("### 📝 Texto transcrito")
        highlighted_text, _ = highlight_text_simple(texto, opciones_elegidas)
        st.markdown(highlighted_text, unsafe_allow_html=True)
        
        # Sección de reportes
        st.markdown("### 📄 Generar Reporte")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            try:
                # Parsear segmentos SRT para incluir en el reporte
                srt_segments = None
                if srt_path and os.path.exists(srt_path):
                    srt_segments = parse_srt_file(srt_path)
                    # Marcar segmentos que contienen keywords
                    for segment in srt_segments:
                        if check_segment_for_keywords(segment, opciones_elegidas):
                            segment.contains_keywords = True
                
                pdf_content = create_pdf_report(
                    filename=original_filename,
                    transcription_text=texto,
                    keywords=opciones_elegidas,
                    found_keywords=found_terms,
                    srt_segments=srt_segments,
                    processing_time=processing_time,
                    audio_duration="N/A"
                )
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                pdf_filename = f"reporte_transcripcion_{timestamp}.pdf"
                
                st.download_button(
                    label="📄 Descargar Reporte Completo (PDF)",
                    data=pdf_content,
                    file_name=pdf_filename,
                    mime="application/pdf",
                    help="Reporte profesional con transcripción, análisis de palabras clave y estadísticas",
                    use_container_width=True,
                    type="primary",
                    key="pdf_persistent"
                )
                
            except Exception as e:
                st.error(f"Error generando reporte: {e}")
                st.info("Asegúrate de tener instalado: pip install reportlab")
        
        with col2:
            st.info("""
            📋 **El reporte incluye:**
            • Información del archivo
            • Análisis de palabras clave
            • Estadísticas del texto
            • Transcripción completa
            • ⏱️ Segmentos con marcas de tiempo
            • 🎯 Solo momentos relevantes
            • Marca institucional
            """)
    else:
        st.error("❌ No se encontraron los términos especificados")
        st.markdown("### 📝 Texto transcrito completo")
        st.write(texto)
        
        # Botón de reporte PDF incluso sin palabras clave encontradas
        st.markdown("### 📄 Generar Reporte")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            try:
                # Parsear segmentos SRT para incluir en el reporte
                srt_segments = None
                if srt_path and os.path.exists(srt_path):
                    srt_segments = parse_srt_file(srt_path)
                    # Marcar segmentos que contienen keywords
                    for segment in srt_segments:
                        if check_segment_for_keywords(segment, opciones_elegidas):
                            segment.contains_keywords = True
                
                pdf_content = create_pdf_report(
                    filename=original_filename,
                    transcription_text=texto,
                    keywords=opciones_elegidas,
                    found_keywords=set(),  # Sin palabras encontradas
                    srt_segments=srt_segments,
                    processing_time=processing_time,
                    audio_duration="N/A"
                )
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                pdf_filename = f"reporte_transcripcion_{timestamp}.pdf"
                
                st.download_button(
                    label="📄 Descargar Reporte Completo (PDF)",
                    data=pdf_content,
                    file_name=pdf_filename,
                    mime="application/pdf",
                    help="Reporte profesional con transcripción completa",
                    use_container_width=True,
                    type="primary",
                    key="pdf_persistent_no_keywords"
                )
                
            except Exception as e:
                st.error(f"Error generando reporte: {e}")
                st.info("Asegúrate de tener instalado: pip install reportlab")
        
        with col2:
            st.info("""
            📋 **El reporte incluye:**
            • Información del archivo
            • Análisis de palabras clave
            • Estadísticas del texto
            • Transcripción completa
            • ⏱️ Segmentos con marcas de tiempo
            • 🎯 Solo momentos relevantes
            • Marca institucional
            """)
            
        with st.expander("💡 Sugerencias"):
            st.write("• Verifica que las palabras estén escritas correctamente")
            st.write("• Intenta con sinónimos o variaciones de las palabras")
            st.write("• El audio podría no contener los términos buscados")

if __name__ == "__main__":
    st.title('🎙️ Transcripción de Audio a Texto')
    st.markdown("---")
    
    # Mostrar resultados persistentes si existen
    if st.session_state.transcription_complete and st.session_state.transcription_text:
        st.success("✅ Transcripción disponible")
        display_results()
        
        # Mostrar análisis SRT 
        if st.session_state.srt_path:
            with st.expander("📋 Ver transcripción con marcas de tiempo", expanded=False):
                if st.session_state.keywords:
                    display_enhanced_srt(st.session_state.srt_path, st.session_state.keywords)
                else:
                    st.warning("No hay palabras clave seleccionadas para el análisis.")
        
        # Botón para procesar nuevo audio
        st.markdown("---")
        if st.button("🔄 Procesar nuevo audio", type="secondary"):
            st.session_state.transcription_complete = False
            st.session_state.transcription_text = ""
            st.session_state.found_keywords = set()
            st.session_state.processing_time = 0
            st.session_state.original_filename = ""
            st.session_state.srt_path = None
            st.session_state.keywords = []
            st.rerun()
    
    else:
        # Interfaz principal para nueva transcripción
        upload_result = upload_audio()
        
        if upload_result is not None:
            audio_transcribir, original_filename = upload_result
            st.success("✅ Audio cargado exitosamente")

            opciones_elegidas = opciones()
            
            # Guardar keywords en session state
            st.session_state.keywords = opciones_elegidas

            if opciones_elegidas:
                st.info(f"🔍 Términos seleccionados: **{', '.join(opciones_elegidas)}**")
            else:
                st.warning("⚠️ Selecciona al menos una palabra clave para analizar")

            if st.button('🚀 Ejecutar Transcripción', type="primary"):
                if not opciones_elegidas:
                    st.error("Por favor selecciona al menos una palabra clave")
                else:
                    try:
                        with st.status('Ejecutando transcripción...', expanded=True) as status:
                            start_time = time.time()
                            result = get_transcribe(audio=audio_transcribir)
                            end_time = time.time()
                            status.update(
                                label=f'✅ Transcripción completada en {end_time - start_time:.2f} segundos.', 
                                state='complete', 
                                expanded=False
                            )

                        texto = result.get('text', '')
                        
                        # Save files
                        save_file(result)
                        save_file(result, 'txt')
                        srt_path = save_file(result, 'srt')
                        
                        # Highlight keywords in main text
                        highlighted_text, found_terms = highlight_text_simple(texto, opciones_elegidas)

                        # Guardar TODO en session state para persistencia
                        st.session_state.srt_path = srt_path
                        st.session_state.transcription_complete = True
                        st.session_state.transcription_text = texto
                        st.session_state.found_keywords = found_terms
                        st.session_state.processing_time = end_time - start_time
                        st.session_state.original_filename = original_filename
                        st.session_state.transcription_result = {
                            'text': texto,
                            'filename': original_filename,
                            'processing_time': end_time - start_time,
                            'keywords': opciones_elegidas
                        }
                        
                        # Forzar recarga para mostrar la vista persistente
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error durante la transcripción: {e}")
                        st.write("Por favor, intenta de nuevo o verifica que el archivo de audio sea válido.")
                    finally:
                        # Clean up
                        try:
                            if audio_transcribir and os.path.exists(audio_transcribir):
                                os.remove(audio_transcribir)
                        except:
                            pass

        else:
            st.info('📁 Por favor, carga un archivo de audio para comenzar')
            
            # Instructions
            with st.expander("ℹ️ Instrucciones de uso"):
                st.write("""
                1. **Sube un archivo de audio** en formato WAV, MP3 o WAVE
                2. **Selecciona palabras clave** para buscar en la transcripción
                3. **Ejecuta la transcripción** y revisa los resultados
                4. **Explora la transcripción** con marcas de tiempo para ubicar contexto específico
                5. **Descarga el reporte PDF** con el análisis completo
                """)
                
            with st.expander("🔧 Características"):
                st.write("""
                • **Transcripción automática** usando modelo Whisper
                • **Búsqueda de palabras clave** con resaltado visual
                • **Marcas de tiempo precisas** para navegación fácil
                • **Filtrado inteligente** de segmentos relevantes
                • **Estadísticas de resultados** para análisis rápido
                • **Reportes PDF profesionales** con marca institucional
                """)
