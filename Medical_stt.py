import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='whisper')

import whisper
import sounddevice as sd
import wavio
import os
from spellchecker import SpellChecker
import spacy
from nltk.corpus import wordnet
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Cargar el modelo de Whisper
model = whisper.load_model("base")

# Diccionario de abreviaturas médicas
diccionario_abreviaturas = {
    "Dx": "Diagnóstico",
    "Tx": "Tratamiento",
    "Hx": "Historia"
}

# Cargar el modelo de spaCy
nlp = spacy.load("en_core_web_sm")

# Función para transcribir audio médico
def transcribir_audio_medico(ruta_audio):
    try:
        # Verificar si el archivo de audio existe
        if not os.path.isfile(ruta_audio):
            raise FileNotFoundError(f"El archivo de audio {ruta_audio} no existe.")
        
        # Transcribir el audio
        resultado = model.transcribe(ruta_audio)
        
        # Retornar la transcripción
        return resultado["text"]
    except Exception as e:
        return str(e)

# Función para corregir la ortografía
def corregir_ortografia(texto):
    spell = SpellChecker()
    palabras = texto.split()
    palabras_corregidas = [spell.correction(p) if spell.correction(p) is not None else p for p in palabras]
    return ' '.join(palabras_corregidas)

# Función para expandir abreviaturas
def expandir_abreviaturas(texto, diccionario_abreviaturas):
    palabras = texto.split()
    palabras_expandidas = [diccionario_abreviaturas.get(p, p) for p in palabras]
    return ' '.join(palabras_expandidas)

# Función para etiquetar entidades nombradas
def etiquetar_entidades(texto):
    doc = nlp(texto)
    entidades = [(ent.text, ent.label_) for ent in doc.ents]
    return entidades

# Función para normalizar el texto
def normalizar_texto(texto):
    # Convertir a minúsculas
    texto = texto.lower()
    # Eliminar caracteres no deseados
    texto = re.sub(r'[^a-zA-Z0-9\s]', '', texto)
    return texto

# Función para analizar el sentimiento
def analizar_sentimiento(texto):
    sid = SentimentIntensityAnalyzer()
    sentimiento = sid.polarity_scores(texto)
    return sentimiento

# Función principal que combina todas las técnicas
def procesar_transcripcion(ruta_audio):
    # Transcribir el audio
    texto_transcrito = transcribir_audio_medico(ruta_audio)
    print("Texto transcrito inicial:")
    print(texto_transcrito)
    
    # Corregir la ortografía
    texto_corregido = corregir_ortografia(texto_transcrito)
    print("\nTexto después de corregir la ortografía:")
    print(texto_corregido)
    
    # Expandir abreviaturas
    texto_expandid = expandir_abreviaturas(texto_corregido, diccionario_abreviaturas)
    print("\nTexto después de expandir abreviaturas:")
    print(texto_expandid)
    
    # Etiquetar entidades nombradas
    entidades = etiquetar_entidades(texto_expandid)
    print("\nEntidades nombradas en el texto:")
    print(entidades)
    
    # Normalizar el texto
    texto_normalizado = normalizar_texto(texto_expandid)
    print("\nTexto normalizado:")
    print(texto_normalizado)
    
    # Analizar el sentimiento
    sentimiento = analizar_sentimiento(texto_normalizado)
    print("\nAnálisis de sentimiento:")
    print(sentimiento)
    
    return texto_normalizado

# Función para guardar el texto en un archivo .txt
def guardar_texto_en_archivo(texto, nombre_archivo):
    with open(nombre_archivo, 'w', encoding='utf-8') as archivo:
        archivo.write(texto)
    print(f"Texto guardado en {nombre_archivo}")

# Grabar audio desde el micrófono
def grabar_audio(duracion, frecuencia_muestreo=44100):
    print("Grabando...")
    audio = sd.rec(int(duracion * frecuencia_muestreo), samplerate=frecuencia_muestreo, channels=2)
    sd.wait()  # Esperar hasta que termine la grabación
    archivo_audio = "grabacion.wav"
    wavio.write(archivo_audio, audio, frecuencia_muestreo, sampwidth=2)
    print("Grabación completa")
    return archivo_audio

# Grabar 10 segundos de audio
archivo_audio = grabar_audio(10)

# Procesar la transcripción
try:
    texto_final = procesar_transcripcion(archivo_audio)
    print("\nTexto final procesado:")
    print(texto_final)
    
    # Guardar el texto en un archivo .txt
    nombre_archivo_txt = "transcripcion.txt"
    guardar_texto_en_archivo(texto_final, nombre_archivo_txt)
except Exception as e:
    print(f"Error al procesar la transcripción: {e}")
