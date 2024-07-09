import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='whisper')

import whisper
import sounddevice as sd
import wavio
import os
import json

# Cargar el modelo de Whisper
model = whisper.load_model("base")

# Cargar el diccionario médico desde el archivo JSON
def cargar_diccionario_medico(ruta_archivo):
    with open(ruta_archivo, 'r', encoding='utf-8') as archivo:
        diccionario_medico = json.load(archivo)
    return diccionario_medico

diccionario_medico = cargar_diccionario_medico('diccionario_medico.json')

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

# Función principal que procesa la transcripción
def procesar_transcripcion(ruta_audio):
    # Transcribir el audio
    texto_transcrito = transcribir_audio_medico(ruta_audio)
    print("Texto transcrito inicial:")
    print(texto_transcrito)
    
    return texto_transcrito

# Función para guardar el texto en un archivo .txt
def guardar_texto_en_archivo(texto, nombre_archivo):
    with open(nombre_archivo, 'w', encoding='utf-8') as archivo:
        archivo.write(texto)
    print(f"Texto guardado en {nombre_archivo}")

# Grabar audio desde el micrófono
def grabar_audio(duracion, frecuencia_muestreo=44100):
    print("Por favor haga su dictado durante 10 segundos...")
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
