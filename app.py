import os
import re
import nltk
import spacy
from unidecode import unidecode
from pydub import AudioSegment
from huggingsound import SpeechRecognitionModel

def audio_to_text(audio_paths):
    # Cargamos el modelo para español
    model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-spanish", device="cuda")

    transcriptions = model.transcribe(audio_paths)
    return transcriptions

# Tokenización
def tokenize_text(text):
    doc = nlp(text)
    return [token.text for token in doc]

def split_audio():
    path = "audio.mp3"
    os.makedirs("chunks", exist_ok=True)
    audio = AudioSegment.from_file(path)
    chunk_length_ms = 5 * 60 * 1000
    total_ms = len(audio)
    splits = []

    for i, start in enumerate(range(0, total_ms, chunk_length_ms)):
        chunk = audio[start:start + chunk_length_ms]
        chunk_path = os.path.join("chunks", f"chunk_{i:02d}.wav")
        chunk.export(chunk_path, format="wav")
        splits.append(chunk_path)

    return splits

# Eliminación de stop words
def delete_stopwords(tokens):
    return [t for t in tokens if t not in spanish_stopwords and re.match(r'\w+', t)]

# Normalización , lo hacemos pasando el texto a minúsculas y eliminando los acentos
def normalize(text):
    text = text.lower()
    return unidecode(text)

if __name__ == "__main__":

    #Primero dividimos el audio
    audios = split_audio()

    # Transcribimos cada division del audio 
    transcripts = audio_to_text(audios)

    # Guardamos las transcripciones en un archivo de texto
    text = "\n".join(t['transcription'] for t in transcripts)

    with open("transcripcion.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("📄 Transcripción completa guardada en 'transcripcion.txt'") #! Para validar que si lo crea

    # Imprimimos las tecnicas utilizadas a la transcripción
    print('\n\n')
    print('='*50)
    print("\n Aplicando preprocesamiento ")
    sample = transcripts[0]['transcription'][:500]  #*Tomamos un fragmento de 500 caracteres

    nltk.download('stopwords')
    spanish_stopwords = set(nltk.corpus.stopwords.words('spanish'))
    nlp = spacy.load("es_core_news_sm")

    print("\n Texto original:")
    print(sample)

    normalize = normalize(sample)
    print("\n Texto normalizado (minúsculas y sin acentos):")
    print(normalize)

    tokens = tokenize_text(normalize)
    print("\n Texto con tokenización:")
    print(tokens[:20], "...")

    text_no_stop = delete_stopwords(tokens)
    print("\n Texto sin stop words:")
    print(text_no_stop[:20], "...")

    print("\n Fin del procesamiento.")