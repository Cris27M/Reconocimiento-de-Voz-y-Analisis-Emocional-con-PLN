# Reconocimiento-de-Voz-y-Analisis-Emocional-con-PLN
Este proyecto implementa un sistema de transcripción de audio en español utilizando un modelo preentrenado de reconocimiento de voz, seguido de técnicas de procesamiento de lenguaje natural (PLN) para el preprocesamiento del texto.

# Transcripcion y PNL con IA

## Creacion del Ambiente Virtual e instalacion de dependencias

```bash
$ python -m venv venv

#Activar
$ venv\Scripts\activate

#Instalar dependencias
$ pip install -r requirements.txt
```

## Tecnicas utilizadas

### Normalización
Al convertir todas las letras a minúsculas y quitar los acentos, la normalización ayuda a reducir inconsistencias que pueden dificultar el análisis del lenguaje. Gracias a esta transformación, términos como “acción” y “accion” se tratan como iguales, lo que mejora la precisión en tareas como la búsqueda de palabras clave o el análisis semántico.

### Tokenización
La tokenización es el proceso que permite descomponer el texto en fragmentos significativos, generalmente palabras, haciendo que sea de mayor facilidad el trabajar con las mismas por separado

### Eliminación de Stop Words
Las stop words son palabras tan comunes que suelen aportar poco valor al significado general del texto. Al eliminarlas, se reducen las distracciones y se mejora la capacidad del modelo para enfocarse en los conceptos relevantes.