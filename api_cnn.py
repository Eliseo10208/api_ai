import os
import json
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow as tf

app = Flask(__name__)

# Configuración
MODELO_DIR = 'modelo_cnn'
MODELO_PATH = os.path.join(MODELO_DIR, 'modelo.h5')
METADATA_PATH = os.path.join(MODELO_DIR, 'metadatos.json')

# Cargar el modelo
modelo = None
metadatos = None

def cargar_modelo():
    global modelo, metadatos
    try:
        # Cargar el modelo
        modelo = load_model(MODELO_PATH)
        modelo.summary()
        
        # Cargar metadatos (clases, tamaño de entrada, etc.)
        with open(METADATA_PATH, 'r') as f:
            metadatos = json.load(f)
            
        print("Modelo cargado correctamente")
        return True
    except Exception as e:
        print(f"Error al cargar el modelo: {str(e)}")
        return False

def preprocesar_imagen(img_path, target_size=(224, 224)):
    """
    Preprocesa una imagen para la entrada del modelo.
    Si los metadatos contienen información sobre el tamaño de entrada, 
    se usará en lugar del valor predeterminado.
    """
    global metadatos
    
    if metadatos and 'input_shape' in metadatos:
        target_height = metadatos['input_shape'][0]
        target_width = metadatos['input_shape'][1]
        target_size = (target_height, target_width)
    
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Aplicar el preprocesamiento según los metadatos
    if metadatos and 'preprocessing' in metadatos:
        if metadatos['preprocessing'] == 'imagenet':
            img_array = preprocess_input(img_array)
        # Aquí puedes agregar más tipos de preprocesamiento
    
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    global modelo, metadatos
    
    if modelo is None:
        return jsonify({'error': 'El modelo no está cargado'}), 500
    
    # Verificar si hay un archivo en la solicitud
    if 'imagen' not in request.files:
        return jsonify({'error': 'No se envió ninguna imagen'}), 400
    
    try:
        # Guardar imagen temporalmente
        file = request.files['imagen']
        temp_path = 'temp_img.jpg'
        file.save(temp_path)
        
        # Preprocesar imagen
        img_array = preprocesar_imagen(temp_path)
        
        # Realizar predicción
        prediccion = modelo.predict(img_array)
        
        # Procesar resultados según el tipo de modelo
        resultado = {}
        
        if metadatos and 'tipo_modelo' in metadatos:
            if metadatos['tipo_modelo'] == 'clasificacion':
                # Para modelos de clasificación
                clases = metadatos.get('clases', [f'clase_{i}' for i in range(prediccion.shape[1])])
                resultado = {
                    'prediccion': prediccion.tolist()[0],
                    'clase_predicha': clases[np.argmax(prediccion[0])],
                    'confianza': float(np.max(prediccion[0])),
                    'todas_clases': {clases[i]: float(prediccion[0][i]) for i in range(len(clases))}
                }
            elif metadatos['tipo_modelo'] == 'comparacion':
                # Para modelos de comparación/embedding
                resultado = {
                    'embedding': prediccion.tolist()[0]
                }
        else:
            # Si no hay metadatos, devolver la predicción en bruto
            resultado = {
                'prediccion': prediccion.tolist()[0]
            }
        
        # Eliminar imagen temporal
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return jsonify(resultado)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/compare', methods=['POST'])
def compare():
    global modelo
    
    if modelo is None:
        return jsonify({'error': 'El modelo no está cargado'}), 500
    
    # Verificar si hay archivos en la solicitud
    if 'imagen1' not in request.files or 'imagen2' not in request.files:
        return jsonify({'error': 'Se requieren dos imágenes para comparar'}), 400
    
    try:
        # Guardar imágenes temporalmente
        file1 = request.files['imagen1']
        file2 = request.files['imagen2']
        
        temp_path1 = 'temp_img1.jpg'
        temp_path2 = 'temp_img2.jpg'
        
        file1.save(temp_path1)
        file2.save(temp_path2)
        
        # Preprocesar imágenes
        img_array1 = preprocesar_imagen(temp_path1)
        img_array2 = preprocesar_imagen(temp_path2)
        
        # Realizar predicciones
        prediccion1 = modelo.predict(img_array1)
        prediccion2 = modelo.predict(img_array2)
        
        # Calcular similitud (distancia coseno)
        # La similitud coseno va de -1 a 1, donde 1 es muy similar
        similitud = np.dot(prediccion1[0], prediccion2[0]) / (np.linalg.norm(prediccion1[0]) * np.linalg.norm(prediccion2[0]))
        
        # Calcular distancia euclidiana
        # Menor distancia significa mayor similitud
        distancia = np.linalg.norm(prediccion1[0] - prediccion2[0])
        
        resultado = {
            'similitud_coseno': float(similitud),
            'distancia_euclidiana': float(distancia),
            'embedding1': prediccion1.tolist()[0],
            'embedding2': prediccion2.tolist()[0],
        }
        
        # Eliminar imágenes temporales
        for path in [temp_path1, temp_path2]:
            if os.path.exists(path):
                os.remove(path)
        
        return jsonify(resultado)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.before_first_request
def setup():
    # Cargar el modelo cuando se reciba la primera solicitud
    cargar_modelo()

if __name__ == '__main__':
    # Ejecutar la aplicación en modo de desarrollo local
    # Cargar el modelo antes de iniciar el servidor
    if cargar_modelo():
        # Puerto configurable a través de variable de entorno (Render lo configurará automáticamente)
        port = int(os.environ.get("PORT", 5000))
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        print("No se pudo iniciar la API porque el modelo no se cargó correctamente.")