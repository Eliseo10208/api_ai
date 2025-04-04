# Configuración para Gunicorn en entorno de producción
bind = "0.0.0.0:$PORT"
workers = 1  # Para la mayoría de las aplicaciones con TensorFlow, un solo worker es suficiente
timeout = 120  # Tiempo límite extendido para la carga del modelo
preload_app = True  # Cargar la aplicación antes de crear workers