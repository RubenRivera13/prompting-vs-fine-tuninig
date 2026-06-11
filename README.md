# prompting-vs-fine-tuninig
Prompting contra Fine-Tuning

Este proyecto compara dos enfoques para la clasificación de sentimiento en reseñas en español (positivo, negativo o neutro):

Enfoque basado en prompting con LLMs
Zero-shot prompting
Few-shot prompting

Enfoque basado en fine-tuning
Ajuste de un modelo preentrenado (BETO o equivalente) usando Hugging Face en Google Colab

El objetivo es comparar ambos enfoques en términos de exactitud, matriz de confusión, costo, esfuerzo de implementación y consideraciones éticas.

Instalación

Clona el repositorio

git clone <URL_DEL_REPOSITORIO>
cd prompting-vs-fine-tuning

Instala dependencias:

pip install -r requirements.txt:

pandas
scikit-learn
transformers
datasets
evaluate
accelerate
python-dotenev
google-genia
