# Se aplica SOLO al build de servicio-ia/Dockerfile (BuildKit lee "<Dockerfile>.dockerignore" antes que el .dockerignore de la raíz).
# Lista blanca: el contexto es la raíz del repositorio, así que se excluye todo y se deja pasar lo único que la imagen necesita.
# Así no entran .git, .env, datos crudos, tests, reportes ni el código de entrenamiento.
*

!servicio-ia/main.py
!servicio-ia/requirements.txt
!servicio-ia/model
!datos/preparar.py

# aunque estén dentro de lo permitido
**/__pycache__
**/*.pyc
**/*.tmp
