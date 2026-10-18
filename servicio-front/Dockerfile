# servicio-front · nginx oficial alpine sirviendo una página HTML/JS. Contexto de build = raíz del repositorio:
#   docker build -f servicio-front/Dockerfile -t servicio-front .
FROM nginx:1.28-alpine
COPY servicio-front/nginx-front.conf /etc/nginx/conf.d/default.conf
COPY servicio-front/index.html /usr/share/nginx/html/index.html
EXPOSE 80
HEALTHCHECK --interval=10s --timeout=3s --retries=3 CMD wget -qO- http://127.0.0.1/health || exit 1
