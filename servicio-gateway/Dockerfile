# servicio-gateway · nginx oficial alpine. 1.28 (estable) soporta 'resolve' en upstream, necesario para
# re-descubrir las réplicas de la IA por el DNS de Docker. Contexto de build = raíz del repositorio:
#   docker build -f servicio-gateway/Dockerfile -t servicio-gateway .
FROM nginx:1.28-alpine
COPY servicio-gateway/nginx.conf /etc/nginx/nginx.conf
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s --retries=3 CMD wget -qO- http://127.0.0.1:8080/health || exit 1
