FROM node:18-alpine AS build
ARG VITE_LLM_CONTEXT_SIZE=10000
ENV VITE_LLM_CONTEXT_SIZE=${VITE_LLM_CONTEXT_SIZE}
WORKDIR /app
COPY package*.json ./
RUN npm install --no-audit --fund=false
COPY . ./
RUN npm run build

FROM nginx:alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
