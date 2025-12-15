docker compose --profile qwen3 --profile mineru build

docker compose --profile qwen3 up -d
docker compose --profile mineru up -d

docker compose --profile qwen3 --profile mineru down

docker compose logs