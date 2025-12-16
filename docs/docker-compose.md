# Docker Compose notes

## Set a default profile via `.env`

Docker Compose automatically reads `.env` from the project directory. To avoid passing `--profile` on every command, set:

```bash
COMPOSE_PROFILES=mineru
```

Then you can run:

```bash
docker compose up -d
```

## Check RAM usage

Proper term: **container** (a Compose **service** is the definition; it typically runs as one container, but can be scaled to multiple).

Live view (updates):

```bash
docker stats
```

One snapshot:

```bash
docker stats --no-stream
```

Only containers from this Compose project:

```bash
docker compose ps -q | xargs -r docker stats --no-stream
```

