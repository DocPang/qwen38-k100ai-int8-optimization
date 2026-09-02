#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./healthcheck.sh tp1|tp2|tp4

Optional overrides:
  BASE_URL=http://127.0.0.1:8068
  MODEL=Qwen3.8-27B-W8A8-DFlash2-TP4
  CONTAINER=q38-tp4-v30-prod
EOF
}

(($# == 1)) || { usage >&2; exit 2; }
profile=$1
case "$profile" in
  tp1)
    default_port=8042
    default_model=Qwen3.8-27B-W8A8-DFlash2-TP1
    default_container=qwen38-tp1
    ;;
  tp2)
    default_port=8062
    default_model=Qwen3.8-27B-W8A8-DFlash2-TP2
    default_container=qwen38-tp2
    ;;
  tp4)
    default_port=8068
    default_model=Qwen3.8-27B-W8A8-DFlash2-TP4
    default_container=qwen38-tp4
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

BASE_URL=${BASE_URL:-http://127.0.0.1:$default_port}
MODEL=${MODEL:-$default_model}
CONTAINER=${CONTAINER:-$default_container}

echo "== container =="
if docker inspect "$CONTAINER" >/dev/null 2>&1; then
  docker inspect -f 'status={{.State.Status}} restart={{.RestartCount}} oom={{.State.OOMKilled}} restart_policy={{.HostConfig.RestartPolicy.Name}} image={{.Image}}' "$CONTAINER"
else
  echo "WARNING: container not found: $CONTAINER"
fi

echo "== /v1/models =="
curl --max-time 10 -fsS "$BASE_URL/v1/models" | python3 -m json.tool

echo "== /v1/loads =="
if ! curl --max-time 10 -fsS "$BASE_URL/v1/loads" | python3 -m json.tool; then
  echo "WARNING: /v1/loads timed out or is temporarily unavailable (the server may be in a long prefill)." >&2
fi

echo "== chat smoke =="
python3 - "$BASE_URL" "$MODEL" <<'PY'
import json, sys, urllib.request
base, model = sys.argv[1:]
payload = {
    "model": model,
    "messages": [{"role": "user", "content": "Reply with exactly OK and nothing else."}],
    "temperature": 0,
    "max_completion_tokens": 32,
    "chat_template_kwargs": {"enable_thinking": False},
}
req = urllib.request.Request(
    base.rstrip("/") + "/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=30) as resp:
    doc = json.load(resp)
msg = doc["choices"][0]["message"]
content = msg.get("content")
print(json.dumps({"http": 200, "content": content, "finish_reason": doc["choices"][0].get("finish_reason")}, ensure_ascii=False))
if content != "OK":
    raise SystemExit("chat smoke did not return exact OK")
PY

echo "HEALTHCHECK PASS profile=$profile base=$BASE_URL model=$MODEL"
