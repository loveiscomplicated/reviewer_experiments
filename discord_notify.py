from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request


def send_discord_message(message: str, bot_name: str = "TEDS GNN Bot") -> bool:
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        print("Discord notify skipped: DISCORD_WEBHOOK_URL is not set.")
        return False

    payload = {
        "content": message[:1900],
        "username": bot_name[:80],
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        webhook_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            ok = response.status == 204
            if ok:
                print("Discord message sent.")
            else:
                print(f"Discord notify failed: HTTP {response.status}")
            return ok
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"Discord notify failed: HTTP {exc.code}: {body}")
    except Exception as exc:
        print(f"Discord notify failed: {exc}")
    return False


def main() -> int:
    message = sys.argv[1] if len(sys.argv) > 1 else "TEDS job update"
    bot_name = sys.argv[2] if len(sys.argv) > 2 else os.getenv("DISCORD_BOT_NAME", "TEDS GNN Bot")
    send_discord_message(message, bot_name=bot_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
