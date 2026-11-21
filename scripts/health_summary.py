"""Resume /api/health (JSON por la entrada estándar) en una línea: instancia, arranque y memoria."""

import json
import sys

health = json.load(sys.stdin)
instance, cold = health["instance"], health["cold_start"]
print(f"instancia={instance['id']} uptime={instance['uptime_s']}s peticiones={instance['requests']} "
      f"boot->listo={cold['boot_to_engine_ready_s']}s primera_peticion={json.dumps(cold['first_request'])} "
      f"rss_mb={health['memory_mb']['rss']} carga={health['load_seconds']}s")
