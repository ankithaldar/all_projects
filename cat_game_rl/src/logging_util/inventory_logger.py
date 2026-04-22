from __future__ import annotations

import gzip
import json
import logging
import os
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional


class GzipRotatingFileHandler(RotatingFileHandler):
    def rotation_filename(self, default_name: str) -> str:
        return default_name + ".gz"

    def rotate(self, source: str, dest: str) -> None:
        with open(source, "rb") as f_in:
            with gzip.open(dest, "wb") as f_out:
                f_out.writelines(f_in)
        os.remove(source)


class InventoryLogger:
    def __init__(
        self,
        log_dir: str = "output/logs",
        max_bytes: int = 10_000_000,
        backup_count: int = 5,
        name: str = "crafting_rl",
    ):
        os.makedirs(log_dir, exist_ok=True)
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False

        if not self._logger.handlers:
            log_path = os.path.join(log_dir, f"{name}.log")
            handler = GzipRotatingFileHandler(
                log_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
            )
            fmt = logging.Formatter(
                fmt="%(asctime)s.%(msecs)03d | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
            handler.setFormatter(fmt)
            self._logger.addHandler(handler)

    def _ts(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")

    def _serialize(self, data: Dict[str, Any]) -> str:
        def default(obj: Any) -> Any:
            if hasattr(obj, "tolist"):
                return obj.tolist()
            if hasattr(obj, "name"):
                return obj.name
            return str(obj)
        return json.dumps(data, default=default, separators=(",", ":"))

    def log_decision(
        self, tick: int, action: Dict, mask_info: Optional[Dict] = None
    ) -> None:
        msg = self._serialize({
            "event": "decision",
            "tick": tick,
            "action": action,
            "mask_info": mask_info,
        })
        self._logger.info(msg)

    def log_transition(
        self, tick: int, obs_before: Dict, obs_after: Dict
    ) -> None:
        msg = self._serialize({
            "event": "transition",
            "tick": tick,
            "stash_before": obs_before.get("stash"),
            "stash_after": obs_after.get("stash"),
            "coins_before": obs_before.get("coins"),
            "coins_after": obs_after.get("coins"),
        })
        self._logger.debug(msg)

    def log_reward(
        self, tick: int, reward: float, components: Dict[str, float]
    ) -> None:
        msg = self._serialize({
            "event": "reward",
            "tick": tick,
            "reward": reward,
            "components": components,
        })
        self._logger.info(msg)

    def log_warning(self, tick: int, message: str) -> None:
        msg = self._serialize({
            "event": "warning",
            "tick": tick,
            "message": message,
        })
        self._logger.warning(msg)

    def log_slot_event(
        self,
        tick: int,
        event_type: str,
        item_name: str,
        details: Optional[Dict] = None,
    ) -> None:
        msg = self._serialize({
            "event": "slot",
            "tick": tick,
            "type": event_type,
            "item": item_name,
            "details": details or {},
        })
        self._logger.info(msg)

    def log_ga_event(
        self, generation: int, event_type: str, details: Dict
    ) -> None:
        msg = self._serialize({
            "event": "ga",
            "generation": generation,
            "type": event_type,
            "details": details,
        })
        self._logger.info(msg)

    def log_frame_skip(self, tick: int, skip_count: int) -> None:
        msg = self._serialize({
            "event": "frame_skip",
            "tick": tick,
            "skip_count": skip_count,
        })
        self._logger.debug(msg)
