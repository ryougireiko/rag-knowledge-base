"""Redis 检索结果缓存；Redis 不可用时自动降级为无缓存模式。"""
from __future__ import annotations

import json
import logging
from typing import Any

log = logging.getLogger(__name__)


class NullCache:
    """本地演示和测试使用的空缓存实现，保持与 RedisCache 相同接口。"""

    enabled = False

    def get_json(self, key: str) -> Any | None:
        return None

    def set_json(self, key: str, value: Any, ttl: int) -> bool:
        return False

    def ping(self) -> bool:
        return False

    def close(self) -> None:
        pass


class RedisCache:
    """使用 Redis 存储 JSON 检索结果，网络故障时不影响主检索流程。"""

    enabled = True

    def __init__(self, redis_url: str, client=None):
        if client is None:
            import redis

            client = redis.Redis.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=1.0,
                socket_timeout=1.0,
                health_check_interval=30,
            )
        self.client = client

    def get_json(self, key: str) -> Any | None:
        try:
            raw = self.client.get(key)
            return json.loads(raw) if raw is not None else None
        except Exception as exc:
            log.warning("Redis 读取缓存失败，回退到正常检索：%s", type(exc).__name__)
            return None

    def set_json(self, key: str, value: Any, ttl: int) -> bool:
        try:
            self.client.set(
                key,
                json.dumps(value, ensure_ascii=False, separators=(",", ":")),
                ex=max(1, int(ttl)),
            )
            return True
        except Exception as exc:
            log.warning("Redis 写入缓存失败，继续返回检索结果：%s", type(exc).__name__)
            return False

    def ping(self) -> bool:
        try:
            return bool(self.client.ping())
        except Exception:
            return False

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            log.debug("关闭 Redis 客户端时出现异常", exc_info=True)


def make_cache(settings):
    """只有配置 REDIS_URL 时才创建 Redis 客户端。"""
    if not settings.redis_url:
        return NullCache()
    return RedisCache(settings.redis_url)
