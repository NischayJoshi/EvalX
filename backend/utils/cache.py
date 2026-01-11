"""
Redis Cache - Caching layer for expensive operations
"""
import redis
import json
import hashlib
from typing import Optional, Any
from functools import wraps
import os

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/1")

try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    redis_client.ping()
    CACHE_ENABLED = True
except:
    redis_client = None
    CACHE_ENABLED = False


class Cache:
    """Simple Redis cache wrapper"""
    
    DEFAULT_TTL = 3600
    
    EVALUATION_TTL = 86400
    USER_TTL = 300
    EVENT_TTL = 600
    LEADERBOARD_TTL = 60
    
    @staticmethod
    def _make_key(*args) -> str:
        """Create a cache key from arguments"""
        key_data = ":".join(str(arg) for arg in args)
        return f"evalx:{key_data}"
    
    @staticmethod
    def _hash_content(content: str) -> str:
        """Create hash of content for cache keys"""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    @classmethod
    def get(cls, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not CACHE_ENABLED:
            return None
        try:
            data = redis_client.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            print(f"Cache get error: {e}")
        return None
    
    @classmethod
    def set(cls, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in cache"""
        if not CACHE_ENABLED:
            return False
        try:
            ttl = ttl or cls.DEFAULT_TTL
            redis_client.setex(key, ttl, json.dumps(value, default=str))
            return True
        except Exception as e:
            print(f"Cache set error: {e}")
        return False
    
    @classmethod
    def delete(cls, key: str) -> bool:
        """Delete key from cache"""
        if not CACHE_ENABLED:
            return False
        try:
            redis_client.delete(key)
            return True
        except Exception as e:
            print(f"Cache delete error: {e}")
        return False
    
    @classmethod
    def delete_pattern(cls, pattern: str) -> int:
        """Delete all keys matching pattern"""
        if not CACHE_ENABLED:
            return 0
        try:
            keys = redis_client.keys(f"evalx:{pattern}")
            if keys:
                return redis_client.delete(*keys)
        except Exception as e:
            print(f"Cache delete pattern error: {e}")
        return 0
    
    @classmethod
    def get_repo_evaluation(cls, repo_url: str, commit_hash: str) -> Optional[dict]:
        """Get cached repo evaluation result"""
        key = cls._make_key("repo_eval", cls._hash_content(repo_url), commit_hash[:8])
        return cls.get(key)
    
    @classmethod
    def set_repo_evaluation(cls, repo_url: str, commit_hash: str, result: dict) -> bool:
        """Cache repo evaluation result"""
        key = cls._make_key("repo_eval", cls._hash_content(repo_url), commit_hash[:8])
        return cls.set(key, result, cls.EVALUATION_TTL)
    
    @classmethod
    def get_ppt_evaluation(cls, file_hash: str) -> Optional[dict]:
        """Get cached PPT evaluation result"""
        key = cls._make_key("ppt_eval", file_hash)
        return cls.get(key)
    
    @classmethod
    def set_ppt_evaluation(cls, file_hash: str, result: dict) -> bool:
        """Cache PPT evaluation result"""
        key = cls._make_key("ppt_eval", file_hash)
        return cls.set(key, result, cls.EVALUATION_TTL)
    
    @classmethod
    def get_leaderboard(cls, event_id: str) -> Optional[list]:
        """Get cached leaderboard"""
        key = cls._make_key("leaderboard", event_id)
        return cls.get(key)
    
    @classmethod
    def set_leaderboard(cls, event_id: str, leaderboard: list) -> bool:
        """Cache leaderboard"""
        key = cls._make_key("leaderboard", event_id)
        return cls.set(key, leaderboard, cls.LEADERBOARD_TTL)
    
    @classmethod
    def invalidate_leaderboard(cls, event_id: str) -> bool:
        """Invalidate leaderboard cache when scores change"""
        key = cls._make_key("leaderboard", event_id)
        return cls.delete(key)
    
    @classmethod
    def get_event(cls, event_id: str) -> Optional[dict]:
        """Get cached event data"""
        key = cls._make_key("event", event_id)
        return cls.get(key)
    
    @classmethod
    def set_event(cls, event_id: str, event_data: dict) -> bool:
        """Cache event data"""
        key = cls._make_key("event", event_id)
        return cls.set(key, event_data, cls.EVENT_TTL)
    
    @classmethod
    def invalidate_event(cls, event_id: str) -> bool:
        """Invalidate event cache"""
        key = cls._make_key("event", event_id)
        return cls.delete(key)


def cached(key_prefix: str, ttl: int = None, key_args: list = None):
    """
    Decorator to cache function results.
    
    Usage:
        @cached("user_profile", ttl=300, key_args=["user_id"])
        async def get_user_profile(user_id: str):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not CACHE_ENABLED:
                return await func(*args, **kwargs)
            
            if key_args:
                key_parts = [key_prefix]
                for arg_name in key_args:
                    if arg_name in kwargs:
                        key_parts.append(str(kwargs[arg_name]))
                cache_key = Cache._make_key(*key_parts)
            else:
                cache_key = Cache._make_key(key_prefix, *[str(a) for a in args])
            
            cached_result = Cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            result = await func(*args, **kwargs)
            
            if result is not None:
                Cache.set(cache_key, result, ttl or Cache.DEFAULT_TTL)
            
            return result
        return wrapper
    return decorator


def get_latest_commit_hash(repo_url: str) -> Optional[str]:
    """Get latest commit hash for a repo (for cache key generation)"""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "ls-remote", repo_url, "HEAD"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return result.stdout.split()[0]
    except:
        pass
    return None
