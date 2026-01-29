"""
Rate Limiter for Gemini API
Đảm bảo không vượt quá 15 requests/phút (free tier limit)
"""
import time
import threading
from typing import Optional


class RateLimiter:
    """
    Thread-safe rate limiter để đảm bảo không vượt quá giới hạn requests/phút
    
    Usage:
        limiter = RateLimiter(requests_per_minute=15, min_interval=4.5)
        limiter.wait_if_needed()  # Gọi trước mỗi API call
    """
    
    def __init__(self, requests_per_minute: int = 15, min_interval: float = 4.5):
        """
        Args:
            requests_per_minute: Số requests tối đa mỗi phút
            min_interval: Khoảng thời gian tối thiểu giữa các requests (giây)
        """
        self.requests_per_minute = requests_per_minute
        self.min_interval = min_interval
        self.last_request_time: Optional[float] = None
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """
        Chờ nếu cần thiết để đảm bảo không vượt quá rate limit
        Thread-safe, có thể gọi từ nhiều threads
        """
        with self.lock:
            if self.last_request_time is not None:
                elapsed = time.time() - self.last_request_time
                if elapsed < self.min_interval:
                    wait_time = self.min_interval - elapsed
                    print(f"⏳ Rate limiter: waiting {wait_time:.2f}s to respect {self.requests_per_minute} RPM limit...")
                    time.sleep(wait_time)
            
            self.last_request_time = time.time()
    
    def reset(self):
        """Reset rate limiter (dùng khi cần reset manual)"""
        with self.lock:
            self.last_request_time = None


# Global rate limiter instance cho Gemini API
# Free tier: 15 requests/phút = 1 request mỗi 4 giây
# Để an toàn, dùng 4.5 giây giữa các requests
_gemini_rate_limiter = RateLimiter(requests_per_minute=15, min_interval=4.5)


def get_gemini_rate_limiter() -> RateLimiter:
    """Get global Gemini API rate limiter instance"""
    return _gemini_rate_limiter
