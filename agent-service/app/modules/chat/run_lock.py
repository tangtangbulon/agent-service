from uuid import UUID


class ThreadRunLock:
    """记录正在生成回复的会话，同一会话同一时间只允许一个请求在执行。

    只在单个进程内有效；多进程或多实例部署时需要换成 Redis 等分布式锁。
    """

    def __init__(self) -> None:
        self._running: set[UUID] = set()

    def try_acquire(self, thread_id: UUID) -> bool:
        if thread_id in self._running:
            return False
        self._running.add(thread_id)
        return True

    def release(self, thread_id: UUID) -> None:
        self._running.discard(thread_id)


thread_run_lock = ThreadRunLock()
