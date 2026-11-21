def pending_interrupt_value(snapshot) -> dict | None:
    """取出仍在等待用户确认的 interrupt 载荷。"""
    interrupts = getattr(snapshot, "interrupts", None) or ()
    if not interrupts:
        return None
    value = getattr(interrupts[0], "value", interrupts[0])
    if isinstance(value, dict):
        return value
    return None
