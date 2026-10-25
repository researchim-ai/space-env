"""Локальный HTTP-интерфейс смены и снимка группировки."""

__all__ = ["serve"]


def __getattr__(name: str):
    if name == "serve":
        from space_env.api.server import serve

        return serve
    raise AttributeError(name)
