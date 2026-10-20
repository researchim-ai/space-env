"""Разбор телеметрии через ccsdspy. Свой первичный заголовок — в core.packets."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def require_ccsdspy() -> Any:
    try:
        import ccsdspy
    except ImportError as exc:
        raise ImportError("Для разбора CCSDS установите extra ccsds: pip install 'space-env[ccsds]'") from exc
    return ccsdspy


def variable_length_definition(fields: list[tuple[str, str, int]]) -> Any:
    """
    fields: (имя, тип ccsdspy, длина в битах).
    Пример типа: 'uint'.
    """
    ccsdspy = require_ccsdspy()
    packet_fields = [
        ccsdspy.PacketField(name=name, data_type=data_type, bit_length=bit_length)
        for name, data_type, bit_length in fields
    ]
    return ccsdspy.VariableLength(packet_fields)


def parse_packet_file(path: str | Path, definition: Any) -> Any:
    """Читает файл пакетов определением ccsdspy. Результат — то, что вернёт definition.load."""
    return definition.load(str(path))
