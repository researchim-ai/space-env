"""
Минимальный первичный заголовок CCSDS Space Packet (6 байт).
Разбор файлов телеметрии через ccsdspy — в space_env.integrations.ccsds_backend.
"""
from __future__ import annotations


def encode_space_packet(apid: int, seq_count: int, payload: bytes, *, packet_type: int = 0) -> bytes:
    """
    Собирает пакет: версия 0, тип TM/TC, без вторичного заголовка, seq_flags=0b11 (unsegmented).
    Поле длины по стандарту — len(payload) - 1.
    """
    if not 0 <= apid <= 0x7FF:
        raise ValueError("APID должен быть в диапазоне 0..2047")
    if not 0 <= seq_count <= 0x3FFF:
        raise ValueError("счётчик последовательности должен быть в диапазоне 0..16383")
    if len(payload) < 1 or len(payload) > 65536:
        raise ValueError("поле данных пакета: от 1 до 65536 байт")
    version = 0
    sec_hdr = 0
    seq_flags = 0b11
    byte0 = ((version & 0x7) << 5) | ((packet_type & 0x1) << 4) | ((sec_hdr & 0x1) << 3) | ((apid >> 8) & 0x7)
    byte1 = apid & 0xFF
    byte2 = ((seq_flags & 0x3) << 6) | ((seq_count >> 8) & 0x3F)
    byte3 = seq_count & 0xFF
    length = len(payload) - 1
    byte4 = (length >> 8) & 0xFF
    byte5 = length & 0xFF
    return bytes([byte0, byte1, byte2, byte3, byte4, byte5]) + payload


def decode_space_packet_header(packet: bytes) -> dict[str, int]:
    """Читает первичный заголовок. Поле data_length — как в стандарте (N-1)."""
    if len(packet) < 6:
        raise ValueError("пакет короче первичного заголовка")
    version = (packet[0] >> 5) & 0x7
    packet_type = (packet[0] >> 4) & 0x1
    sec_hdr = (packet[0] >> 3) & 0x1
    apid = ((packet[0] & 0x7) << 8) | packet[1]
    seq_flags = (packet[2] >> 6) & 0x3
    seq_count = ((packet[2] & 0x3F) << 8) | packet[3]
    data_length = (packet[4] << 8) | packet[5]
    return {
        "version": version,
        "packet_type": packet_type,
        "secondary_header": sec_hdr,
        "apid": apid,
        "sequence_flags": seq_flags,
        "sequence_count": seq_count,
        "data_length": data_length,
    }
