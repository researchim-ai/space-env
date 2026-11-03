"""Owned copies of scalar container trees with standard deepcopy fallback.

Exact built-in dict/list nodes use a memoized fast path. Other types, including
subclasses, tuples, NumPy arrays and extension objects retain Python deepcopy
semantics. No JSON conversion, history sharing or loss of repeated references.
"""

from __future__ import annotations

import copy

_ATOMIC = frozenset({type(None), bool, int, float, complex, str, bytes})
# Python versions differ on whether preseeded memo entries override atoms.
_ATOMIC_MEMO = copy.deepcopy(None, {id(None): True}) is True


def owned_copy(value, memo=None):
    kind = type(value)
    if kind in _ATOMIC:
        return (
            memo.get(id(value), value) if _ATOMIC_MEMO and memo is not None else value
        )
    if memo is None:
        memo = {}
    identity = id(value)
    if identity in memo:
        return memo[identity]
    if kind is dict:
        result = {}
        memo[identity] = result
        for key, item in value.items():
            # Match deepcopy's value-before-key evaluation order for hooks.
            item = (
                (memo.get(id(item), item) if _ATOMIC_MEMO else item)
                if type(item) in _ATOMIC
                else owned_copy(item, memo)
            )
            key = (
                (memo.get(id(key), key) if _ATOMIC_MEMO else key)
                if type(key) in _ATOMIC
                else copy.deepcopy(key, memo)
            )
            result[key] = item
        memo[identity] = result
        memo.setdefault(id(memo), []).append(value)
        return result
    if kind is list:
        result = []
        memo[identity] = result
        for item in value:
            result.append(
                (memo.get(id(item), item) if _ATOMIC_MEMO else item)
                if type(item) in _ATOMIC
                else owned_copy(item, memo)
            )
        memo[identity] = result
        memo.setdefault(id(memo), []).append(value)
        return result
    return copy.deepcopy(value, memo)
