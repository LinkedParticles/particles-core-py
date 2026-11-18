# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""Prompt-free macOS generic-password access for scheduled credentials.

No call enumerates credentials or changes Keychain search/default settings.
The explicit enrolment API may permit UI; background reads never do.
"""

from __future__ import annotations

import ctypes
import sys
import threading
from typing import Any

_LOCK = threading.RLock()


class KeychainError(ValueError):
    """A credential is unavailable; the exception contains no secret value."""

    def __init__(self, status: int, operation: str = "read") -> None:
        self.status = status
        self.reason = "locked/denied" if status in {-25293, -25308} else "unavailable"
        super().__init__(f"Keychain {operation}: {self.reason} (OSStatus {status})")


class _API:
    security: ctypes.CDLL
    cf: ctypes.CDLL

    def __init__(self) -> None:
        if sys.platform != "darwin":
            raise KeychainError(-4)
        self.security = ctypes.CDLL("/System/Library/Frameworks/Security.framework/Security")
        self.cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
        p = ctypes.c_void_p
        out = ctypes.POINTER(p)
        signatures: dict[str, list[Any]] = {
            "SecKeychainGetUserInteractionAllowed": [ctypes.POINTER(ctypes.c_ubyte)],
            "SecKeychainSetUserInteractionAllowed": [ctypes.c_ubyte],
            "SecKeychainCopyDefault": [out],
            "SecKeychainFindGenericPassword": [
                p,
                ctypes.c_uint32,
                ctypes.c_char_p,
                ctypes.c_uint32,
                ctypes.c_char_p,
                ctypes.POINTER(ctypes.c_uint32),
                out,
                out,
            ],
            "SecKeychainAddGenericPassword": [
                p,
                ctypes.c_uint32,
                ctypes.c_char_p,
                ctypes.c_uint32,
                ctypes.c_char_p,
                ctypes.c_uint32,
                p,
                out,
            ],
            "SecKeychainItemModifyAttributesAndData": [p, p, ctypes.c_uint32, p],
            "SecKeychainItemFreeContent": [p, p],
        }
        for name, arguments in signatures.items():
            function = getattr(self.security, name)
            function.argtypes = arguments
            function.restype = ctypes.c_int32
        self.cf.CFRelease.argtypes = [p]
        self.cf.CFRelease.restype = None

    def call(self, name: str, *args: Any) -> int:
        return int(getattr(self.security, name)(*args))

    def check(self, name: str, *args: Any) -> None:
        status = self.call(name, *args)
        if status:
            raise KeychainError(status)


def _access(service: str, account: str, value: str | None = None) -> str | None:
    # Security.framework's interaction setting is process-wide. Serialize all
    # SDK operations and restore the caller's setting even on a failed lookup.
    with _LOCK:
        api = _API()
        previous = ctypes.c_ubyte()
        api.check("SecKeychainGetUserInteractionAllowed", ctypes.byref(previous))
        api.check("SecKeychainSetUserInteractionAllowed", value is not None)
        chain, item, data = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
        length = ctypes.c_uint32()
        service_bytes, account_bytes = service.encode(), account.encode()
        try:
            api.check("SecKeychainCopyDefault", ctypes.byref(chain))
            status = api.call(
                "SecKeychainFindGenericPassword",
                chain,
                len(service_bytes),
                service_bytes,
                len(account_bytes),
                account_bytes,
                ctypes.byref(length) if value is None else None,
                ctypes.byref(data) if value is None else None,
                ctypes.byref(item),
            )
            if value is None:
                if status == -25300:
                    return None
                if status:
                    raise KeychainError(status)
                return ctypes.string_at(data, length.value).decode("utf-8")
            raw = value.encode()
            if not raw:
                raise ValueError("Cannot enrol an empty credential")
            if status == -25300:
                api.check(
                    "SecKeychainAddGenericPassword",
                    chain,
                    len(service_bytes),
                    service_bytes,
                    len(account_bytes),
                    account_bytes,
                    len(raw),
                    raw,
                    ctypes.byref(item),
                )
            elif status:
                raise KeychainError(status, "enrol")
            else:
                api.check("SecKeychainItemModifyAttributesAndData", item, None, len(raw), raw)
            return None
        finally:
            if data:
                api.call("SecKeychainItemFreeContent", None, data)
            for ref in (item, chain):
                if ref:
                    api.cf.CFRelease(ref)
            api.call("SecKeychainSetUserInteractionAllowed", previous.value)


def read_generic_password(service: str, account: str) -> str | None:
    """Read exactly one item without UI; missing returns None, denial raises."""
    return _access(service, account)


def enrol_generic_password(service: str, account: str, value: str) -> None:
    """Explicit interactive enrolment; the value never crosses a process argument."""
    _access(service, account, value)
