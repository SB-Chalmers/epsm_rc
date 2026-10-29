# SPDX-License-Identifier: MIT
# Copyright (c) 2026 EPSM Development contributors
# License: See LICENSE at the project root.
"""Package-local standard-library logging helpers; no output on import."""

from __future__ import annotations

import inspect
import logging
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from time import monotonic
from uuid import uuid4

_context = ContextVar(__name__ + ".context", default=None)


def current_context():
    """Copy identifiers for an explicit package or thread boundary."""
    return dict(_context.get() or {})


class _ContextLogger(logging.LoggerAdapter):
    """Enrich local module events without configuring handlers or levels."""

    def process(self, message, kwargs):
        """Merge active operation identifiers with explicit fields on this record."""
        extra = dict(kwargs.get("extra") or {})
        extra["epsm_context"] = {**current_context(), **extra.get("epsm_context", {})}
        kwargs["extra"] = extra
        return message, kwargs


def context_logger(name):
    """Get a module logger enriched with this package's operation identifiers."""
    return _ContextLogger(logging.getLogger(name), {})


def bind_context(**identifiers):
    """Add known identifiers to the current operation without changing inputs."""
    context = _context.get()
    if context is not None:
        context.update(identifiers)


def call_with_context(function, *args, log_context=None, **kwargs):
    """Pass context only to an explicitly supported public keyword.

    Signature inspection preserves older collaborator versions. A TypeError
    inside the collaborator is never interpreted as an unsupported keyword.
    """
    parameters = inspect.signature(function).parameters
    if "log_context" in parameters:
        kwargs.setdefault("log_context", current_context() if log_context is None else log_context)
    return function(*args, **kwargs)


def operation(event, *, level=logging.INFO, traceback=False):
    """Trace a public operation, preserving results, exceptions and signatures.

    ``log_context`` is an optional keyword-only mapping of scalar identifiers.
    Identifiers never enter physical models or scientific calculation options.
    RC operations are synchronous; callers retain their surrounding context.
    """

    def decorate(function):
        """Wrap one operation without changing its public calculation contract."""
        signature = inspect.signature(function)
        logger = logging.getLogger(function.__module__)

        def begin(args, kwargs, supplied):
            """Build an isolated operation context from explicit scalar identifiers."""
            parent = current_context() if supplied is None else dict(supplied)
            if any(
                not isinstance(key, str) or not isinstance(value, (str, int, bool, type(None)))
                for key, value in parent.items()
            ):
                raise TypeError("log_context must contain scalar identifiers")
            context = dict(parent)
            if parent.get("operation_id"):
                context["parent_operation_id"] = parent["operation_id"]
            context["operation_id"] = uuid4().hex
            try:
                bound = signature.bind(*args, **kwargs)
            except TypeError as error:
                # Name the public operation, as a direct call would.
                raise TypeError(f"{function.__qualname__}(): {error}") from None
            for key in ("building_id", "run_id", "study_id", "evaluation_id"):
                if key in bound.arguments and isinstance(bound.arguments[key], (str, int)):
                    context[key] = bound.arguments[key]
            for key in ("building", "spec", "self"):
                value = bound.arguments.get(key)
                identifier = getattr(value, "identifier", None)
                if isinstance(identifier, str):
                    context.setdefault("building_id", identifier)
            selected = {}
            for key in ("path", "directory", "output"):
                value = bound.arguments.get(key)
                if isinstance(value, (str, Path)):
                    selected[key] = str(value)
            return context, selected

        def emit(context, suffix, started, *, selected=None, error=None):
            """Emit one lifecycle event with bounded timing and error metadata."""
            details = dict(selected or {})
            if started is not None:
                details["duration_s"] = monotonic() - started
            if error is not None:
                details.update(
                    code=getattr(error, "code", event + ".failed"), error_type=type(error).__name__
                )
            logger.log(
                logging.ERROR if error is not None else level,
                "%s %s",
                event.replace(".", " "),
                suffix,
                extra={
                    "epsm_event": event + "." + suffix,
                    "epsm_context": dict(context),
                    "epsm_details": details,
                },
                exc_info=traceback and error is not None,
            )

        def summary(result):
            """Describe bounded output identities and counts without storing records."""
            details = {}
            if isinstance(result, Path):
                details["path"] = str(result)
            for name in ("run_id",):
                value = vars(result).get(name) if hasattr(result, "__dict__") else None
                if isinstance(value, (str, int)):
                    details[name] = value
            if event == "energy.rc" and hasattr(result, "__dict__"):
                axis = vars(result).get("time_axis")
                if axis is not None:
                    details["hours"] = len(axis)
            if isinstance(result, (list, tuple)):
                details["item_count"] = len(result)
            return details

        @wraps(function)
        def wrapped(*args, log_context=None, **kwargs):
            """Execute a synchronous operation and always restore the caller context."""
            context, selected = begin(args, kwargs, log_context)
            token = _context.set(context)
            started = monotonic()
            emit(context, "started", None, selected=selected)
            try:
                result = function(*args, **kwargs)
            except Exception as error:
                emit(context, "failed", started, selected=selected, error=error)
                raise
            except BaseException:
                emit(context, "cancelled", started, selected=selected)
                raise
            else:
                selected.update(summary(result))
                emit(context, "completed", started, selected=selected)
                return result
            finally:
                _context.reset(token)

        result = wrapped
        parameters = list(signature.parameters.values())
        index = next(
            (i for i, p in enumerate(parameters) if p.kind == p.VAR_KEYWORD), len(parameters)
        )
        parameters.insert(
            index, inspect.Parameter("log_context", inspect.Parameter.KEYWORD_ONLY, default=None)
        )
        result.__signature__ = signature.replace(parameters=parameters)
        return result

    return decorate
