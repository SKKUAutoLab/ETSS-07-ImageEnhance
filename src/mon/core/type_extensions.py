#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Extends basic Python data types: ``list``, ``dict``, ``tuple``, ``set``, and
``collections``.
"""

__all__ = [
    "Enum",
    "are_all_items_in_dict",
    "concat_lists",
    "generate_combinations",
    "get_module_vars",
    "intersect_dicts",
    "intersect_ordered_dicts",
    "is_any_item_in_dict",
    "is_float",
    "is_int",
    "iter_to_iter",
    "iter_to_list",
    "iter_to_tuple",
    "shuffle_dict",
    "split_list",
    "to_1list",
    "to_1tuple",
    "to_2list",
    "to_2tuple",
    "to_3list",
    "to_3tuple",
    "to_4list",
    "to_4tuple",
    "to_5list",
    "to_5tuple",
    "to_6list",
    "to_6tuple",
    "to_float",
    "to_float_list",
    "to_int",
    "to_int_list",
    "to_list",
    "to_nlist",
    "to_ntuple",
    "to_pair",
    "to_quadruple",
    "to_single",
    "to_str",
    "to_triple",
    "to_tuple",
    "unique",
    "upcast",
    "wrap_str",
]

import enum
import itertools
import random
import re
from collections import OrderedDict
from types import ModuleType
from typing import Any, Callable, Iterable

import numpy as np
import torch


# ----- Numeric -----
def to_int(x: Any) -> int | None:
    """Converts a value to an integer.

    Args:
        x: Value to convert.

    Returns:
        Converted ``int`` or ``None`` if ``x`` is ``None``.

    Raises:
        ValueError: If ``x`` cannot be converted to an integer.
    """
    if x is None:
        return None
    try:
        return int(x)
    except (ValueError, TypeError):
        raise ValueError(f"[x] must be convertible to int, got {x} ({type(x).__name__}).")


def to_float(x: Any) -> float | None:
    """Converts a value to a float.

    Args:
        x: Value to convert.

    Returns:
        Converted ``float`` or ``None`` if ``x`` is ``None``.

    Raises:
        ValueError: If ``x`` cannot be converted to a float.
    """
    if x is None:
        return None
    try:
        return float(x)
    except (ValueError, TypeError):
        raise ValueError(f"[x] must be convertible to float, got {x} ({type(x).__name__}).")


def is_int(x: Any) -> bool:
    """Checks if a value can be converted to an integer.

    Args:
        x: Value to check.

    Returns:
        ``True`` if convertible to ``int``, ``False`` otherwise.
    """
    try:
        int(x)
        return True
    except (ValueError, TypeError):
        return False


def is_float(x: Any) -> bool:
    """Checks if a value can be converted to a float.

    Args:
        x: Value to check.

    Returns:
        ``True`` if convertible to ``float``, ``False`` otherwise.
    """
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False


# ----- String -----
def to_str(x: Any, sep: str = ",") -> str:
    """Converts a value to a string, joining iterable elements with a delimiter.

    Args:
        x: Value to convert.
        sep: Delimiter for separating elements. Default is ``","``.

    Returns:
        String representation of ``x``, with elements joined by ``sep`` if iterable.
    """
    if isinstance(x, dict):
        items = [str(item) for item in x.values()]
    elif isinstance(x, (list, tuple)):
        items = [str(item) for item in x]
    else:
        return str(x) if x else ""
    
    return sep.join(items) if items else ""


def wrap_str(text: str, max_length: int = 80, indent: str = None) -> str:
    """Wrap a comma-separated string if it exceeds ``max_length``, keeping words intact.

    Args:
        text: Input string.
        max_length: Maximum line length. Default is ``80``.
        indent: Indent for continuation lines. Default is ``None``.

    Returns:
        Wrapped string with lines ≤ ``max_length``.
    """
    if len(text) <= max_length:
        return text

    words = [w.strip() for w in text.split(",")]
    lines = []
    current_line   = []
    current_length = 0
    indent_len     = len(indent or "")

    for word in words:
        # Include ", " for all but the first word in line
        word_length   = len(word)
        separator     = ", " if current_line else ""
        separator_len = len(separator)
        
        # Check if adding word exceeds max_length
        if current_length + separator_len + word_length <= max_length - indent_len:
            current_line.append(word)
            current_length += separator_len + word_length
        else:
            # Finalize current line
            if current_line:
                lines.append(", ".join(current_line))
            # Start new line
            current_line   = [word]
            current_length = word_length

    # Add final line
    if current_line:
        lines.append(", ".join(current_line))

    # Apply an indent to continuation lines
    if indent:
        return "\n".join([lines[0]] + [indent + line for line in lines[1:]])
    return "\n".join(lines)


# ----- Sequence -----
def concat_lists(x: list[list]) -> list:
    """Concatenates a list of lists into a single flattened list.

    Args:
        x: List of lists to flatten.

    Returns:
        Flattened list with all elements from nested lists.
    """
    return list(itertools.chain.from_iterable(x))


def generate_combinations(x: list | tuple) -> list:
    """Generate all combinations of a list or tuple.

    Args:
        x: Input list or tuple (e.g., ``[1, 2, 3]``).

    Returns:
        List of all combinations of elements in ``x`` (e.g.,
        ``[[1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3]]``).
    """
    x = list(x) if not isinstance(x, list) else x
    return [list(comb) for r in range(1, len(x) + 1) for comb in itertools.combinations(x, r)]


def iter_to_iter(x: Iterable, item_type: type, return_type: type = None):
    """Converts an iterable to a sequence type, casting items to a type.

    Args:
        x: Input iterable (must be ``list``, ``tuple``, or ``dict``).
        item_type: Type to cast each item to.
        return_type: Output type (``list``, ``tuple``, or ``None``). Default is ``None``.

    Returns:
        Iterable cast to ``return_type`` with items as ``item_type``.

    Raises:
        TypeError: If ``x`` is not a ``list``, ``tuple``, or ``dict``.
    """
    if not isinstance(x, (list, tuple, dict)):
        raise TypeError(f"[x] must be list, tuple, or dict, got {type(x).__name__}.")
    items = map(item_type, x)
    return (list(items) if return_type is list else
            tuple(items) if return_type is tuple else items)


def iter_to_list(x: Iterable, item_type: type) -> list:
    """Converts an iterable to a list, casting items to a specified type.

    Args:
        x: Input iterable.
        item_type: Type to cast each item to.

    Returns:
        List of items cast to ``item_type``.
    """
    return list(map(item_type, x))


def iter_to_tuple(x: Iterable, item_type: type) -> tuple:
    """Converts an iterable to a tuple, casting items to a specified type.

    Args:
        x: Input iterable.
        item_type: Type to cast each item to.

    Returns:
        Tuple of items cast to ``item_type``.
    """
    return tuple(map(item_type, x))


def split_list(x: list, n: int | list[int]) -> list[list]:
    """Splits a list into sub-lists based on a count or sizes.
    
    Args:
        x: List to split.
        n: Int for equal sub-lists or list of ints for sub-list lengths.
    
    Returns:
        List of sub-lists.
    
    Raises:
        ValueError: If ``x`` cannot be split evenly by ``n`` or sizes mismatch.
    
    Examples:
        >>> x = [1, 2, 3, 4, 5, 6]
        >>> split_list(x, n=2)          # [[1, 2, 3], [4, 5, 6]]
        >>> split_list(x, n=[1, 3, 2])  # [[1], [2, 3, 4], [5, 6]]
    """
    total_len = len(x)
    
    if isinstance(n, int):
        if total_len % n != 0:
            raise ValueError(f"[x] length [{total_len}] can't be split into "
                             f"[{n}] sub-lists.")
        sizes = [total_len // n] * n
    else:
        sizes = n
        if sum(sizes) != total_len:
            raise ValueError(f"Sum of sizes [{sum(sizes)}] must equal [x] "
                             f"length [{total_len}].")
    
    start_indices = [sum(sizes[:i]) for i in range(len(sizes))]
    return [x[start:start + size] for start, size in zip(start_indices, sizes)]


def to_list(x: Any, sep: list[str] = [",", ";", ":"]) -> list:
    """Converts a value to a list, splitting strings by delimiters if needed.

    Args:
        x: Value to convert.
        sep: List of delimiters for splitting. Default is [",", ";", ":"].

    Returns:
        List representation of ``x``.
    """
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    if isinstance(x, dict):
        return list(x.values())
    if isinstance(x, str):
        stripped = re.sub(r"^\s+|\s+$|\s", "", x)
        for delimiter in sep:
            if delimiter in stripped:
                return stripped.split(delimiter)
        return [stripped]
    return [x] if x else []


def to_int_list(x: Any, sep: list[str] = [",", ";", ":"]) -> list[int]:
    """Converts a value to a list of integers, splitting strings by delimiters.

    Args:
        x: Value to convert.
        sep: List of delimiters for splitting. Default is [",", ";", ":"].

    Returns:
        List of integers from ``x``.
    """
    return [int(item) for item in to_list(x, sep=sep)]


def to_float_list(x: Any, sep: list[str] = [",", ";", ":"]) -> list[float]:
    """Converts a value to a list of floats, splitting strings by delimiters.

    Args:
        x: Value to convert.
        sep: List of delimiters for splitting. Default is [",", ";", ":"].

    Returns:
        List of floats from ``x``.
    """
    return [float(item) for item in to_list(x, sep=sep)]


def to_nlist(n: int) -> Callable[[Any], list]:
    """Creates a function to convert an input to a list of length ``n``.

    Args:
        n: Desired list length.

    Returns:
        Function converting input to list of length ``n`` via replication or truncation.
    """
    def parse(x: Any) -> list:
        items = list(x) if isinstance(x, Iterable) and not isinstance(x, str) else [x]
        return items * n if len(items) == 1 else items[:n]
    return parse


to_1list = to_nlist(1)
to_2list = to_nlist(2)
to_3list = to_nlist(3)
to_4list = to_nlist(4)
to_5list = to_nlist(5)
to_6list = to_nlist(6)


def to_tuple(x: Any) -> tuple:
    """Converts a value to a tuple.

    Args:
        x: Value to convert.

    Returns:
        Tuple representation of ``x``.

    Raises:
        TypeError: If ``x`` is not iterable and not a single value.
    """
    if isinstance(x, list):
        return tuple(x)
    if isinstance(x, dict):
        return tuple(x.values())
    if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
        return tuple(x)
    return (x,) if x else ()


def to_ntuple(n: int) -> Callable[[Any], tuple]:
    """Creates a function to convert an input to a tuple of length ``n``.

    Args:
        n: Desired tuple length.

    Returns:
        Function converting input to tuple of length ``n`` via replication or truncation.
    """
    def parse(x: Any) -> tuple:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            items = tuple(x)
            return tuple(items * (n // len(items) + 1))[:n] if len(items) == 1 else items[:n]
        return tuple(itertools.repeat(x, n))
    return parse


to_1tuple    = to_ntuple(1)
to_2tuple    = to_ntuple(2)
to_3tuple    = to_ntuple(3)
to_4tuple    = to_ntuple(4)
to_5tuple    = to_ntuple(5)
to_6tuple    = to_ntuple(6)
to_single    = to_ntuple(1)
to_pair      = to_ntuple(2)
to_triple    = to_ntuple(3)
to_quadruple = to_ntuple(4)


def unique(x: list | tuple) -> list | tuple:
    """Returns unique items from a list or tuple, preserving input type.

    Args:
        x: Input ``list`` or ``tuple`` to deduplicate.

    Returns:
        Deduplicated ``list`` or ``tuple`` matching type of ``x``.

    Raises:
        TypeError: If ``x`` is not a ``list`` or ``tuple``.
    """
    if not isinstance(x, (list, tuple)):
        raise TypeError(f"[x] must be a list or tuple, got {type(x).__name__}.")
    return type(x)(set(x))


# ----- Collection -----
def intersect_dicts(x: dict, y: dict, exclude: list = []) -> dict:
    """Finds the intersection between two dictionaries.

    Args:
        x: First dictionary.
        y: Second dictionary.
        exclude: List of keys to exclude. Default is [].

    Returns:
        Dict with keys in both ``x`` and ``y``, excluding ``exclude``, where values
        match.
    """
    return {k: v for k, v in x.items() if k in y and k not in exclude and v == y[k]}


def intersect_ordered_dicts(x: OrderedDict, y: OrderedDict, exclude: list = []) -> OrderedDict:
    """Finds the intersection between two OrderedDict instances.

    Args:
        x: First ``OrderedDict``.
        y: Second ``OrderedDict``.
        exclude: List of keys to exclude. Default is [].

    Returns:
        ``OrderedDict`` with keys in both ``x`` and ``y``, excluding ``exclude``, where
        values match.
    """
    return OrderedDict((k, v) for k, v in x.items() if k in y and k not in exclude and v == y[k])


def shuffle_dict(x: dict) -> dict:
    """Shuffles a dictionary's keys randomly.

    Args:
        x: Dictionary to shuffle.

    Returns:
        New dictionary with keys in random order.
    """
    keys = list(x.keys())
    random.shuffle(keys)
    return dict((key, x[key]) for key in keys)


def is_any_item_in_dict(items: list, d: dict) -> bool:
    """Check if any item in the list is a key in the dictionary.
    
    Args:
        items: List of items to check (e.g., strings, ints).
        d: Dictionary to check against (keys can be any type).

    Returns:
        bool: ``True`` if at least one item is a key in the dictionary, ``False`` otherwise.
    """
    return any(item in d for item in items)


def are_all_items_in_dict(items: list, d: dict) -> bool:
    """Check if all items in the list are keys in the dictionary.
    
    Args:
        items: List of items to check (e.g., strings, ints).
        d: Dictionary to check against (keys can be any type).

    Returns:
        bool: ``True`` if all items are keys in the dictionary, ``False`` otherwise.
    """
    return all(item in d for item in items)


# ----- Enum -----
class Enum(enum.Enum):
    """Extension of Python ``enum.Enum`` with utility methods."""
    
    @classmethod
    def __init_subclass__(cls):
        """Initialize the set of values when the subclass is created."""
        cls._names         = list(cls)
        cls._values        = [member.value for member in cls]
        cls._int_to_enum   = {i: member for i, member in enumerate(cls)}
        cls._value_to_enum = {member.value: member for member in cls}
        cls._str_to_enum   = {str(member.name).lower(): member for member in cls}
        
    @classmethod
    def __contains__(cls, value: Any) -> bool:
        """Checks if a value is in the enum.

        Args:
            value: Value to check.

        Returns:
            ``True`` if value is in the enum, ``False`` otherwise.
        
        Notes:
            Usage: ``if value in EnumClass: ...``
        """
        return value in cls or value in cls._values
    
    @classmethod
    def random(cls):
        """Returns a random enum member.

        Returns:
            Random member of the enum class.
        """
        return random.choice(list(cls))
    
    @classmethod
    def random_value(cls):
        """Returns a random enum value.

        Returns:
            Value of a random enum member.
        """
        return cls.random().value
    
    @classmethod
    def names(cls) -> list:
        """Returns all enum members.

        Returns:
            List of all enum members.
        """
        return cls._names
    
    @classmethod
    def values(cls) -> list[Any]:
        """Returns all enum values.

        Returns:
            List of values from all enum members.
        """
        return cls._values
    
    @classmethod
    def int_to_enum(cls) -> dict:
        """Dynamically create a dict mapping Enum indexes to Enum members."""
        return cls._int_to_enum
    
    @classmethod
    def value_to_enum(cls) -> dict:
        """Dynamically create a dict mapping Enum values to Enum members."""
        return cls._value_to_enum
    
    @classmethod
    def str_to_enum(cls) -> dict:
        """Dynamically create a dict mapping Enum names (lowercase) to Enum members."""
        return cls._str_to_enum
    
    @classmethod
    def from_str(cls, value: str):
        """Create an Enum member from a string.
        
        Args:
            value: The string representation.
    
        Returns:
            The corresponding Enum member.
    
        Raises:
            ValueError: If the string is not a valid Enum name.
        """
        str_to_enum = cls.str_to_enum()
        value_lower = value.lower()
        if value_lower not in str_to_enum:
            raise ValueError(f"`value` must be one of {str_to_enum}, got {value_lower}.")
        return str_to_enum[value_lower]
    
    @classmethod
    def from_int(cls, value: int):
        """Create an Enum member from an index.
        
        Args:
            value: The index.
    
        Returns:
            The corresponding Enum member.
    
        Raises:
            ValueError: If the index is not a valid Enum index.
        """
        int_to_enum = cls.int_to_enum()
        if value not in int_to_enum:
            raise ValueError(f"`value` must be one of {int_to_enum}, got {value}.")
        return int_to_enum[value]
    
    @classmethod
    def from_value(cls, value: Any):
        """Create an Enum member from an arbitrary value."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls.from_str(value)
        if isinstance(value, int):
            return cls.from_int(value)
        return None
    

# ----- Module -----
def get_module_vars(module: ModuleType) -> dict:
    """Returns public variables of a module as a dictionary.

    Args:
        module: Module to inspect for public variables.

    Returns:
        Dict of public vars, excluding private, callable, or module types.
    """
    return {
        k: v for k, v in vars(module).items()
        if not (
            k.startswith(("_", "annotations"))
            or k == "__init__"
            or callable(v)
            or isinstance(v, ModuleType)
        )
    }


# ----- Parsing -----
def upcast(
    x        : torch.Tensor | np.ndarray,
    keep_type: bool = False
) -> torch.Tensor | np.ndarray:
    """Upcasts an array or tensor to a higher type to prevent overflows.

    Args:
        x: Input as ``torch.Tensor`` or ``numpy.ndarray``.
        keep_type: If ``True``, upcasts to higher int type. Default is ``False``.

    Returns:
        Upcasted ``torch.Tensor`` or ``numpy.ndarray``.
    """
    is_tensor = isinstance(x, torch.Tensor)
    
    match x.dtype:
        case torch.float16 | np.float16:
            return x.to(torch.float32) if is_tensor else x.astype(np.float32)
        case torch.float32 | np.float32:
            return x
        case torch.int8 | np.int8:
            return (
                x.to(torch.int16) if keep_type and is_tensor else
                x.to(torch.float16) if is_tensor else
                x.astype(np.float32)
            )
        case torch.int16 | np.int16:
            return (
                x.to(torch.int32) if keep_type and is_tensor else
                x.to(torch.float32) if is_tensor else
                x.astype(np.float32)
            )
        case torch.int32 | np.int32:
            return x if is_tensor else x.astype(np.float32)
        case _:
            return x
