import inspect
from functools import partial
from typing import Any, Callable, cast

import pytest
from autogen_core._function_utils import (
    get_function_schema,
    get_typed_return_annotation,
    get_typed_signature,
)


def _fn_unannotated_required(a: int, b):  # type: ignore[no-untyped-def]
    return a


def _fn_unannotated_default(a: int, b=1) -> str:  # type: ignore[no-untyped-def]
    return str(a)


def _fn_fully_annotated(a: int, b: str = "x") -> str:
    return str(a) + b


def _fn_no_return_annotation(a: int):  # type: ignore[no-untyped-def]
    return a


class _PartialReturnModel:
    """Module-level model so a forward-ref return can be resolved."""


def _fn_forward_ref_return(a: int) -> "_PartialReturnModel":
    return _PartialReturnModel()


def test_get_typed_signature_preserves_missing_annotation() -> None:
    # The parameter is intentionally unannotated, so the checker sees it as partially unknown.
    func = cast(Callable[..., Any], _fn_unannotated_required)
    signature = get_typed_signature(func)

    assert signature.parameters["a"].annotation is int
    assert signature.parameters["b"].annotation is inspect.Signature.empty
    assert signature.return_annotation is inspect.Signature.empty


def test_get_function_schema_unannotated_parameter_raises_type_error() -> None:
    func = cast(Callable[..., Any], _fn_unannotated_required)
    with pytest.raises(TypeError, match="annotations are missing for the following parameters: 'b'"):
        get_function_schema(func, description="d")


def test_get_function_schema_unannotated_default_is_omitted() -> None:
    schema = get_function_schema(_fn_unannotated_default, description="d")

    assert schema["function"]["name"] == "_fn_unannotated_default"
    assert schema["function"]["parameters"]["required"] == ["a"]
    assert "a" in schema["function"]["parameters"]["properties"]
    assert "b" not in schema["function"]["parameters"]["properties"]


def test_get_function_schema_fully_annotated() -> None:
    schema = get_function_schema(_fn_fully_annotated, description="function f")

    assert schema["type"] == "function"
    assert schema["function"]["description"] == "function f"
    assert schema["function"]["name"] == "_fn_fully_annotated"
    assert schema["function"]["parameters"]["required"] == ["a"]
    assert set(schema["function"]["parameters"]["properties"]) == {"a", "b"}


def test_get_typed_return_annotation() -> None:
    assert get_typed_return_annotation(_fn_fully_annotated) is str
    assert get_typed_return_annotation(_fn_unannotated_default) is str
    assert get_typed_return_annotation(_fn_no_return_annotation) is None


def test_get_typed_return_annotation_partial() -> None:
    assert get_typed_return_annotation(partial(_fn_fully_annotated, 1)) is str


def test_get_typed_return_annotation_partial_without_return_annotation() -> None:
    assert get_typed_return_annotation(partial(_fn_no_return_annotation, 1)) is None


def test_get_typed_return_annotation_partial_resolves_forward_ref() -> None:
    assert get_typed_return_annotation(partial(_fn_forward_ref_return, 1)) is _PartialReturnModel
