"""Human confirmation and correction of extracted values."""
from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel

from .geometry.utilities import rebuild_line_profile
from .models import Fact, Project, Status

EDITABLE = {
    "structure": {"rim", "type"},
    "pipe": {"dia_in", "material", "inv_up", "inv_dn"},
    "line": {"dia_in", "material", "cover"},
}


class FieldEdit(BaseModel):
    object_id: str
    field: str
    action: Literal["confirm", "set"]
    value: Optional[Union[float, str]] = None


class EditError(ValueError):
    pass


def apply_edit(prj: Project, e: FieldEdit) -> str:
    obj = prj.structure(e.object_id)
    kind = "structure"
    if obj is None:
        obj, kind = prj.pipe(e.object_id), "pipe"
    if obj is None:
        obj, kind = prj.line(e.object_id), "line"
    if obj is None:
        raise EditError(f"No object {e.object_id}")
    if e.field not in EDITABLE[kind]:
        raise EditError(f"{e.field} is not editable on a {kind}")
    fact: Fact = getattr(obj, e.field)
    if e.action == "confirm":
        if fact.value is None:
            raise EditError("Nothing to confirm; enter a value instead")
        fact.status, fact.confidence = Status.user, None
        return f"{e.object_id} {e.field} confirmed"
    if e.value is None:
        raise EditError("A value is required")
    if isinstance(fact.value, (int, float)) or e.field in ("rim", "inv_up", "inv_dn", "cover", "dia_in"):
        try:
            v = float(e.value)
        except (TypeError, ValueError):
            raise EditError("Enter a number")
        if e.field in ("rim", "inv_up", "inv_dn") and fact.value is not None and abs(v - fact.value) > 40:
            raise EditError("Elevation is more than 40 ft from the current value; check the datum")
        fact.value = round(v, 2)
    else:
        fact.value = str(e.value).strip()
    fact.status, fact.confidence = Status.user, None
    if kind == "line" and e.field == "cover":
        rebuild_line_profile(obj)
    return f"{e.object_id} {e.field} set to {fact.value}"
