"""Auto-generate read-only MCP tools from a Frappe DocType.

Public surface:
    expose_doctype(mcp, doctype, ...)  — register get/list tools on `mcp`
    build_tool(doctype, operation)     — build one Tool for one op

Frappe speaks fieldtypes ("Currency", "Link", "Check"); MCP clients speak
JSON Schema. This module bridges them so hand-written and auto-generated
tools follow the same code path.

Deliberately read-only: this fork drops create/update/delete from
upstream frappe/mcp PR #6 (frappe/mcp#6). Lyfe Hardware's MCP integration
is phase-1 read-only by design (see apps/lh/docs/erpnext-developer-handoff.md)
and the source PR's default operations included create/update, with delete
as opt-in — a foot-gun for a read-only-only deployment. Only get/list survive
here; there is no code path in this module that can write to Frappe.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from frappe_mcp.server.tools import ToolOptions, get_tool

if TYPE_CHECKING:
    from frappe_mcp.server.server import MCP


# Operations exposed by expose_doctype() — read-only only, by design.
DEFAULT_OPERATIONS: tuple[str, ...] = ("get", "list")

# Not derived from `frappe.model.numeric_fieldtypes` etc. — Frappe's groupings
# classify by DB storage, not JSON Schema shape: Check would fall in as int,
# Rating wouldn't fall in at all.
_BASE_SCHEMAS: dict[str, dict[str, Any]] = {
    "Check": {"type": "boolean"},  # Frappe stores 0/1; JSON bool is nicer for LLMs.
    "Int": {"type": "integer"},
    "Long Int": {"type": "integer"},
    "Float": {"type": "number"},
    "Currency": {"type": "number"},
    "Percent": {"type": "number"},
    "Rating": {"type": "number"},
    "Date": {"type": "string", "format": "date"},
    "Datetime": {"type": "string", "format": "date-time"},
    "Time": {"type": "string", "format": "time"},
}


def frappe_fieldtype_to_json_schema(field: dict[str, Any]) -> dict[str, Any]:
    """One entry for a tool's `inputSchema.properties[field.fieldname]`.

    Reads only `fieldtype`, `options`, `label`, `fieldname` from `field`.
    Select gets an `enum`; Link/Dynamic Link get an `x-frappe-link-doctype`
    hint carrying the target DocType.
    """
    fieldtype = field.get("fieldtype") or "Data"

    if fieldtype == "Select":
        schema = _select_schema(field)
    elif fieldtype in ("Link", "Dynamic Link"):
        schema = _link_schema(field)
    else:
        schema = dict(_BASE_SCHEMAS.get(fieldtype, {"type": "string"}))

    return _with_label(schema, field)


def _select_schema(field: dict[str, Any]) -> dict[str, Any]:
    # Frappe stores Select options as a newline-separated string.
    raw = (field.get("options") or "").strip()
    options = [line.strip() for line in raw.splitlines() if line.strip()]
    schema: dict[str, Any] = {"type": "string"}
    if options:
        schema["enum"] = options
    return schema


def _link_schema(field: dict[str, Any]) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "string"}
    target = (field.get("options") or "").strip()
    if target:
        schema["x-frappe-link-doctype"] = target
    return schema


def _with_label(schema: dict[str, Any], field: dict[str, Any]) -> dict[str, Any]:
    if "description" not in schema:
        label = (field.get("label") or field.get("fieldname") or "").strip()
        if label:
            schema["description"] = label
    return schema


def _frappe():
    # Lazy: keep this module importable without a bench, so pure helpers stay
    # unit-testable and the Frappe dependency is confined to the tool handlers.
    import frappe
    return frappe


def _snake(doctype: str) -> str:
    # Matches frappe.scrub: whitespace -> underscore, lowercased.
    return doctype.replace(" ", "_").lower()


def _require_permission(doctype: str, ptype: str, *, doc: str | None = None) -> None:
    """Raise `PermissionError` if the current user can't `ptype` on `doctype`."""
    frappe = _frappe()
    if frappe.has_permission(doctype, ptype=ptype, doc=doc, throw=False):
        return
    target = f"{doctype} '{doc}'" if doc else doctype
    raise PermissionError(f"No {ptype} permission on {target}")


def build_tool(doctype: str, operation: str, *, check_permissions: bool = True):
    """Return the MCP Tool spec for one `operation` on `doctype`.

    A spec — this doesn't call Frappe. Frappe is invoked only when a client
    (Claude, an MCP Inspector session, ...) actually calls the returned tool.

    Only "get" and "list" are supported — this fork is read-only by design.
    """
    match operation:
        case "get":
            return _get(doctype, check_permissions=check_permissions)
        case "list":
            return _list(doctype, check_permissions=check_permissions)
        case _:
            raise ValueError(
                f"Unsupported operation: {operation!r} "
                f"(this fork only supports read-only operations: 'get', 'list')"
            )


_HIDDEN_FIELD_SENTINEL = "[hidden — insufficient permission]"


def _masked_permlevel_fieldnames(doc) -> set[str]:
    """Fieldnames apply_fieldlevel_read_permissions() is about to strip.

    Computed before that call runs, since afterwards the attribute is gone
    and there's no way to tell "field is legitimately empty" apart from
    "field was masked" just by looking at the resulting dict.
    """
    frappe = _frappe()
    if frappe.session.user == "Administrator":
        return set()
    has_access_to = doc.get_permlevel_access("read")
    return {
        df.fieldname
        for df in doc.meta.fields
        if df.permlevel and df.permlevel not in has_access_to
    }


def _get(doctype: str, *, check_permissions: bool):
    def handler(name: str) -> dict[str, Any]:
        if check_permissions:
            _require_permission(doctype, "read", doc=name)
        doc = _frappe().get_doc(doctype, name)
        # frappe.get_doc() does NOT apply permlevel field-hiding on its own —
        # that only happens automatically via frappe.client.get_doc's REST path
        # or frappe.model.db_query (list views). Since this handler calls
        # get_doc() directly, permlevel-restricted fields (e.g. Lyfe Order's
        # cost_of_goods, customer_email) would otherwise leak their real value
        # to any caller with base doctype read access, regardless of their
        # permlevel clearance. Confirmed by direct testing — do not remove
        # this call.
        masked_fields = _masked_permlevel_fieldnames(doc)
        doc.apply_fieldlevel_read_permissions()
        result = doc.as_dict()
        # Document.as_dict() re-populates every declared field via self.get(),
        # so a masked field's KEY still appears with value None — that reads
        # as "this field is genuinely empty/zero" to an LLM (and to a human
        # reading Claude's summary), not "you're not allowed to see this."
        # Replace with an explicit sentinel so the distinction survives into
        # whatever Claude says back to the user.
        for fieldname in masked_fields:
            if fieldname in result:
                result[fieldname] = _HIDDEN_FIELD_SENTINEL
        return result

    handler.__name__ = f"get_{_snake(doctype)}"
    handler.__doc__ = (
        f"Get a {doctype} by name.\n"
        f"\n"
        f"Args:\n"
        f"    name: {doctype} name."
    )
    return get_tool(handler, ToolOptions(annotations={"readOnlyHint": True}))


_DEFAULT_LIST_LIMIT = 20


def _list(doctype: str, *, check_permissions: bool):
    def handler(
        filters: dict | None = None,
        fields: list[str] | None = None,
        limit: int = _DEFAULT_LIST_LIMIT,
        offset: int = 0,
        order_by: str | None = None,
    ) -> list[dict[str, Any]]:
        if check_permissions:
            _require_permission(doctype, "read")
        return _frappe().db.get_list(
            doctype,
            filters=filters or {},
            fields=fields or ["name"],
            limit=limit,
            start=offset,
            order_by=order_by,
        )

    handler.__name__ = f"list_{_snake(doctype)}"
    handler.__doc__ = (
        f"List {doctype} records with optional filters, fields, limit, and ordering.\n"
        f"\n"
        f"Args:\n"
        f"    filters: Frappe filters, e.g. {{'status': 'Open'}}.\n"
        f"    fields: Fieldnames to return; defaults to ['name'].\n"
        f"    limit: Max rows to return (default {_DEFAULT_LIST_LIMIT}).\n"
        f"    offset: Row offset for pagination.\n"
        f"    order_by: e.g. 'modified desc'."
    )

    tool = get_tool(handler, ToolOptions(annotations={"readOnlyHint": True}))
    # Semantic constraints the type system can't express.
    tool["input_schema"]["properties"]["limit"]["minimum"] = 1
    tool["input_schema"]["properties"]["offset"]["minimum"] = 0
    return tool


def expose_doctype(
    mcp: "MCP",
    doctype: str,
    *,
    operations: tuple[str, ...] = DEFAULT_OPERATIONS,
    check_permissions: bool = True,
) -> None:
    """Register one read-only MCP tool per `operations` entry for `doctype` on `mcp`.

    Example:
        mcp = frappe_mcp.MCP("my-server")
        expose_doctype(mcp, "ToDo")
        # → registers get_todo, list_todo

        expose_doctype(mcp, "ToDo", operations=("get",))  # get only

    Only "get" and "list" are accepted — this fork is read-only by design
    (see module docstring). Passing "create"/"update"/"delete" raises.
    """
    for op in operations:
        mcp.add_tool(build_tool(doctype, op, check_permissions=check_permissions))
