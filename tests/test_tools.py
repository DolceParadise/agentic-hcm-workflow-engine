import pytest

from hcm_engine.tools import MockHRTools, ToolError, call_with_retry, definitions_as_dicts


def test_tool_schemas_are_structured():
    definitions = definitions_as_dicts()
    apply_leave = next(item for item in definitions if item["name"] == "apply_leave")
    assert apply_leave["input_schema"]["required"] == [
        "employee_id",
        "start_date",
        "end_date",
        "leave_type",
    ]
    assert apply_leave["input_schema"]["additionalProperties"] is False


def test_leave_application_updates_balance():
    tools = MockHRTools()
    result = tools.apply_leave("E001", "2026-07-06", "2026-07-08", "annual")
    assert result["working_days"] == 3
    assert tools.check_leave_balance("E001")["balances_days"]["annual"] == 15


def test_business_error_is_meaningful():
    tools = MockHRTools()
    with pytest.raises(ToolError, match="Insufficient"):
        tools.apply_leave("E001", "2026-07-01", "2026-08-31", "annual")


def test_transient_failure_is_retried():
    attempts = 0

    def flaky():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError("temporary")
        return {"status": "ok"}

    assert call_with_retry(flaky, {}, delay_seconds=0) == {"status": "ok"}
    assert attempts == 2

