import json

from tau2.data_model.message import AssistantMessage, Message, ToolMessage, UserMessage


def _tool_calls(message) -> list[dict] | None:
    if not message.tool_calls:
        return None
    return [
        {
            "id": call.id,
            "type": "function",
            "function": {
                "name": call.name,
                "arguments": json.dumps(call.arguments, ensure_ascii=False),
            },
        }
        for call in message.tool_calls
    ]


def student_view(system_prompt: str, history: list[Message]) -> list[dict]:
    rows = [{"role": "system", "content": system_prompt}]
    for message in history:
        if isinstance(message, AssistantMessage):
            row = {"role": "assistant", "content": message.content or ""}
            calls = _tool_calls(message)
            if calls:
                row["tool_calls"] = calls
            rows.append(row)
        elif isinstance(message, UserMessage) and not message.is_tool_call():
            rows.append({"role": "user", "content": message.content})
        elif isinstance(message, ToolMessage) and message.requestor == "assistant":
            rows.append(
                {"role": "tool", "content": message.content, "tool_call_id": message.id}
            )
    return rows


def buyer_view(system_prompt: str, history: list[Message]) -> list[dict]:
    rows = [{"role": "system", "content": system_prompt}]
    for message in history:
        if isinstance(message, AssistantMessage) and not message.is_tool_call():
            rows.append({"role": "user", "content": message.content})
        elif isinstance(message, UserMessage):
            row = {"role": "assistant", "content": message.content or ""}
            calls = _tool_calls(message)
            if calls:
                row["tool_calls"] = calls
            rows.append(row)
        elif isinstance(message, ToolMessage) and message.requestor == "user":
            rows.append(
                {"role": "tool", "content": message.content, "tool_call_id": message.id}
            )
    return rows
