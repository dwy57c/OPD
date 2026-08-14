import json
from copy import deepcopy

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


def swift_training_messages(messages: list[dict]) -> list[dict]:
    """Convert OpenAI tool-call messages to ms-swift's loss-bearing schema.

    Swift's dataset preprocessor intentionally keeps only role/content/loss on
    each message. Tool calls must therefore be represented as ``tool_call``
    messages instead of an OpenAI ``assistant.tool_calls`` side field.
    """
    rows = []
    for source in deepcopy(messages):
        role = source.get("role")
        content = source.get("content")
        tool_calls = source.get("tool_calls") or []
        if role == "assistant" and tool_calls:
            if content:
                raise ValueError("assistant training action cannot mix text and tools")
            for call in tool_calls:
                function = call.get("function") or {}
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                rows.append(
                    {
                        "role": "tool_call",
                        "content": json.dumps(
                            {
                                "name": function.get("name"),
                                "arguments": arguments,
                            },
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    }
                )
        elif role == "tool":
            rows.append({"role": "tool_response", "content": content or ""})
        else:
            rows.append({"role": role, "content": content or ""})
    return rows


def swift_cached_target_messages(messages: list[dict]) -> list[dict]:
    """Keep the state in Swift format and reserve one assistant token slot.

    The actual Teacher action is injected through ``response_token_ids`` so
    Swift cannot re-tokenize it with a different tool-call template.
    """
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("cached Teacher trajectory must end in an assistant action")
    rows = swift_training_messages(messages[:-1])
    rows.append({"role": "assistant", "content": "<cached_teacher_action>"})
    return rows


def swift_on_policy_prompt_messages(messages: list[dict]) -> list[dict]:
    """Keep a reached decision state and reserve a fresh Student action turn."""
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("on-policy decision trajectory must end in an assistant action")
    rows = swift_training_messages(messages[:-1])
    rows.append({"role": "assistant", "content": ""})
    return rows
