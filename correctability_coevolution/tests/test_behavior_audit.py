from coevo.audit import BehaviorAuditor


def test_behavior_audit_counts_lookup_clarification_and_unsupported_fact():
    messages = [
        {"role": "system", "content": "Use tools before stating booking facts."},
        {"role": "user", "content": "I need to change my flight."},
        {
            "role": "assistant",
            "content": "Could you provide the booking number so I can check it?",
        },
        {"role": "user", "content": "It is ABC123."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_booking_details",
                        "arguments": '{"booking":"ABC123"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": "Booking ABC123 is confirmed.",
            "tool_call_id": "1",
        },
        {
            "role": "assistant",
            "content": "The change fee is $275 and the booking is confirmed.",
        },
    ]
    report = BehaviorAuditor().analyze(messages)
    assert report.decisions == 3
    assert report.clarifying_decisions == 1
    assert report.lookup_decisions == 1
    assert report.ungrounded_assertion_count == 1
    assert report.actions[-1].ungrounded_assertions == ("$275",)
