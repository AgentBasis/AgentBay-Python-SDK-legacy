class MessageAttributes:
	PROMPT_ROLE = type("Indexed", (), {"format": staticmethod(lambda **kw: f"gen_ai.prompt.{kw['i']}.role")})()
	PROMPT_CONTENT = type("Indexed", (), {"format": staticmethod(lambda **kw: f"gen_ai.prompt.{kw['i']}.content")})()
	PROMPT_SPEAKER = type("Indexed", (), {"format": staticmethod(lambda **kw: f"gen_ai.prompt.{kw['i']}.speaker")})()

	COMPLETION_ROLE = type("Indexed", (), {"format": staticmethod(lambda **kw: f"gen_ai.response.{kw['i']}.role")})()
	COMPLETION_CONTENT = type("Indexed", (), {"format": staticmethod(lambda **kw: f"gen_ai.response.{kw['i']}.content")})()
	COMPLETION_ID = type("Indexed", (), {"format": staticmethod(lambda **kw: f"gen_ai.response.{kw['i']}.id")})()
	COMPLETION_TYPE = type("Indexed", (), {"format": staticmethod(lambda **kw: f"gen_ai.response.{kw['i']}.type")})()
	COMPLETION_FINISH_REASON = type(
		"Indexed",
		(),
		{"format": staticmethod(lambda **kw: f"gen_ai.response.{kw['i']}.finish_reason")},
	)()

	TOOL_CALL_ID = type("Indexed", (), {"format": staticmethod(lambda **kw: f"gen_ai.request.tools.{kw['i']}.id")})()
	TOOL_CALL_TYPE = type("Indexed", (), {"format": staticmethod(lambda **kw: f"gen_ai.request.tools.{kw['i']}.type")})()
	TOOL_CALL_NAME = type("Indexed", (), {"format": staticmethod(lambda **kw: f"gen_ai.request.tools.{kw['i']}.name")})()
	TOOL_CALL_DESCRIPTION = type(
		"Indexed",
		(),
		{"format": staticmethod(lambda **kw: f"gen_ai.request.tools.{kw['i']}.description")},
	)()
	TOOL_CALL_ARGUMENTS = type(
		"Indexed",
		(),
		{"format": staticmethod(lambda **kw: f"gen_ai.request.tools.{kw['i']}.arguments")},
	)()

	COMPLETION_TOOL_CALL_ID = type(
		"Indexed",
		(),
		{"format": staticmethod(lambda **kw: f"gen_ai.response.{kw['i']}.tool_calls.{kw.get('j', 0)}.id")},
	)()
	COMPLETION_TOOL_CALL_TYPE = type(
		"Indexed",
		(),
		{"format": staticmethod(lambda **kw: f"gen_ai.response.{kw['i']}.tool_calls.{kw.get('j', 0)}.type")},
	)()
	COMPLETION_TOOL_CALL_NAME = type(
		"Indexed",
		(),
		{"format": staticmethod(lambda **kw: f"gen_ai.response.{kw['i']}.tool_calls.{kw.get('j', 0)}.name")},
	)()
	COMPLETION_TOOL_CALL_ARGUMENTS = type(
		"Indexed",
		(),
		{"format": staticmethod(lambda **kw: f"gen_ai.response.{kw['i']}.tool_calls.{kw.get('j', 0)}.arguments")},
	)()
	COMPLETION_TOOL_CALL_STATUS = type(
		"Indexed",
		(),
		{"format": staticmethod(lambda **kw: f"gen_ai.response.{kw['i']}.tool_calls.{kw.get('j', 0)}.status")},
	)()
	COMPLETION_ANNOTATION_END_INDEX = type(
		"Indexed",
		(),
		{"format": staticmethod(lambda **kw: f"gen_ai.response.{kw['i']}.annotations.{kw.get('j', 0)}.end_index")},
	)()
	COMPLETION_ANNOTATION_START_INDEX = type(
		"Indexed",
		(),
		{"format": staticmethod(lambda **kw: f"gen_ai.response.{kw['i']}.annotations.{kw.get('j', 0)}.start_index")},
	)()
	COMPLETION_ANNOTATION_TITLE = type(
		"Indexed",
		(),
		{"format": staticmethod(lambda **kw: f"gen_ai.response.{kw['i']}.annotations.{kw.get('j', 0)}.title")},
	)()
	COMPLETION_ANNOTATION_URL = type(
		"Indexed",
		(),
		{"format": staticmethod(lambda **kw: f"gen_ai.response.{kw['i']}.annotations.{kw.get('j', 0)}.url")},
	)()


