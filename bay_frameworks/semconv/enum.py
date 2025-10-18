class LLMRequestTypeValues:
	TEXT = type("Enum", (), {"value": "text"})()
	CHAT = type("Enum", (), {"value": "chat"})()
	EMBEDDING = type("Enum", (), {"value": "embedding"})()
	IMAGE = type("Enum", (), {"value": "image"})()
	AUDIO = type("Enum", (), {"value": "audio"})()


