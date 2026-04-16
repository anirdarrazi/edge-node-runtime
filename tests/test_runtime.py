from node_agent.runtime import VLLMRuntime


class DummyClient:
    def __init__(self) -> None:
        self.responses = {
            "/v1/chat/completions": {
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                "choices": [{"message": {"content": "hello"}}],
            },
            "/v1/embeddings": {
                "usage": {"prompt_tokens": 12, "total_tokens": 12},
                "data": [{"embedding": [0.1, 0.2]}],
            },
        }
        self.calls = []

    def post(self, path, json):
        self.calls.append((path, json))
        payload = self.responses[path]

        class Response:
            def raise_for_status(self):
                return None

            def json(self_nonlocal):
                return payload

        return Response()


def test_response_result_shape():
    runtime = VLLMRuntime("http://localhost")
    runtime.client = DummyClient()
    result = runtime.execute(
        "responses",
        "meta-llama/Llama-3.1-8B-Instruct",
        [{"batch_item_id": "item_1", "customer_item_id": "cust_1", "input": {"messages": [{"role": "user", "content": "hi"}]}}],
    )[0]
    assert result["status"] == "completed"
    assert result["usage"]["total_tokens"] == 15


def test_embedding_result_unwraps_texts_input_shape():
    runtime = VLLMRuntime("http://localhost")
    runtime.client = DummyClient()
    result = runtime.execute(
        "embeddings",
        "BAAI/bge-large-en-v1.5",
        [{"batch_item_id": "item_1", "customer_item_id": "cust_1", "input": {"texts": ["workflow green"]}}],
    )[0]
    assert result["status"] == "completed"
    assert result["usage"]["input_texts"] == 1
    assert runtime.client.calls[0] == (
        "/v1/embeddings",
        {"model": "BAAI/bge-large-en-v1.5", "input": ["workflow green"]},
    )


def test_embedding_result_accepts_multiple_raw_texts():
    runtime = VLLMRuntime("http://localhost")
    runtime.client = DummyClient()
    result = runtime.execute(
        "embeddings",
        "BAAI/bge-large-en-v1.5",
        [{"batch_item_id": "item_1", "customer_item_id": "cust_1", "input": ["one", "two"]}],
    )[0]
    assert result["usage"]["input_texts"] == 2
    assert runtime.client.calls[0] == (
        "/v1/embeddings",
        {"model": "BAAI/bge-large-en-v1.5", "input": ["one", "two"]},
    )
