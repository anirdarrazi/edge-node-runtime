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


def test_response_result_maps_responses_generation_controls_to_chat_completions():
    runtime = VLLMRuntime("http://localhost")
    runtime.client = DummyClient()
    runtime.execute(
        "responses",
        "google/gemma-4-E4B-it",
        [
            {
                "batch_item_id": "item_1",
                "customer_item_id": "cust_1",
                "input": {
                    "input": "Return exactly ok",
                    "max_output_tokens": 12,
                    "temperature": 0,
                    "top_p": 0.8,
                },
            }
        ],
    )
    assert runtime.client.calls[0] == (
        "/v1/chat/completions",
        {
            "model": "google/gemma-4-E4B-it",
            "messages": [{"role": "user", "content": "Return exactly ok"}],
            "max_tokens": 12,
            "temperature": 0,
            "top_p": 0.8,
        },
    )


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


def test_embedding_result_unwraps_openai_input_shape():
    runtime = VLLMRuntime("http://localhost")
    runtime.client = DummyClient()
    result = runtime.execute(
        "embeddings",
        "BAAI/bge-large-en-v1.5",
        [{"batch_item_id": "item_1", "customer_item_id": "cust_1", "input": {"input": ["setup verification ping"]}}],
    )[0]
    assert result["status"] == "completed"
    assert result["usage"]["input_texts"] == 1
    assert runtime.client.calls[0] == (
        "/v1/embeddings",
        {"model": "BAAI/bge-large-en-v1.5", "input": ["setup verification ping"]},
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


def test_embedding_runtime_microbatches_multiple_assignments():
    class DynamicEmbeddingClient:
        def __init__(self) -> None:
            self.calls = []

        def post(self, path, json):
            self.calls.append((path, json))
            payload = {
                "usage": {"prompt_tokens": len(json["input"]) * 3, "total_tokens": len(json["input"]) * 3},
                "data": [{"embedding": [float(index)]} for index, _text in enumerate(json["input"])],
            }

            class Response:
                def raise_for_status(self):
                    return None

                def json(self_nonlocal):
                    return payload

            return Response()

    runtime = VLLMRuntime("http://localhost")
    runtime.client = DynamicEmbeddingClient()

    results = runtime.execute_microbatch(
        "embeddings",
        "BAAI/bge-large-en-v1.5",
        [
            (
                "assign_1",
                [
                    {
                        "batch_item_id": "item_1",
                        "customer_item_id": "cust_1",
                        "input": {"text": "one"},
                    }
                ],
            ),
            (
                "assign_2",
                [
                    {
                        "batch_item_id": "item_2",
                        "customer_item_id": "cust_2",
                        "input": {"texts": ["two", "three"]},
                    }
                ],
            ),
        ],
    )

    assert runtime.client.calls == [
        (
            "/v1/embeddings",
            {"model": "BAAI/bge-large-en-v1.5", "input": ["one", "two", "three"]},
        )
    ]
    assert results["assign_1"][0]["usage"]["input_texts"] == 1
    assert results["assign_1"][0]["output"]["data"] == [{"embedding": [0.0]}]
    assert results["assign_2"][0]["usage"]["input_texts"] == 2
    assert results["assign_2"][0]["output"]["data"] == [{"embedding": [1.0]}, {"embedding": [2.0]}]
