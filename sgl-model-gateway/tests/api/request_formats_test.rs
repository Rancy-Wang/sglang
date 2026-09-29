use serde_json::json;

use crate::common::{
    mock_worker::{HealthStatus, MockWorkerConfig, WorkerType},
    WorkerTestContext,
};

#[cfg(test)]
mod request_format_tests {
    use super::*;

    #[tokio::test]
    async fn test_generate_request_formats() {
        let ctx = WorkerTestContext::new(vec![MockWorkerConfig {
            port: 19001,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let payload = json!({
            "text": "Hello, world!",
            "stream": false
        });

        let result = ctx.make_request("/generate", payload).await;
        assert!(result.is_ok());

        let payload = json!({
            "text": "Tell me a story",
            "sampling_params": {
                "temperature": 0.7,
                "max_new_tokens": 100,
                "top_p": 0.9
            },
            "stream": false
        });

        let result = ctx.make_request("/generate", payload).await;
        assert!(result.is_ok());

        let payload = json!({
            "input_ids": [1, 2, 3, 4, 5],
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": 50
            },
            "stream": false
        });

        let result = ctx.make_request("/generate", payload).await;
        assert!(result.is_ok());

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_v1_chat_completions_formats() {
        let ctx = WorkerTestContext::new(vec![MockWorkerConfig {
            port: 19002,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let payload = json!({
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ],
            "stream": false
        });

        let result = ctx.make_request("/v1/chat/completions", payload).await;
        assert!(result.is_ok());

        let response = result.unwrap();
        assert!(response.get("choices").is_some());
        assert!(response.get("id").is_some());
        assert_eq!(
            response.get("object").and_then(|v| v.as_str()),
            Some("chat.completion")
        );

        let payload = json!({
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Tell me a joke"}
            ],
            "temperature": 0.8,
            "max_tokens": 150,
            "top_p": 0.95,
            "stream": false
        });

        let result = ctx.make_request("/v1/chat/completions", payload).await;
        assert!(result.is_ok());

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_v1_completions_formats() {
        let ctx = WorkerTestContext::new(vec![MockWorkerConfig {
            port: 19003,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let payload = json!({
            "model": "test-model",
            "prompt": "Once upon a time",
            "max_tokens": 50,
            "stream": false
        });

        let result = ctx.make_request("/v1/completions", payload).await;
        assert!(result.is_ok());

        let response = result.unwrap();
        assert!(response.get("choices").is_some());
        assert_eq!(
            response.get("object").and_then(|v| v.as_str()),
            Some("text_completion")
        );

        let payload = json!({
            "model": "test-model",
            "prompt": ["First prompt", "Second prompt"],
            "temperature": 0.5,
            "stream": false
        });

        let result = ctx.make_request("/v1/completions", payload).await;
        assert!(result.is_ok());

        let payload = json!({
            "model": "test-model",
            "prompt": "The capital of France is",
            "max_tokens": 10,
            "logprobs": 5,
            "stream": false
        });

        let result = ctx.make_request("/v1/completions", payload).await;
        assert!(result.is_ok());

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_batch_requests() {
        let ctx = WorkerTestContext::new(vec![MockWorkerConfig {
            port: 19004,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let payload = json!({
            "text": ["First text", "Second text", "Third text"],
            "sampling_params": {
                "temperature": 0.7,
                "max_new_tokens": 50
            },
            "stream": false
        });

        let result = ctx.make_request("/generate", payload).await;
        assert!(result.is_ok());

        let payload = json!({
            "input_ids": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "stream": false
        });

        let result = ctx.make_request("/generate", payload).await;
        assert!(result.is_ok());

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_special_parameters() {
        let ctx = WorkerTestContext::new(vec![MockWorkerConfig {
            port: 19005,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let payload = json!({
            "text": "Test",
            "return_logprob": true,
            "stream": false
        });

        let result = ctx.make_request("/generate", payload).await;
        assert!(result.is_ok());

        let payload = json!({
            "text": "Generate JSON",
            "sampling_params": {
                "temperature": 0.0,
                "json_schema": "$$ANY$$"
            },
            "stream": false
        });

        let result = ctx.make_request("/generate", payload).await;
        assert!(result.is_ok());

        let payload = json!({
            "text": "Continue forever",
            "sampling_params": {
                "temperature": 0.7,
                "max_new_tokens": 100,
                "ignore_eos": true
            },
            "stream": false
        });

        let result = ctx.make_request("/generate", payload).await;
        assert!(result.is_ok());

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_error_handling() {
        let ctx = WorkerTestContext::new(vec![MockWorkerConfig {
            port: 19006,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let payload = json!({});

        let result = ctx.make_request("/generate", payload).await;
        // Mock worker accepts empty body
        assert!(result.is_ok());

        ctx.shutdown().await;
    }
}

#[test]
fn context_fields_survive_protocol_roundtrip() {
    use smg::protocols::validated::Normalizable;
    use validator::Validate;
    let value = json!({"model": "test-model", "messages": [{"role": "user", "content": "hi"}],
        "drop_message": null, "drop_rule": {"type": "message", "messages": [0]},
        "reposition": [], "return_meta_info": true, "bootstrap_host": "untrusted", "unknown": 1});
    let mut req: smg::routers::ExtendedChatRequest = serde_json::from_value(value.clone()).unwrap();
    req.normalize();
    req.validate().unwrap();
    let out = serde_json::to_value(req).unwrap();
    for key in [
        "drop_message",
        "drop_rule",
        "reposition",
        "return_meta_info",
    ] {
        assert!(out.get(key).is_some());
        assert_eq!(out[key], value[key]);
    }
    assert!(out.get("unknown").is_none());
}

#[tokio::test]
async fn context_fields_reach_http_worker() {
    use crate::common::{
        mock_worker::{capture_chat_requests, take_chat_requests},
        AppTestContext,
    };
    use axum::{
        body::{to_bytes, Body},
        http::Request,
    };
    use tower::ServiceExt;
    let port = 19070;
    let ctx = AppTestContext::new(vec![MockWorkerConfig {
        port,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate: 0.0,
    }])
    .await;
    let app = ctx.create_app().await;
    capture_chat_requests(port);
    for stream in [false, true] {
        let payload = json!({"model":"test-model", "messages":[{"role":"user","content":"hi"}],
            "stream":stream, "drop_message":null, "drop_rule":{}, "reposition":[], "return_meta_info":true});
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(payload.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert!(response.status().is_success());
        to_bytes(response.into_body(), 1 << 20).await.unwrap();
    }
    let requests = take_chat_requests(port);
    assert_eq!(requests.len(), 2);
    for value in requests {
        assert_eq!(value.get("drop_message"), Some(&serde_json::Value::Null));
        assert_eq!(value["drop_rule"], json!({}));
        assert_eq!(value["reposition"], json!([]));
        assert_eq!(value["return_meta_info"], true);
    }
    ctx.shutdown().await;
}
