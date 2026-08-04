use std::collections::HashMap;

use serde_json::{Map, Value, json};
use xai_grok_sampling_types::SamplingError;
use xai_grok_sampling_types::messages::{
    self, ContentBlock, MessageContent, MessageDeltaBody, MessageDeltaUsage, MessageStreamEvent,
    MessagesRequest, MessagesResponse, MessagesUsage, StopReason, StreamDelta, StreamError,
    SystemParam, ToolChoiceParam, ToolResultContent,
};

pub(crate) fn radius_payload(
    model: &str,
    request: &MessagesRequest,
    session_id: Option<&str>,
    reasoning: Option<&str>,
) -> Value {
    let mut options = Map::new();
    if let Some(value) = request.temperature {
        options.insert("temperature".to_owned(), json!(value));
    }
    if request.max_tokens > 0 {
        options.insert("maxTokens".to_owned(), json!(request.max_tokens));
    }
    if let Some(value) = reasoning {
        options.insert("reasoning".to_owned(), json!(value));
    }
    if let Some(value) = session_id.filter(|value| !value.is_empty()) {
        options.insert("sessionId".to_owned(), json!(value));
    }
    if let Some(value) = tool_choice(request.tool_choice.as_ref()) {
        options.insert("toolChoice".to_owned(), value);
    }

    let mut context = Map::new();
    if let Some(system) = system_prompt(request.system.as_ref()) {
        context.insert("systemPrompt".to_owned(), Value::String(system));
    }
    context.insert("messages".to_owned(), Value::Array(pi_messages(request)));
    if let Some(tools) = request.tools.as_ref().filter(|tools| !tools.is_empty()) {
        context.insert(
            "tools".to_owned(),
            Value::Array(
                tools
                    .iter()
                    .map(|tool| {
                        json!({
                            "name": tool.name,
                            "description": tool.description.as_deref().unwrap_or_default(),
                            "parameters": tool.input_schema,
                        })
                    })
                    .collect(),
            ),
        );
    }
    json!({
        "model": model,
        "context": context,
        "options": options,
    })
}

fn system_prompt(system: Option<&SystemParam>) -> Option<String> {
    let value = match system? {
        SystemParam::Text(value) => value.clone(),
        SystemParam::Blocks(blocks) => blocks
            .iter()
            .map(|block| block.text.as_str())
            .collect::<Vec<_>>()
            .join("\n\n"),
    };
    (!value.trim().is_empty()).then_some(value)
}

fn tool_choice(choice: Option<&ToolChoiceParam>) -> Option<Value> {
    Some(match choice? {
        ToolChoiceParam::Auto => json!("auto"),
        ToolChoiceParam::Any => json!("required"),
        ToolChoiceParam::Tool { name } => {
            json!({"type": "function", "function": {"name": name}})
        }
    })
}

fn pi_messages(request: &MessagesRequest) -> Vec<Value> {
    let mut result = Vec::new();
    let mut tool_names = HashMap::<String, String>::new();
    for message in &request.messages {
        match message.role {
            messages::MessageRole::Assistant => {
                let content = assistant_content(&message.content, &mut tool_names);
                if !content.is_empty() {
                    result.push(json!({
                        "role": "assistant",
                        "content": content,
                        "api": "pi-messages",
                        "provider": "radius",
                        "model": request.model,
                        "usage": empty_usage(),
                        "stopReason": "stop",
                        "timestamp": 0,
                    }));
                }
            }
            messages::MessageRole::User => {
                append_user_messages(&mut result, &message.content, &tool_names);
            }
        }
    }
    result
}

fn assistant_content(
    content: &MessageContent,
    tool_names: &mut HashMap<String, String>,
) -> Vec<Value> {
    match content {
        MessageContent::Text(text) => vec![json!({"type": "text", "text": text})],
        MessageContent::Blocks(blocks) => blocks
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text, .. } => Some(json!({"type": "text", "text": text})),
                ContentBlock::Thinking {
                    thinking,
                    signature,
                } => Some(json!({
                    "type": "thinking",
                    "thinking": thinking,
                    "thinkingSignature": signature,
                })),
                ContentBlock::RedactedThinking { data } => Some(json!({
                    "type": "thinking",
                    "thinking": "",
                    "thinkingSignature": data,
                    "redacted": true,
                })),
                ContentBlock::ToolUse {
                    id, name, input, ..
                } => {
                    tool_names.insert(id.clone(), name.clone());
                    Some(json!({
                        "type": "toolCall",
                        "id": id,
                        "name": name,
                        "arguments": input,
                    }))
                }
                _ => None,
            })
            .collect(),
    }
}

fn append_user_messages(
    output: &mut Vec<Value>,
    content: &MessageContent,
    tool_names: &HashMap<String, String>,
) {
    match content {
        MessageContent::Text(text) => output.push(json!({
            "role": "user",
            "content": text,
            "timestamp": 0,
        })),
        MessageContent::Blocks(blocks) => {
            let mut user_content = Vec::new();
            for block in blocks {
                match block {
                    ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        ..
                    } => {
                        flush_user_content(output, &mut user_content);
                        output.push(json!({
                            "role": "toolResult",
                            "toolCallId": tool_use_id,
                            "toolName": tool_names.get(tool_use_id).cloned().unwrap_or_default(),
                            "content": tool_result_content(content),
                            "isError": false,
                            "timestamp": 0,
                        }));
                    }
                    ContentBlock::Text { text, .. } => {
                        user_content.push(json!({"type": "text", "text": text}));
                    }
                    ContentBlock::Image { source, .. } => {
                        user_content.push(image_content(source));
                    }
                    _ => {}
                }
            }
            flush_user_content(output, &mut user_content);
        }
    }
}

fn flush_user_content(output: &mut Vec<Value>, content: &mut Vec<Value>) {
    if content.is_empty() {
        return;
    }
    output.push(json!({
        "role": "user",
        "content": std::mem::take(content),
        "timestamp": 0,
    }));
}

fn tool_result_content(content: &ToolResultContent) -> Vec<Value> {
    match content {
        ToolResultContent::Text(text) => vec![json!({"type": "text", "text": text})],
        ToolResultContent::Blocks(blocks) => blocks
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text, .. } => Some(json!({"type": "text", "text": text})),
                ContentBlock::Image { source, .. } => Some(image_content(source)),
                _ => None,
            })
            .collect(),
    }
}

fn image_content(source: &messages::ImageSource) -> Value {
    match source {
        messages::ImageSource::Base64 { media_type, data } => {
            json!({"type": "image", "data": data, "mimeType": media_type})
        }
        messages::ImageSource::Url { url } => {
            json!({"type": "text", "text": format!("[image: {url}]")})
        }
    }
}

fn empty_usage() -> Value {
    json!({
        "input": 0,
        "output": 0,
        "cacheRead": 0,
        "cacheWrite": 0,
        "totalTokens": 0,
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0}
    })
}

pub(crate) struct PiMessagesEventDecoder {
    model: String,
    message_id: String,
    text: HashMap<u32, String>,
    thinking: HashMap<u32, String>,
    tool_json: HashMap<u32, String>,
}

impl PiMessagesEventDecoder {
    pub(crate) fn new(model: String, message_id: String) -> Self {
        Self {
            model,
            message_id,
            text: HashMap::new(),
            thinking: HashMap::new(),
            tool_json: HashMap::new(),
        }
    }

    pub(crate) fn decode(&mut self, data: &str) -> Result<Vec<MessageStreamEvent>, SamplingError> {
        let event: Value = serde_json::from_str(data).map_err(SamplingError::Serialization)?;
        let event_type = event.get("type").and_then(Value::as_str).ok_or_else(|| {
            SamplingError::EventStreamError("Radius event is missing type".into())
        })?;
        match event_type {
            "start" => Ok(vec![MessageStreamEvent::MessageStart {
                message: MessagesResponse {
                    id: self.message_id.clone(),
                    r#type: "message".to_owned(),
                    role: "assistant".to_owned(),
                    content: Vec::new(),
                    model: self.model.clone(),
                    stop_reason: None,
                    usage: MessagesUsage::default(),
                },
            }]),
            "text_start" => {
                let index = content_index(&event)?;
                self.text.insert(index, String::new());
                Ok(vec![MessageStreamEvent::ContentBlockStart {
                    index,
                    content_block: ContentBlock::Text {
                        text: String::new(),
                        cache_control: None,
                    },
                }])
            }
            "text_delta" => {
                let index = content_index(&event)?;
                let delta = string(&event, "delta")?;
                self.text.entry(index).or_default().push_str(delta);
                Ok(vec![MessageStreamEvent::ContentBlockDelta {
                    index,
                    delta: StreamDelta::TextDelta {
                        text: delta.to_owned(),
                    },
                }])
            }
            "text_end" => {
                let index = content_index(&event)?;
                let content = string(&event, "content")?;
                let streamed = self.text.remove(&index).unwrap_or_default();
                let mut events = Vec::new();
                if let Some(suffix) = content
                    .strip_prefix(&streamed)
                    .filter(|value| !value.is_empty())
                {
                    events.push(MessageStreamEvent::ContentBlockDelta {
                        index,
                        delta: StreamDelta::TextDelta {
                            text: suffix.to_owned(),
                        },
                    });
                }
                events.push(MessageStreamEvent::ContentBlockStop { index });
                Ok(events)
            }
            "thinking_start" => {
                let index = content_index(&event)?;
                self.thinking.insert(index, String::new());
                Ok(vec![MessageStreamEvent::ContentBlockStart {
                    index,
                    content_block: ContentBlock::Thinking {
                        thinking: String::new(),
                        signature: String::new(),
                    },
                }])
            }
            "thinking_delta" => {
                let index = content_index(&event)?;
                let delta = string(&event, "delta")?;
                self.thinking.entry(index).or_default().push_str(delta);
                Ok(vec![MessageStreamEvent::ContentBlockDelta {
                    index,
                    delta: StreamDelta::ThinkingDelta {
                        thinking: delta.to_owned(),
                    },
                }])
            }
            "thinking_end" => {
                let index = content_index(&event)?;
                let content = string(&event, "content")?;
                let streamed = self.thinking.remove(&index).unwrap_or_default();
                let mut events = Vec::new();
                if let Some(suffix) = content
                    .strip_prefix(&streamed)
                    .filter(|value| !value.is_empty())
                {
                    events.push(MessageStreamEvent::ContentBlockDelta {
                        index,
                        delta: StreamDelta::ThinkingDelta {
                            thinking: suffix.to_owned(),
                        },
                    });
                }
                if let Some(signature) = event.get("contentSignature").and_then(Value::as_str) {
                    events.push(MessageStreamEvent::ContentBlockDelta {
                        index,
                        delta: StreamDelta::SignatureDelta {
                            signature: signature.to_owned(),
                        },
                    });
                }
                events.push(MessageStreamEvent::ContentBlockStop { index });
                Ok(events)
            }
            "toolcall_start" => {
                let index = content_index(&event)?;
                self.tool_json.insert(index, String::new());
                Ok(vec![MessageStreamEvent::ContentBlockStart {
                    index,
                    content_block: ContentBlock::ToolUse {
                        id: string(&event, "id")?.to_owned(),
                        name: string(&event, "toolName")?.to_owned(),
                        input: json!({}),
                        cache_control: None,
                    },
                }])
            }
            "toolcall_delta" => {
                let index = content_index(&event)?;
                let delta = string(&event, "delta")?;
                self.tool_json.entry(index).or_default().push_str(delta);
                Ok(vec![MessageStreamEvent::ContentBlockDelta {
                    index,
                    delta: StreamDelta::InputJsonDelta {
                        partial_json: delta.to_owned(),
                    },
                }])
            }
            "toolcall_end" => {
                let index = content_index(&event)?;
                let streamed = self.tool_json.remove(&index).unwrap_or_default();
                let complete = event
                    .pointer("/toolCall/arguments")
                    .map(serde_json::to_string)
                    .transpose()
                    .map_err(SamplingError::Serialization)?
                    .unwrap_or_default();
                let mut events = Vec::new();
                if streamed.is_empty() && !complete.is_empty() {
                    events.push(MessageStreamEvent::ContentBlockDelta {
                        index,
                        delta: StreamDelta::InputJsonDelta {
                            partial_json: complete,
                        },
                    });
                }
                events.push(MessageStreamEvent::ContentBlockStop { index });
                Ok(events)
            }
            "done" => {
                let usage = usage(&event);
                Ok(vec![
                    MessageStreamEvent::MessageDelta {
                        delta: MessageDeltaBody {
                            stop_reason: Some(stop_reason(
                                event.get("reason").and_then(Value::as_str),
                            )),
                            stop_sequence: None,
                            stop_details: None,
                        },
                        usage,
                    },
                    MessageStreamEvent::MessageStop,
                ])
            }
            "error" => Ok(vec![MessageStreamEvent::Error {
                error: StreamError {
                    r#type: event
                        .get("reason")
                        .and_then(Value::as_str)
                        .unwrap_or("error")
                        .to_owned(),
                    message: event
                        .get("errorMessage")
                        .and_then(Value::as_str)
                        .unwrap_or("Radius request failed")
                        .to_owned(),
                },
            }]),
            other => Err(SamplingError::EventStreamError(format!(
                "unknown Radius event type `{other}`"
            ))),
        }
    }
}

fn content_index(event: &Value) -> Result<u32, SamplingError> {
    event
        .get("contentIndex")
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| {
            SamplingError::EventStreamError("Radius event has invalid contentIndex".into())
        })
}

fn string<'a>(event: &'a Value, key: &str) -> Result<&'a str, SamplingError> {
    event
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| SamplingError::EventStreamError(format!("Radius event is missing {key}")))
}

fn stop_reason(reason: Option<&str>) -> StopReason {
    match reason {
        Some("length") => StopReason::MaxTokens,
        Some("toolUse") => StopReason::ToolUse,
        _ => StopReason::EndTurn,
    }
}

fn usage(event: &Value) -> MessageDeltaUsage {
    let usage = event.get("usage").unwrap_or(&Value::Null);
    MessageDeltaUsage {
        output_tokens: u32_field(usage, "output"),
        input_tokens: Some(u32_field(usage, "input")),
        cache_read_input_tokens: Some(u32_field(usage, "cacheRead")),
        cache_creation_input_tokens: Some(u32_field(usage, "cacheWrite")),
    }
}

fn u32_field(value: &Value, key: &str) -> u32 {
    value
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radius_terminal_event_maps_usage_and_stop_reason() {
        let mut decoder = PiMessagesEventDecoder::new("radius-1".into(), "req-1".into());
        let events = decoder
            .decode(
                r#"{"type":"done","reason":"toolUse","usage":{"input":10,"output":3,"cacheRead":2,"cacheWrite":1}}"#,
            )
            .unwrap();
        assert!(matches!(
            &events[0],
            MessageStreamEvent::MessageDelta {
                delta: MessageDeltaBody {
                    stop_reason: Some(StopReason::ToolUse),
                    ..
                },
                usage: MessageDeltaUsage {
                    output_tokens: 3,
                    input_tokens: Some(10),
                    ..
                }
            }
        ));
        assert!(matches!(events[1], MessageStreamEvent::MessageStop));
    }

    #[test]
    fn radius_text_end_emits_unstreamed_suffix() {
        let mut decoder = PiMessagesEventDecoder::new("radius-1".into(), "req-1".into());
        decoder
            .decode(r#"{"type":"text_start","contentIndex":0}"#)
            .unwrap();
        decoder
            .decode(r#"{"type":"text_delta","contentIndex":0,"delta":"hel"}"#)
            .unwrap();
        let events = decoder
            .decode(r#"{"type":"text_end","contentIndex":0,"content":"hello"}"#)
            .unwrap();
        assert!(matches!(
            &events[0],
            MessageStreamEvent::ContentBlockDelta {
                delta: StreamDelta::TextDelta { text },
                ..
            } if text == "lo"
        ));
    }
}
