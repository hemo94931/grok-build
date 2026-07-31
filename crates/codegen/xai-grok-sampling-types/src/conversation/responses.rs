//! Responses API wire format.

use super::*;

/// Flatten `response.output` into `ConversationItem`s, preserving emission
/// order. Replaying that order byte for byte on the next turn is what keeps
/// the server-side prefix cache hot.
pub fn response_to_conversation_items(response: rs::Response) -> Vec<ConversationItem> {
    let model_id = response.model.clone();
    let model_fingerprint = response
        .metadata
        .as_ref()
        .and_then(|m| m.get("system_fingerprint"))
        .cloned()
        .filter(|s| !s.is_empty());
    let reasoning_effort = response
        .reasoning
        .as_ref()
        .and_then(|r| r.effort.clone())
        .map(crate::ReasoningEffort::from_responses_api);

    let mut items: Vec<ConversationItem> = Vec::with_capacity(response.output.len() + 1);
    let mut content = String::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let mut backend_tool_count: usize = 0;

    for item in response.output {
        match item {
            rs::OutputItem::Message(msg) => {
                for content_part in msg.content {
                    if let rs::OutputMessageContent::OutputText(text_content) = content_part {
                        if !content.is_empty() {
                            content.push('\n');
                        }
                        content.push_str(&text_content.text);
                    }
                }
            }
            rs::OutputItem::FunctionCall(fc) => {
                // Tied to the assistant turn: a ToolResult must follow each
                // one in conversation order, so they are not siblings.
                tool_calls.push(ToolCall {
                    id: Arc::<str>::from(fc.call_id),
                    name: fc.name,
                    arguments: Arc::<str>::from(fc.arguments),
                });
            }
            rs::OutputItem::Reasoning(r) => {
                items.push(ConversationItem::Reasoning(r));
            }
            // Already run server-side; kept so later turns replay the same
            // context.
            rs::OutputItem::WebSearchCall(ws) => {
                backend_tool_count += 1;
                items.push(ConversationItem::BackendToolCall(BackendToolCallItem {
                    kind: BackendToolKind::WebSearch(ws),
                }));
            }
            rs::OutputItem::CustomToolCall(ct) => {
                backend_tool_count += 1;
                items.push(ConversationItem::BackendToolCall(BackendToolCallItem {
                    kind: BackendToolKind::XSearch(ct),
                }));
            }
            rs::OutputItem::CodeInterpreterCall(ci) => {
                backend_tool_count += 1;
                items.push(ConversationItem::BackendToolCall(BackendToolCallItem {
                    kind: BackendToolKind::CodeInterpreter(ci),
                }));
            }
            rs::OutputItem::McpCall(_) => {
                backend_tool_count += 1;
            }
            _ => {}
        }
    }

    if backend_tool_count > 0 {
        tracing::info!(
            backend_tool_count,
            "response contained backend-executed tool calls"
        );
    }

    tracing::info!(model_id = %model_id, ?model_fingerprint, ?reasoning_effort, "response_to_conversation_items setting model metadata on AssistantItem");
    items.push(ConversationItem::Assistant(AssistantItem {
        content: Arc::<str>::from(content),
        tool_calls,
        model_id: Some(model_id),
        model_fingerprint,
        reasoning_effort,
    }));

    items
}

impl From<&ConversationRequest> for rs::CreateResponse {
    fn from(req: &ConversationRequest) -> Self {
        let input = build_responses_input(req);
        let tools = build_responses_tools(req);

        let tool_choice = req.tool_choice.as_ref().map(|tc| match tc {
            ConversationToolChoice::Auto => rs::ToolChoiceParam::Mode(rs::ToolChoiceOptions::Auto),
            ConversationToolChoice::None => rs::ToolChoiceParam::Mode(rs::ToolChoiceOptions::None),
            ConversationToolChoice::Required => {
                rs::ToolChoiceParam::Mode(rs::ToolChoiceOptions::Required)
            }
            ConversationToolChoice::Function(name) => {
                rs::ToolChoiceParam::Function(rs::ToolChoiceFunction { name: name.clone() })
            }
        });

        let text = req
            .json_schema
            .as_ref()
            .map(|schema| rs::ResponseTextParam {
                format: rs::TextResponseFormatConfiguration::JsonSchema(
                    rs::ResponseFormatJsonSchema {
                        description: None,
                        name: STRUCTURED_OUTPUT_SCHEMA_NAME.to_string(),
                        schema: Some(schema.clone()),
                        strict: Some(true),
                    },
                ),
                verbosity: None,
            });

        rs::CreateResponse {
            background: None,
            conversation: None,
            include: None,
            input,
            instructions: req.instructions.clone(),
            max_output_tokens: req.max_output_tokens,
            max_tool_calls: None,
            metadata: None,
            model: req.model.clone(),
            parallel_tool_calls: None,
            previous_response_id: None,
            prompt: None,
            prompt_cache_key: req.prompt_cache_key.clone(),
            prompt_cache_retention: None,
            reasoning: Some(rs::Reasoning {
                effort: req.reasoning_effort.map(|e| e.to_responses_api()),
                summary: Some(rs::ReasoningSummary::Concise),
            }),
            safety_identifier: None,
            service_tier: None,
            store: None,
            stream: None,
            stream_options: None,
            temperature: req.temperature,
            text,
            tool_choice,
            tools: if tools.is_empty() { None } else { Some(tools) },
            top_logprobs: None,
            top_p: req.top_p,
            truncation: None,
        }
    }
}

/// Reasoning items stay top-level siblings rather than folding into the
/// assistant, so the input replays the model's original order.
pub(super) fn build_responses_input(req: &ConversationRequest) -> rs::InputParam {
    let items: Vec<rs::InputItem> = req
        .items
        .iter()
        .flat_map(conversation_item_to_input_items)
        .collect();
    rs::InputParam::Items(items)
}

/// Inject the `type: "reasoning_text"` discriminator the API requires.
/// `async-openai`'s `ReasoningTextContent` has no `type` field, so it
/// serializes to `{"text": ...}` and the API answers 400. Delete this once
/// upstream grows the field.
pub fn patch_reasoning_text_types(body: &mut serde_json::Value) {
    let Some(input) = body.get_mut("input").and_then(|v| v.as_array_mut()) else {
        return;
    };
    for item in input.iter_mut() {
        if item.get("type").and_then(|t| t.as_str()) != Some("reasoning") {
            continue;
        }
        let Some(content) = item.get_mut("content").and_then(|c| c.as_array_mut()) else {
            continue;
        };
        for c in content.iter_mut() {
            if let Some(obj) = c.as_object_mut() {
                obj.entry("type")
                    .or_insert_with(|| serde_json::Value::String("reasoning_text".into()));
            }
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ResponsesRequestBuildError {
    #[error(transparent)]
    Validation(#[from] ConversationValidationError),
    #[error("failed to serialize responses request: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("serialized responses request has no input array")]
    MissingInput,
}

/// Immutable normal Responses request immediately before transport defaults/headers.
///
/// A checkpoint-bearing request is serialized by preserving the raw canonical
/// prefix and converting only the typed live tail.
#[derive(Debug, Clone)]
pub struct FinalResponsesRequest {
    body: serde_json::Value,
}

impl FinalResponsesRequest {
    pub fn body(&self) -> &serde_json::Value {
        &self.body
    }

    pub fn into_body(self) -> serde_json::Value {
        self.body
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(&self.body)
    }
}

impl TryFrom<&ConversationRequest> for FinalResponsesRequest {
    type Error = ResponsesRequestBuildError;

    fn try_from(request: &ConversationRequest) -> Result<Self, Self::Error> {
        request.validate_for_backend(&crate::ApiBackend::Responses)?;
        let checkpoint = request.items.first().and_then(|item| match item {
            ConversationItem::ResponsesCompactionCheckpoint(checkpoint) => Some(checkpoint),
            _ => None,
        });
        let mut typed_request = request.clone();
        if checkpoint.is_some() {
            typed_request.items.remove(0);
        }

        let create_response = rs::CreateResponse::from(&typed_request);
        let mut body = serde_json::to_value(create_response)?;
        patch_reasoning_text_types(&mut body);

        if let Some(value) = request.prompt_cache_options.clone() {
            body["prompt_cache_options"] = value;
        }
        if let Some(value) = request.prompt_cache_retention.clone() {
            body["prompt_cache_retention"] = serde_json::Value::String(value);
        }
        if let Some(value) = request.service_tier.clone() {
            body["service_tier"] = serde_json::Value::String(value);
        }

        let extra_tools = extra_tool_entries(&request.hosted_tools);
        if !extra_tools.is_empty() {
            let tools = body
                .as_object_mut()
                .expect("CreateResponse serializes as an object")
                .entry("tools")
                .or_insert_with(|| serde_json::Value::Array(Vec::new()));
            let tools = tools
                .as_array_mut()
                .expect("CreateResponse tools serializes as an array");
            tools.extend(extra_tools);
        }

        if let Some(checkpoint) = checkpoint {
            let tail = body
                .get_mut("input")
                .and_then(serde_json::Value::as_array_mut)
                .ok_or(ResponsesRequestBuildError::MissingInput)?;
            let mut input = checkpoint.output.clone();
            input.append(tail);
            body["input"] = serde_json::Value::Array(input);
        }

        Ok(Self { body })
    }
}

/// Serialize JSON with recursively sorted object keys.
pub fn canonical_json_bytes(value: &serde_json::Value) -> Result<Vec<u8>, serde_json::Error> {
    fn sort(value: &serde_json::Value) -> serde_json::Value {
        match value {
            serde_json::Value::Object(object) => {
                let mut entries: Vec<_> = object.iter().collect();
                entries.sort_unstable_by(|(left, _), (right, _)| left.cmp(right));
                let mut sorted = serde_json::Map::new();
                for (key, value) in entries {
                    sorted.insert(key.clone(), sort(value));
                }
                serde_json::Value::Object(sorted)
            }
            serde_json::Value::Array(values) => {
                serde_json::Value::Array(values.iter().map(sort).collect())
            }
            scalar => scalar.clone(),
        }
    }

    serde_json::to_vec(&sort(value))
}

fn conversation_item_to_input_items(item: &ConversationItem) -> Vec<rs::InputItem> {
    match item {
        ConversationItem::System(s) => {
            vec![rs::InputItem::EasyMessage(rs::EasyInputMessage {
                r#type: rs::MessageType::Message,
                role: rs::Role::System,
                content: rs::EasyInputContent::Text(s.content.as_ref().to_owned()),
            })]
        }
        ConversationItem::User(u) => {
            let content = content_parts_to_easy_input_content(&u.content);
            vec![rs::InputItem::EasyMessage(rs::EasyInputMessage {
                r#type: rs::MessageType::Message,
                role: rs::Role::User,
                content,
            })]
        }
        ConversationItem::Reasoning(r) => {
            // `status` is output-only and rejected on input.
            let mut r = r.clone();
            r.status = None;
            vec![rs::InputItem::Item(rs::Item::Reasoning(r))]
        }
        ConversationItem::Assistant(a) => {
            let mut items = Vec::new();

            if !a.content.is_empty() {
                items.push(rs::InputItem::EasyMessage(rs::EasyInputMessage {
                    r#type: rs::MessageType::Message,
                    role: rs::Role::Assistant,
                    content: rs::EasyInputContent::Text(a.content.as_ref().to_owned()),
                }));
            }

            for tc in &a.tool_calls {
                let arguments = sanitize_tool_arguments(&tc.id, &tc.name, tc.arguments.clone());
                items.push(rs::InputItem::Item(rs::Item::FunctionCall(
                    rs::FunctionToolCall {
                        call_id: tc.id.as_ref().to_owned(),
                        name: tc.name.clone(),
                        arguments: arguments.as_ref().to_owned(),
                        id: None,
                        status: None,
                    },
                )));
            }

            items
        }
        ConversationItem::ToolResult(t) => {
            let output = if t.images.is_empty() {
                rs::FunctionCallOutput::Text(t.content.as_ref().to_owned())
            } else {
                let mut parts: Vec<rs::InputContent> =
                    vec![rs::InputContent::InputText(rs::InputTextContent {
                        text: t.content.as_ref().to_owned(),
                    })];
                for img in &t.images {
                    if let ContentPart::Image { url } = img {
                        parts.push(rs::InputContent::InputImage(rs::InputImageContent {
                            detail: rs::ImageDetail::Auto,
                            file_id: None,
                            image_url: Some(url.as_ref().to_owned()),
                        }));
                    }
                }
                rs::FunctionCallOutput::Content(parts)
            };
            vec![rs::InputItem::Item(rs::Item::FunctionCallOutput(
                rs::FunctionCallOutputItemParam {
                    call_id: t.tool_call_id.clone(),
                    output,
                    id: None,
                    status: None,
                },
            ))]
        }
        ConversationItem::BackendToolCall(b) => {
            vec![match &b.kind {
                BackendToolKind::WebSearch(ws) => {
                    rs::InputItem::Item(rs::Item::WebSearchCall(ws.clone()))
                }
                BackendToolKind::XSearch(ct) => {
                    rs::InputItem::Item(rs::Item::CustomToolCall(ct.clone()))
                }
                BackendToolKind::CodeInterpreter(ci) => {
                    rs::InputItem::Item(rs::Item::CodeInterpreterCall(ci.clone()))
                }
            }]
        }
        ConversationItem::ResponsesCompactionCheckpoint(_) => {
            unreachable!("checkpoint input must be flattened by FinalResponsesRequest")
        }
    }
}

fn content_parts_to_easy_input_content(parts: &[ContentPart]) -> rs::EasyInputContent {
    if parts.len() == 1
        && let ContentPart::Text { text } = &parts[0]
    {
        return rs::EasyInputContent::Text(text.as_ref().to_owned());
    }

    let items: Vec<rs::InputContent> = parts
        .iter()
        .map(|part| match part {
            ContentPart::Text { text } => rs::InputContent::InputText(rs::InputTextContent {
                text: text.as_ref().to_owned(),
            }),
            ContentPart::Image { url } => rs::InputContent::InputImage(rs::InputImageContent {
                image_url: Some(url.as_ref().to_owned()),
                file_id: None,
                detail: rs::ImageDetail::default(),
            }),
        })
        .collect();

    rs::EasyInputContent::ContentList(items)
}

/// Client function tools plus backend-hosted tools. On a name collision the
/// hosted tool wins, because sending both is rejected as a duplicate.
fn build_responses_tools(req: &ConversationRequest) -> Vec<rs::Tool> {
    let mut tools: Vec<rs::Tool> = req
        .tools
        .iter()
        .filter(|t| {
            let collides = req.hosted_tools.iter().any(|h| h.wire_name() == t.name);
            if collides {
                tracing::warn!(
                    tool = %t.name,
                    "dropping function tool that collides with a backend-hosted tool"
                );
            }
            !collides
        })
        .map(|t| {
            rs::Tool::Function(rs::FunctionTool {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: Some(t.parameters.clone()),
                strict: None,
            })
        })
        .collect();

    for hosted in &req.hosted_tools {
        match hosted {
            HostedTool::WebSearch { options } => {
                // An empty allowlist is unbounded, so it emits no filter.
                let filters = options
                    .as_ref()
                    .and_then(|o| o.allowed_domains.as_deref())
                    .filter(|domains| !domains.is_empty())
                    .map(|domains| rs::WebSearchToolFilters {
                        allowed_domains: Some(domains.to_vec()),
                    });
                tools.push(rs::Tool::WebSearch(rs::WebSearchTool {
                    filters,
                    ..Default::default()
                }));
            }
            // XSearch is xAI-specific — not in async_openai's rs::Tool enum.
            // Injected as raw JSON by the sampler client after serialization.
            HostedTool::XSearch { .. } => {}
        }
    }

    tools
}

/// Hosted tools with no `rs::Tool` variant. The sampler client splices these
/// into the serialized `tools` array.
pub fn extra_tool_entries(hosted_tools: &[HostedTool]) -> Vec<serde_json::Value> {
    let mut entries = Vec::new();
    for tool in hosted_tools {
        match tool {
            // WebSearch ships natively (rs::Tool::WebSearch), so no JSON entry here.
            HostedTool::WebSearch { .. } => {}
            HostedTool::XSearch { options } => {
                entries.push(match options {
                    Some(o) => o.to_tool_entry(),
                    None => XSearchOptions::default().to_tool_entry(),
                });
            }
        }
    }
    entries
}
