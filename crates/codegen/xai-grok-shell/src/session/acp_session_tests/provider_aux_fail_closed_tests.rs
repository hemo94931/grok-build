use super::PersistenceMsg;
use super::support::{create_test_actor, test_image_content};

#[tokio::test(flavor = "current_thread")]
async fn missing_namespaced_image_description_model_fails_before_parent_sampling() {
    let local = tokio::task::LocalSet::new();
    local
        .run_until(async {
            let (gateway_tx, _gateway_rx) =
                tokio::sync::mpsc::unbounded_channel::<xai_acp_lib::AcpClientMessage>();
            let (persistence_tx, _persistence_rx) =
                tokio::sync::mpsc::unbounded_channel::<PersistenceMsg>();
            let mut actor = create_test_actor(0, 256_000, 85, gateway_tx, persistence_tx).await;
            actor.image_description_model = "deepseek/missing-image-model".to_owned();

            let error = actor
                .transcribe_user_images("describe this".to_owned(), &[test_image_content()])
                .await
                .expect_err("missing namespaced image model must fail closed");
            let rendered = error.to_string();
            assert!(
                rendered.contains(
                    "image-description provider model is unavailable or missing credentials"
                ),
                "unexpected error: {rendered}"
            );
            assert!(!rendered.contains("deepseek/missing-image-model"));
        })
        .await;
}
