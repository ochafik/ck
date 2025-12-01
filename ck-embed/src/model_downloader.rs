//! HuggingFace model downloader for static embeddings.
//!
//! Downloads model files (safetensors, tokenizer) from HuggingFace Hub.

use anyhow::{Context, Result, anyhow};
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::{Path, PathBuf};
use tokio::io::AsyncWriteExt;

/// Files required for static embedding models
const STATIC_MODEL_FILES: &[&str] = &[
    "0_StaticEmbedding/model.safetensors",
    "0_StaticEmbedding/tokenizer.json",
];

/// Get the model cache directory for ck
pub fn get_model_cache_dir() -> Result<PathBuf> {
    let cache_dir = if let Some(cache_home) = std::env::var_os("XDG_CACHE_HOME") {
        PathBuf::from(cache_home).join("ck")
    } else if let Some(home) = std::env::var_os("HOME") {
        PathBuf::from(home).join(".cache").join("ck")
    } else if let Some(appdata) = std::env::var_os("LOCALAPPDATA") {
        PathBuf::from(appdata).join("ck").join("cache")
    } else {
        PathBuf::from(".ck_models")
    };

    Ok(cache_dir.join("models"))
}

/// Convert a HuggingFace repo ID to a local directory name
fn repo_to_dirname(repo_id: &str) -> String {
    repo_id.replace('/', "__")
}

/// Check if a static embedding model is already downloaded
pub fn is_model_downloaded(repo_id: &str) -> Result<bool> {
    let cache_dir = get_model_cache_dir()?;
    let model_dir = cache_dir.join(repo_to_dirname(repo_id));

    for file in STATIC_MODEL_FILES {
        if !model_dir.join(file).exists() {
            return Ok(false);
        }
    }

    Ok(true)
}

/// Get the local path for a downloaded model
pub fn get_model_path(repo_id: &str) -> Result<PathBuf> {
    let cache_dir = get_model_cache_dir()?;
    Ok(cache_dir.join(repo_to_dirname(repo_id)))
}

/// Download a file from HuggingFace with progress reporting
async fn download_file_with_progress(
    client: &reqwest::Client,
    url: &str,
    dest: &Path,
    show_progress: bool,
) -> Result<()> {
    // Create parent directories
    if let Some(parent) = dest.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    let response = client
        .get(url)
        .send()
        .await
        .with_context(|| format!("Failed to GET {}", url))?;

    if !response.status().is_success() {
        return Err(anyhow!(
            "Failed to download {}: HTTP {}",
            url,
            response.status()
        ));
    }

    let total_size = response.content_length();

    let pb = if show_progress {
        let pb = ProgressBar::new(total_size.unwrap_or(0));
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );
        Some(pb)
    } else {
        None
    };

    // Download to a temp file first, then rename
    let temp_dest = dest.with_extension("tmp");
    let mut file = tokio::fs::File::create(&temp_dest).await?;
    let mut stream = response.bytes_stream();
    let mut downloaded: u64 = 0;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.with_context(|| "Error downloading chunk")?;
        file.write_all(&chunk).await?;
        downloaded += chunk.len() as u64;
        if let Some(ref pb) = pb {
            pb.set_position(downloaded);
        }
    }

    file.flush().await?;
    drop(file);

    // Rename temp file to final destination
    tokio::fs::rename(&temp_dest, dest).await?;

    if let Some(pb) = pb {
        pb.finish_with_message("Downloaded");
    }

    Ok(())
}

/// Download a static embedding model from HuggingFace
///
/// # Arguments
/// * `repo_id` - HuggingFace repository ID (e.g., "sentence-transformers/static-retrieval-mrl-en-v1")
/// * `show_progress` - Whether to show download progress bars
///
/// # Returns
/// Path to the downloaded model directory
pub async fn download_static_model(repo_id: &str, show_progress: bool) -> Result<PathBuf> {
    let cache_dir = get_model_cache_dir()?;
    let model_dir = cache_dir.join(repo_to_dirname(repo_id));

    // Check if already downloaded
    let mut all_exist = true;
    for file in STATIC_MODEL_FILES {
        if !model_dir.join(file).exists() {
            all_exist = false;
            break;
        }
    }

    if all_exist {
        tracing::debug!("Model {} already downloaded at {:?}", repo_id, model_dir);
        return Ok(model_dir);
    }

    tracing::info!("Downloading model {} to {:?}", repo_id, model_dir);

    let client = reqwest::Client::builder()
        .user_agent("ck-embed/0.1")
        .build()?;

    for file in STATIC_MODEL_FILES {
        let url = format!("https://huggingface.co/{}/resolve/main/{}", repo_id, file);
        let dest = model_dir.join(file);

        if dest.exists() {
            tracing::debug!("File {} already exists, skipping", file);
            continue;
        }

        if show_progress {
            eprintln!("Downloading {}...", file);
        }

        download_file_with_progress(&client, &url, &dest, show_progress)
            .await
            .with_context(|| format!("Failed to download {}", file))?;
    }

    Ok(model_dir)
}

/// Ensure a model is downloaded, downloading if necessary
///
/// This function handles both sync and async contexts:
/// - If called from within a Tokio runtime, uses the existing runtime
/// - If called from a sync context, creates a new runtime
pub fn ensure_model_downloaded(repo_id: &str, show_progress: bool) -> Result<PathBuf> {
    // Check if already downloaded first (fast path, no async needed)
    if is_model_downloaded(repo_id)? {
        return get_model_path(repo_id);
    }

    // Check if we're already in a Tokio runtime
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        // We're in an async context - use block_in_place to avoid nested runtime
        tokio::task::block_in_place(|| {
            handle.block_on(download_static_model(repo_id, show_progress))
        })
    } else {
        // Not in a runtime - create a new one
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(download_static_model(repo_id, show_progress))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repo_to_dirname() {
        assert_eq!(
            repo_to_dirname("sentence-transformers/static-retrieval-mrl-en-v1"),
            "sentence-transformers__static-retrieval-mrl-en-v1"
        );
    }

    #[test]
    fn test_get_model_cache_dir() {
        let dir = get_model_cache_dir().unwrap();
        assert!(dir.to_string_lossy().contains("ck"));
        assert!(dir.to_string_lossy().contains("models"));
    }
}
