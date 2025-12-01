//! Static embedding model implementation.
//!
//! Static embeddings use pre-computed token embeddings with mean pooling,
//! providing 100-400x speedup over transformer models with ~87-92% quality retention.
//!
//! Architecture: Tokenize -> Embedding Lookup -> Mean Pool -> Output Vector

use crate::model_downloader::{ensure_model_downloaded, is_model_downloaded};
use crate::{Embedder, ModelDownloadCallback};
use anyhow::{Context, Result, anyhow};
use safetensors::SafeTensors;
use std::path::Path;
use tokenizers::Tokenizer;

/// Known static embedding models from HuggingFace
pub const STATIC_RETRIEVAL_EN: &str = "sentence-transformers/static-retrieval-mrl-en-v1";
pub const STATIC_SIMILARITY_MULTILINGUAL: &str =
    "sentence-transformers/static-similarity-mrl-multilingual-v1";

/// Valid MRL (Matryoshka Representation Learning) dimensions
pub const MRL_DIMENSIONS: &[usize] = &[32, 64, 128, 256, 512, 1024];

/// Static embedding model implementation.
///
/// Uses pre-computed token embeddings with mean pooling for ultra-fast inference.
/// Supports Matryoshka Representation Learning (MRL) for dimension truncation.
pub struct StaticEmbedder {
    tokenizer: Tokenizer,
    /// Embedding matrix: [vocab_size, embedding_dim]
    embeddings: Vec<Vec<f32>>,
    vocab_size: usize,
    embedding_dim: usize,
    /// Optional dimension truncation (MRL)
    truncate_dim: Option<usize>,
    model_name: String,
}

impl StaticEmbedder {
    /// Create a new StaticEmbedder from local model files.
    ///
    /// # Arguments
    /// * `model_path` - Path to the model directory containing `0_StaticEmbedding/`
    /// * `truncate_dim` - Optional MRL dimension truncation (must be <= embedding_dim)
    pub fn new(model_path: &Path, truncate_dim: Option<usize>) -> Result<Self> {
        let static_dir = model_path.join("0_StaticEmbedding");

        // Load tokenizer
        let tokenizer_path = static_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("Failed to load tokenizer from {:?}: {}", tokenizer_path, e))?;

        // Load safetensors
        let model_path_file = static_dir.join("model.safetensors");
        let weights_data = std::fs::read(&model_path_file)
            .with_context(|| format!("Failed to read {:?}", model_path_file))?;

        let tensors = SafeTensors::deserialize(&weights_data)
            .with_context(|| "Failed to deserialize safetensors")?;

        // Find the embedding tensor - try common names
        let embedding_tensor = tensors
            .tensor("embedding.weight")
            .or_else(|_| tensors.tensor("embeddings.weight"))
            .or_else(|_| tensors.tensor("weight"))
            .with_context(|| {
                let names: Vec<_> = tensors.names().into_iter().collect();
                format!(
                    "Could not find embedding tensor. Available tensors: {:?}",
                    names
                )
            })?;

        let shape = embedding_tensor.shape();
        if shape.len() != 2 {
            return Err(anyhow!(
                "Expected 2D embedding tensor, got shape: {:?}",
                shape
            ));
        }

        let vocab_size = shape[0];
        let embedding_dim = shape[1];

        tracing::debug!(
            "Loading static embeddings: vocab_size={}, embedding_dim={}",
            vocab_size,
            embedding_dim
        );

        // Parse embedding data
        let data = embedding_tensor.data();
        let embeddings = parse_embeddings(data, vocab_size, embedding_dim)?;

        // Validate truncate_dim
        if let Some(dim) = truncate_dim {
            if dim > embedding_dim {
                return Err(anyhow!(
                    "truncate_dim ({}) cannot be greater than embedding_dim ({})",
                    dim,
                    embedding_dim
                ));
            }
            if !MRL_DIMENSIONS.contains(&dim) {
                tracing::warn!(
                    "truncate_dim {} is not a standard MRL dimension {:?}",
                    dim,
                    MRL_DIMENSIONS
                );
            }
        }

        let model_name = model_path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "static".to_string());

        Ok(Self {
            tokenizer,
            embeddings,
            vocab_size,
            embedding_dim,
            truncate_dim,
            model_name,
        })
    }

    /// Create a StaticEmbedder from a HuggingFace model ID.
    ///
    /// Downloads the model if not already cached.
    ///
    /// # Arguments
    /// * `repo_id` - HuggingFace repository ID (e.g., "sentence-transformers/static-retrieval-mrl-en-v1")
    /// * `truncate_dim` - Optional MRL dimension truncation
    /// * `progress_callback` - Optional callback for download progress
    pub fn from_huggingface(
        repo_id: &str,
        truncate_dim: Option<usize>,
        progress_callback: Option<ModelDownloadCallback>,
    ) -> Result<Self> {
        // Report progress
        if let Some(ref callback) = progress_callback {
            if is_model_downloaded(repo_id)? {
                callback(&format!("Using cached model: {}", repo_id));
            } else {
                callback(&format!("Downloading model: {}", repo_id));
            }
        }

        // Download if needed
        let model_path = ensure_model_downloaded(repo_id, progress_callback.is_some())?;

        if let Some(ref callback) = progress_callback {
            callback("Loading model...");
        }

        let mut embedder = Self::new(&model_path, truncate_dim)?;
        embedder.model_name = repo_id.to_string();

        if let Some(ref callback) = progress_callback {
            callback(&format!(
                "Model loaded: {}d embeddings (vocab: {})",
                embedder.output_dim(),
                embedder.vocab_size
            ));
        }

        Ok(embedder)
    }

    /// Create a StaticEmbedder for English retrieval tasks.
    pub fn english_retrieval(truncate_dim: Option<usize>) -> Result<Self> {
        Self::from_huggingface(STATIC_RETRIEVAL_EN, truncate_dim, None)
    }

    /// Create a StaticEmbedder for multilingual similarity tasks.
    pub fn multilingual_similarity(truncate_dim: Option<usize>) -> Result<Self> {
        Self::from_huggingface(STATIC_SIMILARITY_MULTILINGUAL, truncate_dim, None)
    }

    /// Get the output dimension (accounting for MRL truncation)
    pub fn output_dim(&self) -> usize {
        self.truncate_dim.unwrap_or(self.embedding_dim)
    }

    /// Get the full embedding dimension (before truncation)
    pub fn full_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Get the vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Embed a single text
    fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        // Tokenize (without adding special tokens - the model handles this)
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!("Tokenization error: {}", e))?;

        let token_ids = encoding.get_ids();

        if token_ids.is_empty() {
            // Return zero vector for empty input
            return Ok(vec![0.0; self.output_dim()]);
        }

        // Mean pool embeddings
        let dim = self.output_dim();
        let mut sum = vec![0.0f32; dim];
        let mut valid_tokens = 0usize;

        for &id in token_ids {
            let id = id as usize;
            if id < self.vocab_size {
                let emb = &self.embeddings[id];
                for i in 0..dim {
                    sum[i] += emb[i];
                }
                valid_tokens += 1;
            } else {
                tracing::warn!(
                    "Token ID {} out of vocabulary (size {})",
                    id,
                    self.vocab_size
                );
            }
        }

        if valid_tokens == 0 {
            return Ok(vec![0.0; dim]);
        }

        // Divide by count to get mean
        let n = valid_tokens as f32;
        for v in &mut sum {
            *v /= n;
        }

        Ok(sum)
    }
}

impl Embedder for StaticEmbedder {
    fn id(&self) -> &'static str {
        "static"
    }

    fn dim(&self) -> usize {
        self.output_dim()
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|text| self.embed_one(text)).collect()
    }
}

/// Parse embedding data from safetensors format
fn parse_embeddings(data: &[u8], vocab_size: usize, embedding_dim: usize) -> Result<Vec<Vec<f32>>> {
    let expected_size = vocab_size * embedding_dim * 4; // 4 bytes per f32
    if data.len() != expected_size {
        return Err(anyhow!(
            "Unexpected embedding data size: expected {} bytes, got {}",
            expected_size,
            data.len()
        ));
    }

    // Cast bytes to f32 slice
    let floats: &[f32] = bytemuck::cast_slice(data);

    // Convert to Vec<Vec<f32>>
    let embeddings: Vec<Vec<f32>> = floats
        .chunks_exact(embedding_dim)
        .map(|chunk| chunk.to_vec())
        .collect();

    if embeddings.len() != vocab_size {
        return Err(anyhow!(
            "Expected {} embeddings, got {}",
            vocab_size,
            embeddings.len()
        ));
    }

    Ok(embeddings)
}

/// Check if a model name refers to a static embedding model
pub fn is_static_model(model_name: &str) -> bool {
    let name_lower = model_name.to_lowercase();
    name_lower.contains("static-retrieval")
        || name_lower.contains("static-similarity")
        || name_lower == "static-retrieval-en"
        || name_lower == "static-multilingual"
}

/// Resolve a model alias to a HuggingFace repo ID
pub fn resolve_model_name(model_name: &str) -> &str {
    match model_name.to_lowercase().as_str() {
        "static-retrieval-en" | "static-en" | "static" => STATIC_RETRIEVAL_EN,
        "static-multilingual" | "static-multi" => STATIC_SIMILARITY_MULTILINGUAL,
        _ => model_name, // Assume it's already a full repo ID
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mrl_dimensions() {
        assert!(MRL_DIMENSIONS.contains(&32));
        assert!(MRL_DIMENSIONS.contains(&64));
        assert!(MRL_DIMENSIONS.contains(&128));
        assert!(MRL_DIMENSIONS.contains(&256));
        assert!(MRL_DIMENSIONS.contains(&512));
        assert!(MRL_DIMENSIONS.contains(&1024));
    }

    #[test]
    fn test_is_static_model() {
        assert!(is_static_model("static-retrieval-en"));
        assert!(is_static_model("static-multilingual"));
        assert!(is_static_model(
            "sentence-transformers/static-retrieval-mrl-en-v1"
        ));
        assert!(is_static_model(
            "sentence-transformers/static-similarity-mrl-multilingual-v1"
        ));
        assert!(!is_static_model("BAAI/bge-small-en-v1.5"));
        assert!(!is_static_model("nomic-embed-text-v1.5"));
    }

    #[test]
    fn test_resolve_model_name() {
        assert_eq!(
            resolve_model_name("static-retrieval-en"),
            STATIC_RETRIEVAL_EN
        );
        assert_eq!(resolve_model_name("static-en"), STATIC_RETRIEVAL_EN);
        assert_eq!(resolve_model_name("static"), STATIC_RETRIEVAL_EN);
        assert_eq!(
            resolve_model_name("static-multilingual"),
            STATIC_SIMILARITY_MULTILINGUAL
        );
        assert_eq!(
            resolve_model_name("BAAI/bge-small-en-v1.5"),
            "BAAI/bge-small-en-v1.5"
        );
    }

    #[test]
    fn test_parse_embeddings_small() {
        // Create a small test embedding: 3 tokens, 4 dimensions
        let vocab_size = 3;
        let dim = 4;
        let data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, // token 0
            5.0, 6.0, 7.0, 8.0, // token 1
            9.0, 10.0, 11.0, 12.0, // token 2
        ];
        let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();

        let embeddings = parse_embeddings(&bytes, vocab_size, dim).unwrap();

        assert_eq!(embeddings.len(), 3);
        assert_eq!(embeddings[0], vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(embeddings[1], vec![5.0, 6.0, 7.0, 8.0]);
        assert_eq!(embeddings[2], vec![9.0, 10.0, 11.0, 12.0]);
    }

    // Integration tests that require model download are marked with ignore
    // Run with: cargo test -- --ignored

    #[test]
    #[ignore = "Requires model download"]
    fn test_static_embedder_english() {
        let embedder = StaticEmbedder::english_retrieval(None).unwrap();
        assert_eq!(embedder.dim(), 1024);
        assert!(embedder.vocab_size() > 30000);
    }

    #[test]
    #[ignore = "Requires model download"]
    fn test_static_embedder_with_truncation() {
        let embedder = StaticEmbedder::english_retrieval(Some(256)).unwrap();
        assert_eq!(embedder.dim(), 256);
        assert_eq!(embedder.full_dim(), 1024);
    }

    #[test]
    #[ignore = "Requires model download"]
    fn test_embed_texts() {
        let mut embedder = StaticEmbedder::english_retrieval(Some(256)).unwrap();
        let texts = vec![
            "Hello world".to_string(),
            "The quick brown fox jumps over the lazy dog".to_string(),
        ];

        let embeddings = embedder.embed(&texts).unwrap();

        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), 256);
        assert_eq!(embeddings[1].len(), 256);

        // Embeddings should not be all zeros
        assert!(!embeddings[0].iter().all(|&x| x == 0.0));
        assert!(!embeddings[1].iter().all(|&x| x == 0.0));
    }

    #[test]
    #[ignore = "Requires model download"]
    fn test_cosine_similarity() {
        let mut embedder = StaticEmbedder::english_retrieval(Some(256)).unwrap();

        // Similar sentences
        let similar = embedder
            .embed(&[
                "The cat sits on the mat".to_string(),
                "A feline rests on the rug".to_string(),
            ])
            .unwrap();

        // Dissimilar sentences
        let dissimilar = embedder
            .embed(&[
                "The cat sits on the mat".to_string(),
                "Quantum physics explains wave-particle duality".to_string(),
            ])
            .unwrap();

        fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
            let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm_a == 0.0 || norm_b == 0.0 {
                0.0
            } else {
                dot / (norm_a * norm_b)
            }
        }

        let sim_score = cosine_similarity(&similar[0], &similar[1]);
        let dis_score = cosine_similarity(&dissimilar[0], &dissimilar[1]);

        println!("Similar score: {}", sim_score);
        println!("Dissimilar score: {}", dis_score);

        assert!(
            sim_score > dis_score,
            "Similar sentences should have higher similarity"
        );
    }
}
