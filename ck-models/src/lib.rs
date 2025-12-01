use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub provider: String,
    pub dimensions: usize,
    pub max_tokens: usize,
    pub description: String,
    /// Supported MRL dimensions for static models (optional)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mrl_dims: Option<Vec<usize>>,
    /// Recommended default threshold for semantic search (model-specific)
    /// - Transformer models (BGE, etc): ~0.6 (scores compressed to 0.6-1.0)
    /// - Static models: ~0.4 (wider score distribution 0.3-0.9)
    #[serde(default = "default_threshold")]
    pub default_threshold: f32,
}

fn default_threshold() -> f32 {
    0.6 // Default for backward compatibility
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRegistry {
    pub models: HashMap<String, ModelConfig>,
    pub default_model: String,
}

impl Default for ModelRegistry {
    fn default() -> Self {
        let mut models = HashMap::new();

        // === Transformer-based models (via fastembed) ===
        // These models use contrastive training which compresses scores to 0.6-1.0 range
        // Threshold of 0.6 works well for these models

        models.insert(
            "bge-small".to_string(),
            ModelConfig {
                name: "BAAI/bge-small-en-v1.5".to_string(),
                provider: "fastembed".to_string(),
                dimensions: 384,
                max_tokens: 512,
                description: "Small, fast English embedding model".to_string(),
                mrl_dims: None,
                default_threshold: 0.6,
            },
        );

        models.insert(
            "minilm".to_string(),
            ModelConfig {
                name: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
                provider: "fastembed".to_string(),
                dimensions: 384,
                max_tokens: 256,
                description: "Lightweight English embedding model".to_string(),
                mrl_dims: None,
                default_threshold: 0.6,
            },
        );

        models.insert(
            "nomic-v1.5".to_string(),
            ModelConfig {
                name: "nomic-embed-text-v1.5".to_string(),
                provider: "fastembed".to_string(),
                dimensions: 768,
                max_tokens: 8192,
                description: "High-quality English embedding model with large context window"
                    .to_string(),
                mrl_dims: None,
                default_threshold: 0.6,
            },
        );

        models.insert(
            "jina-code".to_string(),
            ModelConfig {
                name: "jina-embeddings-v2-base-code".to_string(),
                provider: "fastembed".to_string(),
                dimensions: 768,
                max_tokens: 8192,
                description: "Code-specific embedding model optimized for programming tasks"
                    .to_string(),
                mrl_dims: None,
                default_threshold: 0.6,
            },
        );

        // === Static embedding models (100-400x faster) ===
        // These use standard cosine similarity with wider score distribution (0.3-0.9)
        // Lower threshold of 0.4 needed for single-word and short queries

        models.insert(
            "static-retrieval-en".to_string(),
            ModelConfig {
                name: "sentence-transformers/static-retrieval-mrl-en-v1".to_string(),
                provider: "static".to_string(),
                dimensions: 1024,
                max_tokens: usize::MAX, // No limit for static embeddings
                description: "Ultra-fast English retrieval (100-400x faster, 87% quality)"
                    .to_string(),
                mrl_dims: Some(vec![32, 64, 128, 256, 512, 1024]),
                default_threshold: 0.4,
            },
        );

        models.insert(
            "static-multilingual".to_string(),
            ModelConfig {
                name: "sentence-transformers/static-similarity-mrl-multilingual-v1".to_string(),
                provider: "static".to_string(),
                dimensions: 1024,
                max_tokens: usize::MAX,
                description: "Ultra-fast 51-language similarity (125x faster, 92% quality)"
                    .to_string(),
                mrl_dims: Some(vec![32, 64, 128, 256, 512, 1024]),
                default_threshold: 0.3, // Similarity model needs even lower threshold
            },
        );

        Self {
            models,
            default_model: "bge-small".to_string(), // Keep BGE as default for backward compatibility
        }
    }
}

impl ModelRegistry {
    pub fn load(path: &Path) -> Result<Self> {
        if path.exists() {
            let data = std::fs::read_to_string(path)?;
            Ok(serde_json::from_str(&data)?)
        } else {
            Ok(Self::default())
        }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let data = serde_json::to_string_pretty(self)?;
        std::fs::write(path, data)?;
        Ok(())
    }

    pub fn get_model(&self, name: &str) -> Option<&ModelConfig> {
        self.models.get(name)
    }

    pub fn get_default_model(&self) -> Option<&ModelConfig> {
        self.models.get(&self.default_model)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectConfig {
    pub model: String,
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub index_backend: String,
}

impl Default for ProjectConfig {
    fn default() -> Self {
        Self {
            model: "bge-small".to_string(),
            chunk_size: 512,
            chunk_overlap: 128,
            index_backend: "hnsw".to_string(),
        }
    }
}

impl ProjectConfig {
    pub fn load(path: &Path) -> Result<Self> {
        if path.exists() {
            let data = std::fs::read_to_string(path)?;
            Ok(serde_json::from_str(&data)?)
        } else {
            Ok(Self::default())
        }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let data = serde_json::to_string_pretty(self)?;
        std::fs::write(path, data)?;
        Ok(())
    }
}
