# Static Embeddings Integration Plan for CK

**Date:** 2025-11-30
**Status:** ✅ IMPLEMENTED

## Implementation Summary

The following was implemented:

### New Files Created
- `ck-embed/src/static_embedder.rs` - Core StaticEmbedder implementation
- `ck-embed/src/model_downloader.rs` - HuggingFace model downloader

### Modified Files
- `Cargo.toml` (workspace) - Added tokenizers, safetensors, bytemuck, reqwest, futures-util, indicatif
- `ck-embed/Cargo.toml` - Added new dependencies
- `ck-embed/src/lib.rs` - Export static embedder, factory integration
- `ck-models/src/lib.rs` - Added static-retrieval-en and static-multilingual models
- `ck-cli/src/main.rs` - Added --truncate-dim, --list-models, --download-model flags

### New CLI Commands
```bash
ck --list-models                        # List all available models
ck --download-model static-retrieval-en # Pre-download a model
ck --index --model static-retrieval-en  # Index with static model
ck --index --model static-retrieval-en --truncate-dim 256  # With MRL truncation
```

### Available Models
- `static-retrieval-en` - English retrieval (100-400x faster, 87% quality)
- `static-multilingual` - 51-language similarity (125x faster, 92% quality)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [What Are Static Embeddings](#what-are-static-embeddings)
3. [Available Models](#available-models)
4. [Current CK Architecture](#current-ck-architecture)
5. [Implementation Design](#implementation-design)
6. [Implementation Plan](#implementation-plan)
7. [Performance Projections](#performance-projections)
8. [Risk Assessment](#risk-assessment)
9. [Open Questions](#open-questions)

---

## Executive Summary

Static embeddings offer a **100-400x speedup on CPU** (10-25x on GPU) compared to transformer models like BGE/Nomic, achieving ~87-92% of their quality. The architecture is remarkably simple:

```
Input Text → Tokenize → Embedding Lookup (Vec<Vec<f32>>) → Mean Pool → Output Vector
```

This is fundamentally different from transformer models (no attention, no neural network forward pass) - just dictionary lookup + averaging.

### Key Benefits for CK

| Metric | Current (BGE-small) | With Static Embeddings |
|--------|---------------------|------------------------|
| Embedding speed | ~270 sentences/sec | ~107,000 sentences/sec |
| Index 10k chunks | ~37 seconds | <1 second |
| Quality | Baseline | 87-92% of baseline |
| Model complexity | Transformer layers | Lookup table |

---

## What Are Static Embeddings

### Architecture Overview

Static embeddings use a revolutionary approach that fundamentally differs from transformer models:

**Three-Stage Pipeline:**
1. **Tokenization**: Text is broken into tokens using standard tokenizers (BERT-based)
2. **Embedding Lookup**: Each token is mapped to its pre-computed vector via dictionary lookup
3. **Pooling**: Token embeddings are combined (typically via mean pooling) into a single document representation

**Implementation**: Uses PyTorch's `EmbeddingBag` module, which efficiently performs embedding lookup and mean pooling in a single operation.

### Comparison with Transformer Models

| Component | Transformer (BGE/Nomic) | Static Embedding |
|-----------|------------------------|------------------|
| Tokenization | BERT/Custom | BERT (same) |
| Processing | Multi-layer attention | Embedding lookup |
| Pooling | CLS/Mean | Mean |
| Complexity | O(n²) attention | O(n) linear |
| Model Size | 50-500MB | 125-434MB (just vocab×dim) |
| Context awareness | Full contextual | None (static per token) |
| Max sequence length | 512-8192 | Unlimited |

### Why Static Embeddings Work

1. **Pre-computed Knowledge**: Token embeddings are distilled from larger models or trained with contrastive learning
2. **Averaging Effect**: Mean pooling over many tokens captures document-level semantics
3. **Training Innovation**: Multiple Negatives Ranking Loss (MNRL) with large batch sizes (2048)
4. **Matryoshka Learning**: Enables dimensionality reduction without proportional quality loss

### Matryoshka Representation Learning (MRL)

The models support dimension truncation with minimal quality loss:

| Dimensionality | Performance Retention | Use Case |
|----------------|----------------------|----------|
| 1024 (full) | 100% | Maximum quality |
| 512 | 98.5% | Good balance |
| 256 | ~96% | Memory-constrained |
| 128 | ~92% | Edge devices |
| 64 | ~83% | Extreme compression |
| 32 | ~75% | Minimal footprint |

**Key Insight:** Reducing dimensionality by 50% (1024→512) yields only ~1.47% performance loss.

---

## Available Models

### sentence-transformers/static-retrieval-mrl-en-v1

**Purpose:** Information retrieval, semantic search (English)

| Property | Value |
|----------|-------|
| Languages | English only |
| Vocabulary | 30,522 tokens (BERT uncased) |
| Dimensions | 1024 (MRL: 32-1024) |
| Model Size | 125 MB (safetensors) |
| ONNX INT8 | 31 MB |
| Speed | 397x faster than all-mpnet-base-v2 (CPU) |
| Quality | 87.4% of all-mpnet-base-v2 (NanoBEIR) |

**Benchmark Results (NanoBEIR NDCG@10):**
- QuoraRetrieval: 0.8951
- FEVER: 0.6922
- HotpotQA: 0.6547
- Mean: 0.5031

**Files Required:**
```
0_StaticEmbedding/
├── model.safetensors  (125 MB)
└── tokenizer.json     (711 KB)
```

### sentence-transformers/static-similarity-mrl-multilingual-v1

**Purpose:** Semantic similarity, paraphrase detection (51 languages)

| Property | Value |
|----------|-------|
| Languages | 51 languages |
| Vocabulary | 105,879 tokens (BERT multilingual uncased) |
| Dimensions | 1024 (MRL: 32-1024) |
| Model Size | 434 MB (safetensors) |
| ONNX INT8 | 108 MB |
| Speed | 125x faster than multilingual-e5-small (CPU) |
| Quality | 92.3% of multilingual-e5-small (STS) |

**Supported Languages:**
- Western European: en, de, fr, es, pt, nl, it, da, sv, nb, fi, ca, gl
- Eastern European: ru, pl, cs, bg, uk, hr, sk, sl, lt, lv, et, mk, sq, sr, ro, hu
- Middle Eastern/South Asian: ar, fa, he, hi, ur, gu, mr, ku, hy, ka, mn
- East/Southeast Asian: zh, ja, ko, th, vi, id, ms, my
- Other: el, tr

**Files Required:**
```
0_StaticEmbedding/
├── model.safetensors  (434 MB)
└── tokenizer.json     (2.56 MB)
```

---

## Current CK Architecture

### Relevant Components

#### ck-embed (Embedding Providers)

**Embedder Trait:**
```rust
pub trait Embedder: Send + Sync {
    fn id(&self) -> &'static str;
    fn dim(&self) -> usize;
    fn model_name(&self) -> &str;
    fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}
```

**Current Implementations:**
- `FastEmbedder` - Uses fastembed crate (ONNX runtime)
- `DummyEmbedder` - Returns zero vectors (testing)

**Supported Models (via fastembed):**
- BAAI/bge-small-en-v1.5 (384d, default)
- sentence-transformers/all-MiniLM-L6-v2 (384d)
- nomic-embed-text-v1.5 (768d)
- jina-embeddings-v2-base-code (768d)
- BAAI/bge-base-en-v1.5 (768d)
- BAAI/bge-large-en-v1.5 (1024d)

#### ck-models (Model Registry)

```rust
pub struct ModelConfig {
    pub name: String,
    pub provider: String,      // "fastembed" currently
    pub dimensions: usize,
    pub max_tokens: usize,
    pub description: String,
}

pub struct ModelRegistry {
    pub models: HashMap<String, ModelConfig>,
    pub default_model: String,
}
```

#### ck-index (Index Storage)

```rust
pub struct IndexManifest {
    version: String,
    created: u64,
    updated: u64,
    files: HashMap<PathBuf, FileMetadata>,
    embedding_model: Option<String>,
    embedding_dimensions: Option<usize>,
    chunk_hash_version: Option<u32>,
}
```

### Key Dependencies

From workspace `Cargo.toml`:
- `fastembed = "5.1"` - Current embedding provider
- `tokenizers` - Already used indirectly
- `tantivy = "0.24"` - Lexical search
- `tree-sitter` - Code parsing

---

## Implementation Design

### Recommended Approach: Pure Rust

Given the simplicity of static embeddings (just lookup + mean), a pure Rust implementation is recommended:

```rust
// ck-embed/src/static_embedder.rs

use tokenizers::Tokenizer;
use safetensors::SafeTensors;

pub struct StaticEmbedder {
    tokenizer: Tokenizer,           // From tokenizers crate
    embeddings: Vec<Vec<f32>>,      // vocab_size × embedding_dim
    vocab_size: usize,
    embedding_dim: usize,
    truncate_dim: Option<usize>,    // MRL support
    model_name: String,
}

impl StaticEmbedder {
    pub fn new(model_path: &Path, truncate_dim: Option<usize>) -> Result<Self> {
        // 1. Load tokenizer.json
        let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))?;

        // 2. Load model.safetensors embeddings
        let weights_data = std::fs::read(model_path.join("model.safetensors"))?;
        let tensors = SafeTensors::deserialize(&weights_data)?;
        let embedding_tensor = tensors.tensor("embedding.weight")?;

        // 3. Parse to Vec<Vec<f32>> [vocab_size, embedding_dim]
        let shape = embedding_tensor.shape();  // e.g. [30522, 1024]
        let data: Vec<f32> = bytemuck::cast_slice(embedding_tensor.data()).to_vec();
        let embeddings = data.chunks(shape[1])
            .map(|chunk| chunk.to_vec())
            .collect();

        Ok(Self {
            tokenizer,
            embeddings,
            vocab_size: shape[0],
            embedding_dim: shape[1],
            truncate_dim,
            model_name: model_name.to_string(),
        })
    }

    fn output_dim(&self) -> usize {
        self.truncate_dim.unwrap_or(self.embedding_dim)
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
        texts.iter().map(|text| {
            // 1. Tokenize (no padding needed)
            let encoding = self.tokenizer.encode(text, false)
                .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;
            let token_ids = encoding.get_ids();

            if token_ids.is_empty() {
                return Ok(vec![0.0; self.output_dim()]);
            }

            // 2. Mean pool embeddings
            let dim = self.output_dim();
            let mut sum = vec![0.0f32; dim];
            for &id in token_ids {
                if (id as usize) < self.vocab_size {
                    let emb = &self.embeddings[id as usize];
                    for i in 0..dim {
                        sum[i] += emb[i];
                    }
                }
            }
            let n = token_ids.len() as f32;
            for v in &mut sum {
                *v /= n;
            }

            Ok(sum)
        }).collect()
    }
}
```

### Why Pure Rust Over ONNX

| Aspect | Pure Rust | ONNX Runtime |
|--------|-----------|--------------|
| Dependencies | +2 crates (safetensors, tokenizers) | +ort (~20MB binary) |
| Complexity | Simple (lookup + mean) | Session management |
| Speed | Fastest possible | Slight overhead |
| Portability | Any Rust target | Platform-specific |
| Model format | safetensors (native) | Requires ONNX export |

### New Dependencies

Add to workspace `Cargo.toml`:
```toml
tokenizers = "0.22"       # Tokenizer loading
safetensors = "0.5"       # Model weight loading
bytemuck = "1.14"         # Safe f32 slice casting
```

### Model Registry Updates

```rust
// ck-models/src/lib.rs

models.insert(
    "static-retrieval-en".to_string(),
    ModelConfig {
        name: "sentence-transformers/static-retrieval-mrl-en-v1".to_string(),
        provider: "static".to_string(),
        dimensions: 1024,
        max_tokens: usize::MAX,  // No limit!
        description: "Ultra-fast English retrieval (100-400x faster, 87% quality)".to_string(),
    },
);

models.insert(
    "static-multilingual".to_string(),
    ModelConfig {
        name: "sentence-transformers/static-similarity-mrl-multilingual-v1".to_string(),
        provider: "static".to_string(),
        dimensions: 1024,
        max_tokens: usize::MAX,
        description: "Ultra-fast 51-language similarity (125x faster, 92% quality)".to_string(),
    },
);
```

### CLI Integration

New flags:
```bash
# Use static embedding model
ck --index --model static-retrieval-en

# With MRL dimension truncation (faster + smaller index)
ck --index --model static-retrieval-en --truncate-dim 256

# Download model explicitly
ck --download-model static-retrieval-en

# List available models with speed/quality tradeoffs
ck --list-models
```

### Model Download Strategy

```rust
// ck-embed/src/model_downloader.rs

pub struct HuggingFaceDownloader;

impl HuggingFaceDownloader {
    pub fn download_static_model(repo_id: &str, cache_dir: &Path) -> Result<PathBuf> {
        // Required files for static embedding
        let files = [
            "0_StaticEmbedding/model.safetensors",
            "0_StaticEmbedding/tokenizer.json",
        ];

        for file in &files {
            let url = format!(
                "https://huggingface.co/{}/resolve/main/{}",
                repo_id, file
            );
            download_file_with_progress(&url, &cache_dir.join(file))?;
        }

        Ok(cache_dir.join(repo_id.replace("/", "_")))
    }
}
```

**Cache locations:**
- Linux/macOS: `~/.cache/ck/models/`
- Windows: `%LOCALAPPDATA%/ck/cache/models/`

### Embedder Factory Update

```rust
// ck-embed/src/lib.rs

pub fn create_embedder_with_progress(
    model_name: Option<&str>,
    truncate_dim: Option<usize>,
    progress_callback: Option<ModelDownloadCallback>,
) -> Result<Box<dyn Embedder>> {
    let model = model_name.unwrap_or("BAAI/bge-small-en-v1.5");

    // Check if this is a static model
    if model.starts_with("static-") || model.contains("static-retrieval") || model.contains("static-similarity") {
        let model_path = ensure_model_downloaded(model, progress_callback)?;
        return Ok(Box::new(StaticEmbedder::new(&model_path, truncate_dim)?));
    }

    // Fall back to fastembed for transformer models
    #[cfg(feature = "fastembed")]
    {
        Ok(Box::new(FastEmbedder::new_with_progress(model, progress_callback)?))
    }

    #[cfg(not(feature = "fastembed"))]
    {
        Ok(Box::new(DummyEmbedder::new_with_model(model)))
    }
}
```

---

## Implementation Plan

### Phase 1: Core Static Embedder (2-3 days)

**Tasks:**
1. Add new dependencies to workspace Cargo.toml
2. Create `ck-embed/src/static_embedder.rs` with core implementation
3. Implement safetensors loading and parsing
4. Implement tokenization with `tokenizers` crate
5. Implement mean pooling
6. Add MRL truncation support
7. Unit tests for basic functionality

**Files to create/modify:**
- `Cargo.toml` (workspace) - Add dependencies
- `ck-embed/Cargo.toml` - Add dependencies
- `ck-embed/src/static_embedder.rs` - NEW
- `ck-embed/src/lib.rs` - Export and factory update

### Phase 2: Model Registry Integration (1 day)

**Tasks:**
1. Add static model entries to ModelRegistry
2. Add `provider` field validation
3. Add `mrl_dims` optional field for MRL-capable models
4. Update model listing/display

**Files to modify:**
- `ck-models/src/lib.rs`

### Phase 3: Model Downloader (1 day)

**Tasks:**
1. Create HuggingFace model downloader
2. Implement progress reporting
3. Add retry logic for robustness
4. Integrate with embedder factory

**Files to create:**
- `ck-embed/src/model_downloader.rs` - NEW

### Phase 4: CLI Integration (1 day)

**Tasks:**
1. Add `--truncate-dim` flag
2. Add `--download-model` command
3. Add `--list-models` command
4. Update `--model` to accept static models
5. Update help text with speed/quality info

**Files to modify:**
- `ck-cli/src/main.rs`

### Phase 5: Testing & Benchmarks (1 day)

**Tasks:**
1. Integration tests with real models
2. Quality comparison benchmarks
3. Speed benchmarks
4. Index size comparisons
5. Documentation updates

**Files to create/modify:**
- `ck-embed/src/static_embedder.rs` - Tests
- `ck-cli/tests/` - Integration tests
- `README.md` - Documentation

### Timeline Summary

| Phase | Duration | Description |
|-------|----------|-------------|
| 1 | 2-3 days | Core StaticEmbedder implementation |
| 2 | 1 day | Model registry integration |
| 3 | 1 day | HuggingFace model downloader |
| 4 | 1 day | CLI flags and commands |
| 5 | 1 day | Testing and benchmarks |
| **Total** | **6-8 days** | Full implementation |

---

## Performance Projections

### Embedding Speed

| Model | CPU Speed | GPU Speed | Speedup vs BGE |
|-------|-----------|-----------|----------------|
| BGE-small (current) | ~270 sent/sec | ~5,000 sent/sec | 1x |
| Static-retrieval-en | ~107,000 sent/sec | ~200,000 sent/sec | 400x |

### Indexing Time

| Codebase Size | Current (BGE) | Static (1024d) | Static (256d) |
|---------------|---------------|----------------|---------------|
| 1k chunks | ~4 sec | <0.1 sec | <0.1 sec |
| 10k chunks | ~37 sec | <0.5 sec | <0.5 sec |
| 100k chunks | ~6 min | <5 sec | <5 sec |

### Index Size

| Model | Dimensions | Size per 10k chunks |
|-------|------------|---------------------|
| BGE-small | 384 | ~15 MB |
| Static (full) | 1024 | ~40 MB |
| Static (MRL 512) | 512 | ~20 MB |
| Static (MRL 256) | 256 | ~10 MB |
| Static (MRL 128) | 128 | ~5 MB |

### Quality (Relative to Transformer Baseline)

| Model | Retrieval Quality | Notes |
|-------|-------------------|-------|
| BGE-small | 100% (baseline) | Current default |
| Static-retrieval-en | 87% | Acceptable for most use cases |
| Static-multilingual | 92% | Better on similarity tasks |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Quality regression for edge cases | Medium | Medium | Benchmark extensively, document limitations |
| Large index sizes (1024d default) | Low | Low | Recommend MRL 256d as default |
| Model download failures | Low | Medium | Robust retry logic, progress indicators |
| Tokenizer edge cases (Unicode, etc.) | Low | Low | Use battle-tested `tokenizers` crate |
| safetensors parsing issues | Very Low | Medium | Well-maintained crate, test thoroughly |
| Breaking change for existing indexes | Medium | Medium | Clear migration path, version check |

---

## Open Questions

### 1. Default Model Strategy

**Question:** Should static embeddings become the new default?

**Options:**
- A) Keep BGE-small as default (quality-first, backward compatible)
- B) Make static-retrieval-en the default (speed-first)
- C) Auto-select based on codebase size (>10k files → static)

**Recommendation:** Option A initially, with clear documentation of speed benefits.

### 2. Default MRL Dimension

**Question:** What should the default dimension be for static models?

**Options:**
- A) Full 1024d (maximum quality)
- B) 512d (good balance: 98.5% quality, 2x smaller)
- C) 256d (great balance: 96% quality, 4x smaller)

**Recommendation:** Option C (256d) - best speed/quality/size tradeoff.

### 3. Multilingual Model Inclusion

**Question:** Should the multilingual model be included in initial release?

**Recommendation:** Yes, but as opt-in (`--model static-multilingual`).

### 4. Fallback Behavior

**Question:** If static model download fails, should we fall back to fastembed?

**Recommendation:** No automatic fallback. Fail clearly with instructions.

### 5. Index Migration

**Question:** How to handle users switching from 384d to 1024d models?

**Recommendation:**
- Detect dimension mismatch in manifest
- Require explicit `--reindex` flag
- Clear error message explaining the situation

---

## References

- [HuggingFace Blog: Static Embeddings](https://huggingface.co/blog/static-embeddings)
- [static-retrieval-mrl-en-v1 Model Card](https://huggingface.co/sentence-transformers/static-retrieval-mrl-en-v1)
- [static-similarity-mrl-multilingual-v1 Model Card](https://huggingface.co/sentence-transformers/static-similarity-mrl-multilingual-v1)
- [Matryoshka Representation Learning](https://huggingface.co/blog/matryoshka)
- [safetensors Crate](https://crates.io/crates/safetensors)
- [tokenizers Crate](https://crates.io/crates/tokenizers)
