//! Ensemble search: query multiple embedding models in parallel and merge results using RRF.

use anyhow::Result;
use ck_core::{SearchOptions, SearchResult, SearchResults};
use std::collections::HashMap;
use std::path::PathBuf;

use crate::semantic_search_v3;

/// Reciprocal Rank Fusion constant (standard value from literature)
const RRF_K: f32 = 60.0;

/// Search using all indexed embedding models and merge results with RRF.
pub async fn ensemble_search(options: &SearchOptions) -> Result<SearchResults> {
    ensemble_search_with_progress(options, None).await
}

/// Search using all indexed embedding models with progress callback.
pub async fn ensemble_search_with_progress(
    options: &SearchOptions,
    progress_callback: Option<crate::SearchProgressCallback>,
) -> Result<SearchResults> {
    // Find the index root and load manifest to get available models
    let index_root = crate::find_nearest_index_root(&options.path).unwrap_or_else(|| {
        if options.path.is_file() {
            options.path.parent().unwrap_or(&options.path).to_path_buf()
        } else {
            options.path.clone()
        }
    });

    let manifest_path = index_root.join(".ck").join("manifest.json");
    if !manifest_path.exists() {
        return Err(ck_core::CkError::Index(
            "No index found. Run 'ck --index' first.".to_string(),
        )
        .into());
    }

    let manifest_data = std::fs::read(&manifest_path)?;
    let manifest: ck_index::IndexManifest = serde_json::from_slice(&manifest_data)?;
    let models = manifest.get_indexed_models();

    if models.is_empty() {
        return Err(ck_core::CkError::Index(
            "No embedding models found in index. Run 'ck --index --model <model>' first."
                .to_string(),
        )
        .into());
    }

    if let Some(ref callback) = progress_callback {
        callback(&format!(
            "Ensemble search across {} models: {}",
            models.len(),
            models.join(", ")
        ));
    }

    // Collect results from each model
    // We run sequentially to avoid loading multiple large models simultaneously
    // (each model needs to be loaded to embed the query)
    let mut results_per_model: Vec<(String, Vec<SearchResult>)> = Vec::new();

    for model_name in &models {
        if let Some(ref callback) = progress_callback {
            callback(&format!("Searching with model: {}", model_name));
        }

        // Create options for this specific model
        let mut model_options = options.clone();
        model_options.embedding_model = Some(model_name.clone());

        // Disable threshold for individual model searches - we'll apply RRF ranking
        // But keep a reasonable top_k to limit results per model
        let per_model_top_k = options.top_k.unwrap_or(20).max(20);
        model_options.top_k = Some(per_model_top_k);
        model_options.threshold = None; // Don't filter by threshold per-model

        match semantic_search_v3(&model_options).await {
            Ok(search_results) => {
                if !search_results.matches.is_empty() {
                    results_per_model.push((model_name.clone(), search_results.matches));
                }
            }
            Err(e) => {
                tracing::warn!("Search with model {} failed: {}", model_name, e);
                // Continue with other models
            }
        }
    }

    if results_per_model.is_empty() {
        return Ok(SearchResults {
            matches: Vec::new(),
            closest_below_threshold: None,
        });
    }

    if let Some(ref callback) = progress_callback {
        callback("Merging results with Reciprocal Rank Fusion...");
    }

    // Merge results using RRF
    let merged = merge_with_rrf(results_per_model, options.top_k);

    Ok(SearchResults {
        matches: merged,
        closest_below_threshold: None,
    })
}

/// Merge results from multiple models using Reciprocal Rank Fusion (RRF).
///
/// RRF score = Σ 1/(k + rank_i) for each model where the document appears
///
/// This handles different score scales between models and naturally boosts
/// documents that appear in multiple models' results.
fn merge_with_rrf(
    results_per_model: Vec<(String, Vec<SearchResult>)>,
    top_k: Option<usize>,
) -> Vec<SearchResult> {
    // Key: file path (we merge at file level since chunk boundaries differ per model)
    // Value: (RRF score, best SearchResult for this file, contributing models)
    let mut merged: HashMap<PathBuf, (f32, SearchResult, Vec<String>)> = HashMap::new();

    for (model_name, results) in results_per_model {
        for (rank, result) in results.iter().enumerate() {
            let rrf_contribution = 1.0 / (RRF_K + rank as f32 + 1.0);

            merged
                .entry(result.file.clone())
                .and_modify(|(score, existing, models)| {
                    *score += rrf_contribution;
                    models.push(model_name.clone());
                    // Keep the result with higher original score (better preview/span)
                    if result.score > existing.score {
                        *existing = result.clone();
                    }
                })
                .or_insert((rrf_contribution, result.clone(), vec![model_name.clone()]));
        }
    }

    // Sort by RRF score descending
    let mut sorted: Vec<_> = merged.into_values().collect();
    sorted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Apply top_k and convert to final results
    let limit = top_k.unwrap_or(sorted.len());
    sorted
        .into_iter()
        .take(limit)
        .map(|(rrf_score, mut result, _models)| {
            // Store RRF score (normalized to 0-1 range for display)
            // Max possible RRF for a single result is ~num_models * 0.016
            result.score = rrf_score;
            result
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ck_core::Span;

    fn make_result(file: &str, score: f32, line: usize) -> SearchResult {
        SearchResult {
            file: PathBuf::from(file),
            span: Span {
                start_line: line,
                end_line: line + 10,
                start_col: 0,
                end_col: 0,
            },
            score,
            preview: format!("Preview for {}", file),
            lang: None,
            symbol: None,
            chunk_hash: None,
            index_epoch: None,
        }
    }

    #[test]
    fn test_rrf_single_model() {
        let results = vec![(
            "model-a".to_string(),
            vec![
                make_result("file1.rs", 0.9, 1),
                make_result("file2.rs", 0.8, 1),
            ],
        )];

        let merged = merge_with_rrf(results, Some(10));

        assert_eq!(merged.len(), 2);
        // First result should have higher RRF score
        assert!(merged[0].score > merged[1].score);
        assert_eq!(merged[0].file, PathBuf::from("file1.rs"));
    }

    #[test]
    fn test_rrf_boosts_multi_model_matches() {
        // file1 appears in both models, file2 only in model-a, file3 only in model-b
        let results = vec![
            (
                "model-a".to_string(),
                vec![
                    make_result("file2.rs", 0.95, 1), // Rank 0 in model-a
                    make_result("file1.rs", 0.85, 1), // Rank 1 in model-a
                ],
            ),
            (
                "model-b".to_string(),
                vec![
                    make_result("file3.rs", 0.92, 1), // Rank 0 in model-b
                    make_result("file1.rs", 0.82, 1), // Rank 1 in model-b
                ],
            ),
        ];

        let merged = merge_with_rrf(results, Some(10));

        // file1 should be ranked highest because it appears in both models
        // RRF(file1) = 1/(60+2) + 1/(60+2) = 2 * 0.0161 = 0.0323
        // RRF(file2) = 1/(60+1) = 0.0164
        // RRF(file3) = 1/(60+1) = 0.0164
        assert_eq!(merged[0].file, PathBuf::from("file1.rs"));
    }
}
