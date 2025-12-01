//! Trigram index for accelerating regex searches.
//!
//! Trigram indexing extracts all 3-byte sequences from documents and builds
//! an inverted index. When searching with a regex pattern, we extract literal
//! substrings from the pattern, convert them to trigrams, and use the index
//! to find candidate files that contain all required trigrams. This can
//! reduce the number of files to scan by 10-100x.
//!
//! Based on the approach described by Russ Cox:
//! https://swtch.com/~rsc/regexp/regexp4.html

use anyhow::{Context, Result};
use rayon::prelude::*;
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

/// A trigram is exactly 3 bytes.
/// We work on bytes (not chars) to handle all encodings uniformly.
pub type Trigram = [u8; 3];

/// Convert a trigram to a u32 for use as a map key.
#[inline]
fn trigram_to_u32(t: Trigram) -> u32 {
    ((t[0] as u32) << 16) | ((t[1] as u32) << 8) | (t[2] as u32)
}

/// Convert a u32 back to a trigram.
#[inline]
#[allow(dead_code)]
fn u32_to_trigram(n: u32) -> Trigram {
    [
        ((n >> 16) & 0xFF) as u8,
        ((n >> 8) & 0xFF) as u8,
        (n & 0xFF) as u8,
    ]
}

/// Extract all trigrams from a byte slice.
pub fn extract_trigrams(content: &[u8]) -> HashSet<Trigram> {
    if content.len() < 3 {
        return HashSet::new();
    }

    let mut trigrams = HashSet::with_capacity(content.len().saturating_sub(2));
    for window in content.windows(3) {
        trigrams.insert([window[0], window[1], window[2]]);
    }
    trigrams
}

/// Extract trigrams with case normalization (for case-insensitive search).
pub fn extract_trigrams_lowercase(content: &[u8]) -> HashSet<Trigram> {
    if content.len() < 3 {
        return HashSet::new();
    }

    let mut trigrams = HashSet::with_capacity(content.len().saturating_sub(2));
    for window in content.windows(3) {
        trigrams.insert([
            window[0].to_ascii_lowercase(),
            window[1].to_ascii_lowercase(),
            window[2].to_ascii_lowercase(),
        ]);
    }
    trigrams
}

/// Extract literal sequences from a regex pattern using regex-syntax.
/// Returns None if no usable literals can be extracted.
///
/// We try both prefix and suffix extraction to handle patterns like "Error.*Handler"
/// where literals appear on both sides of a wildcard.
pub fn extract_literals_from_pattern(pattern: &str) -> Option<Vec<Vec<u8>>> {
    use regex_syntax::Parser;
    use regex_syntax::hir::literal::{ExtractKind, Extractor};

    let hir = Parser::new().parse(pattern).ok()?;

    let mut all_literals = Vec::new();

    // Extract prefix literals
    let prefix_extractor = Extractor::new();
    let prefix_seq = prefix_extractor.extract(&hir);
    if let Some(lits) = prefix_seq.literals() {
        for lit in lits {
            if lit.as_bytes().len() >= 3 {
                all_literals.push(lit.as_bytes().to_vec());
            }
        }
    }

    // Extract suffix literals
    let mut suffix_extractor = Extractor::new();
    suffix_extractor.kind(ExtractKind::Suffix);
    let suffix_seq = suffix_extractor.extract(&hir);
    if let Some(lits) = suffix_seq.literals() {
        for lit in lits {
            if lit.as_bytes().len() >= 3 {
                let bytes = lit.as_bytes().to_vec();
                // Avoid duplicates
                if !all_literals.contains(&bytes) {
                    all_literals.push(bytes);
                }
            }
        }
    }

    if all_literals.is_empty() {
        None
    } else {
        Some(all_literals)
    }
}

/// Extract trigrams from a regex pattern by first extracting literals.
/// Returns None if no usable trigrams can be extracted (fall back to full scan).
pub fn extract_pattern_trigrams(pattern: &str) -> Option<HashSet<Trigram>> {
    let literals = extract_literals_from_pattern(pattern)?;

    let mut all_trigrams = HashSet::new();
    for literal in &literals {
        let trigrams = extract_trigrams(literal);
        all_trigrams.extend(trigrams);
    }

    if all_trigrams.is_empty() {
        None
    } else {
        Some(all_trigrams)
    }
}

/// Extract trigrams for case-insensitive pattern matching.
pub fn extract_pattern_trigrams_lowercase(pattern: &str) -> Option<HashSet<Trigram>> {
    let literals = extract_literals_from_pattern(pattern)?;

    let mut all_trigrams = HashSet::new();
    for literal in &literals {
        let trigrams = extract_trigrams_lowercase(literal);
        all_trigrams.extend(trigrams);
    }

    if all_trigrams.is_empty() {
        None
    } else {
        Some(all_trigrams)
    }
}

/// Serializable index data structure.
#[derive(Serialize, Deserialize)]
struct TrigramIndexData {
    version: u32,
    /// List of indexed file paths (index = file ID).
    files: Vec<PathBuf>,
    /// Posting lists: trigram (as u32) -> serialized RoaringBitmap.
    postings: Vec<(u32, Vec<u8>)>,
    /// Whether the index was built with case normalization.
    case_insensitive: bool,
}

const INDEX_VERSION: u32 = 1;

/// Trigram index for fast regex search filtering.
pub struct TrigramIndex {
    /// Maps trigram (as u32) to bitmap of file IDs.
    posting_lists: HashMap<u32, RoaringBitmap>,
    /// File ID -> path mapping.
    files: Vec<PathBuf>,
    /// Path -> file ID for quick lookups.
    file_ids: HashMap<PathBuf, u32>,
    /// Whether the index uses case normalization.
    case_insensitive: bool,
}

impl Default for TrigramIndex {
    fn default() -> Self {
        Self::new(false)
    }
}

impl TrigramIndex {
    /// Create a new empty trigram index.
    pub fn new(case_insensitive: bool) -> Self {
        Self {
            posting_lists: HashMap::new(),
            files: Vec::new(),
            file_ids: HashMap::new(),
            case_insensitive,
        }
    }

    /// Returns whether the index is case-insensitive.
    pub fn is_case_insensitive(&self) -> bool {
        self.case_insensitive
    }

    /// Returns the number of indexed files.
    pub fn file_count(&self) -> usize {
        self.files.len()
    }

    /// Returns the number of unique trigrams.
    pub fn trigram_count(&self) -> usize {
        self.posting_lists.len()
    }

    /// Add a file to the index.
    pub fn add_file(&mut self, path: &Path, content: &[u8]) {
        // Get or create file ID
        let file_id = if let Some(&id) = self.file_ids.get(path) {
            // File already exists, clear its old trigrams first
            self.remove_file_id(id);
            id
        } else {
            let id = self.files.len() as u32;
            self.files.push(path.to_path_buf());
            self.file_ids.insert(path.to_path_buf(), id);
            id
        };

        // Extract trigrams
        let trigrams = if self.case_insensitive {
            extract_trigrams_lowercase(content)
        } else {
            extract_trigrams(content)
        };

        // Add to posting lists
        for trigram in trigrams {
            let key = trigram_to_u32(trigram);
            self.posting_lists.entry(key).or_default().insert(file_id);
        }
    }

    /// Remove a file from the index by path.
    pub fn remove_file(&mut self, path: &Path) {
        if let Some(&file_id) = self.file_ids.get(path) {
            self.remove_file_id(file_id);
            // Note: We don't remove from files/file_ids to keep IDs stable.
            // The file ID will just have empty posting lists.
        }
    }

    /// Remove a file from all posting lists by ID.
    fn remove_file_id(&mut self, file_id: u32) {
        for bitmap in self.posting_lists.values_mut() {
            bitmap.remove(file_id);
        }
    }

    /// Query the index for files matching the given trigrams.
    /// Returns files that contain ALL of the trigrams.
    pub fn query(&self, trigrams: &HashSet<Trigram>) -> Vec<&Path> {
        if trigrams.is_empty() {
            // No trigrams = return all files
            return self.files.iter().map(|p| p.as_path()).collect();
        }

        // Find the posting list for each trigram
        let mut trigram_keys: Vec<u32> = trigrams.iter().map(|t| trigram_to_u32(*t)).collect();

        // Sort by posting list size (smallest first) for efficient intersection
        trigram_keys.sort_by_key(|key| self.posting_lists.get(key).map(|b| b.len()).unwrap_or(0));

        // Start with the smallest posting list
        let mut result = match trigram_keys.first().and_then(|k| self.posting_lists.get(k)) {
            Some(bitmap) => bitmap.clone(),
            None => return Vec::new(), // Trigram not in index = no matches
        };

        // Intersect with remaining posting lists
        for key in trigram_keys.iter().skip(1) {
            match self.posting_lists.get(key) {
                Some(bitmap) => {
                    result &= bitmap;
                    if result.is_empty() {
                        return Vec::new();
                    }
                }
                None => return Vec::new(), // Trigram not in index = no matches
            }
        }

        // Convert file IDs to paths
        result
            .iter()
            .filter_map(|id| self.files.get(id as usize).map(|p| p.as_path()))
            .collect()
    }

    /// Query the index using a regex pattern.
    /// Returns None if no trigrams could be extracted (caller should fall back to full scan).
    pub fn query_pattern(&self, pattern: &str) -> Option<Vec<&Path>> {
        let trigrams = if self.case_insensitive {
            extract_pattern_trigrams_lowercase(pattern)?
        } else {
            extract_pattern_trigrams(pattern)?
        };

        Some(self.query(&trigrams))
    }

    /// Build a trigram index from files in parallel.
    pub fn build_from_files<I>(files: I, case_insensitive: bool) -> Result<Self>
    where
        I: IntoIterator<Item = (PathBuf, Vec<u8>)>,
    {
        let files_vec: Vec<_> = files.into_iter().collect();
        let file_count = files_vec.len();

        // Extract trigrams in parallel
        let file_trigrams: Vec<(PathBuf, HashSet<Trigram>)> = files_vec
            .into_par_iter()
            .map(|(path, content)| {
                let trigrams = if case_insensitive {
                    extract_trigrams_lowercase(&content)
                } else {
                    extract_trigrams(&content)
                };
                (path, trigrams)
            })
            .collect();

        // Build the index
        let mut index = Self::new(case_insensitive);
        index.files.reserve(file_count);
        index.file_ids.reserve(file_count);

        for (path, trigrams) in file_trigrams {
            let file_id = index.files.len() as u32;
            index.files.push(path.clone());
            index.file_ids.insert(path, file_id);

            for trigram in trigrams {
                let key = trigram_to_u32(trigram);
                index.posting_lists.entry(key).or_default().insert(file_id);
            }
        }

        Ok(index)
    }

    /// Save the index to a file.
    pub fn save(&self, path: &Path) -> Result<()> {
        let postings: Vec<(u32, Vec<u8>)> = self
            .posting_lists
            .iter()
            .map(|(&key, bitmap)| {
                let mut bytes = Vec::new();
                bitmap.serialize_into(&mut bytes).unwrap();
                (key, bytes)
            })
            .collect();

        let data = TrigramIndexData {
            version: INDEX_VERSION,
            files: self.files.clone(),
            postings,
            case_insensitive: self.case_insensitive,
        };

        let encoded = bincode::serialize(&data).context("Failed to serialize trigram index")?;

        // Write atomically via temp file
        let temp_path = path.with_extension("tmp");
        std::fs::write(&temp_path, &encoded).context("Failed to write trigram index")?;
        std::fs::rename(&temp_path, path).context("Failed to rename trigram index")?;

        tracing::info!(
            "Saved trigram index: {} files, {} unique trigrams, {} bytes",
            self.files.len(),
            self.posting_lists.len(),
            encoded.len()
        );

        Ok(())
    }

    /// Load an index from a file.
    pub fn load(path: &Path) -> Result<Self> {
        let data = std::fs::read(path).context("Failed to read trigram index")?;
        let index_data: TrigramIndexData =
            bincode::deserialize(&data).context("Failed to deserialize trigram index")?;

        if index_data.version != INDEX_VERSION {
            anyhow::bail!(
                "Trigram index version mismatch: expected {}, got {}",
                INDEX_VERSION,
                index_data.version
            );
        }

        let mut posting_lists = HashMap::with_capacity(index_data.postings.len());
        for (key, bytes) in index_data.postings {
            let bitmap = RoaringBitmap::deserialize_from(&bytes[..])
                .context("Failed to deserialize posting list")?;
            posting_lists.insert(key, bitmap);
        }

        let file_ids: HashMap<PathBuf, u32> = index_data
            .files
            .iter()
            .enumerate()
            .map(|(id, path)| (path.clone(), id as u32))
            .collect();

        Ok(Self {
            posting_lists,
            files: index_data.files,
            file_ids,
            case_insensitive: index_data.case_insensitive,
        })
    }

    /// Load an index if it exists, otherwise return a new empty index.
    pub fn load_or_new(path: &Path, case_insensitive: bool) -> Self {
        Self::load(path).unwrap_or_else(|_| Self::new(case_insensitive))
    }

    /// Get statistics about the index.
    pub fn stats(&self) -> TrigramIndexStats {
        let total_postings: u64 = self.posting_lists.values().map(|b| b.len()).sum();
        let avg_postings = if self.posting_lists.is_empty() {
            0.0
        } else {
            total_postings as f64 / self.posting_lists.len() as f64
        };

        TrigramIndexStats {
            file_count: self.files.len(),
            trigram_count: self.posting_lists.len(),
            total_postings,
            avg_postings_per_trigram: avg_postings,
            case_insensitive: self.case_insensitive,
        }
    }
}

/// Statistics about a trigram index.
#[derive(Debug, Clone)]
pub struct TrigramIndexStats {
    pub file_count: usize,
    pub trigram_count: usize,
    pub total_postings: u64,
    pub avg_postings_per_trigram: f64,
    pub case_insensitive: bool,
}

impl std::fmt::Display for TrigramIndexStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} files, {} trigrams, {:.1} avg postings{}",
            self.file_count,
            self.trigram_count,
            self.avg_postings_per_trigram,
            if self.case_insensitive {
                " (case-insensitive)"
            } else {
                ""
            }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_trigrams() {
        let text = b"hello";
        let trigrams = extract_trigrams(text);
        assert!(trigrams.contains(&[b'h', b'e', b'l']));
        assert!(trigrams.contains(&[b'e', b'l', b'l']));
        assert!(trigrams.contains(&[b'l', b'l', b'o']));
        assert_eq!(trigrams.len(), 3);
    }

    #[test]
    fn test_extract_trigrams_short() {
        assert!(extract_trigrams(b"").is_empty());
        assert!(extract_trigrams(b"a").is_empty());
        assert!(extract_trigrams(b"ab").is_empty());
        assert_eq!(extract_trigrams(b"abc").len(), 1);
    }

    #[test]
    fn test_extract_trigrams_lowercase() {
        let text = b"HeLLo";
        let trigrams = extract_trigrams_lowercase(text);
        assert!(trigrams.contains(&[b'h', b'e', b'l']));
        assert!(trigrams.contains(&[b'e', b'l', b'l']));
        assert!(trigrams.contains(&[b'l', b'l', b'o']));
    }

    #[test]
    fn test_trigram_conversion() {
        let t: Trigram = [b'a', b'b', b'c'];
        let n = trigram_to_u32(t);
        assert_eq!(u32_to_trigram(n), t);
    }

    #[test]
    fn test_extract_pattern_trigrams() {
        // Simple literal
        let trigrams = extract_pattern_trigrams("hello").unwrap();
        assert!(trigrams.contains(&[b'h', b'e', b'l']));
        assert!(trigrams.contains(&[b'e', b'l', b'l']));
        assert!(trigrams.contains(&[b'l', b'l', b'o']));

        // Pattern with wildcards - should extract literal parts
        let trigrams = extract_pattern_trigrams("Error.*Handler").unwrap();
        assert!(trigrams.contains(&[b'E', b'r', b'r']));
        assert!(trigrams.contains(&[b'H', b'a', b'n']));
    }

    #[test]
    fn test_extract_pattern_trigrams_no_literals() {
        // Pattern with no usable literals
        assert!(extract_pattern_trigrams("[a-z]+").is_none());
        assert!(extract_pattern_trigrams(".*").is_none());
        assert!(extract_pattern_trigrams(".").is_none());
    }

    #[test]
    fn test_index_add_and_query() {
        let mut index = TrigramIndex::new(false);

        index.add_file(Path::new("foo.txt"), b"hello world");
        index.add_file(Path::new("bar.txt"), b"goodbye world");
        index.add_file(Path::new("baz.txt"), b"hello there");

        // Query for "hello" trigrams
        let trigrams = extract_trigrams(b"hello");
        let results = index.query(&trigrams);
        assert_eq!(results.len(), 2);
        assert!(results.contains(&Path::new("foo.txt")));
        assert!(results.contains(&Path::new("baz.txt")));

        // Query for "world" trigrams
        let trigrams = extract_trigrams(b"world");
        let results = index.query(&trigrams);
        assert_eq!(results.len(), 2);
        assert!(results.contains(&Path::new("foo.txt")));
        assert!(results.contains(&Path::new("bar.txt")));

        // Query for "goodbye" trigrams
        let trigrams = extract_trigrams(b"goodbye");
        let results = index.query(&trigrams);
        assert_eq!(results.len(), 1);
        assert!(results.contains(&Path::new("bar.txt")));
    }

    #[test]
    fn test_index_query_pattern() {
        let mut index = TrigramIndex::new(false);

        index.add_file(Path::new("error.rs"), b"fn handle_error() {}");
        index.add_file(Path::new("main.rs"), b"fn main() {}");
        index.add_file(Path::new("handler.rs"), b"struct ErrorHandler {}");

        // Query with pattern
        let results = index.query_pattern("error").unwrap();
        assert!(results.contains(&Path::new("error.rs")));

        // Pattern with no extractable literals
        assert!(index.query_pattern("[a-z]+").is_none());
    }

    #[test]
    fn test_index_case_insensitive() {
        let mut index = TrigramIndex::new(true);

        index.add_file(Path::new("foo.txt"), b"Hello World");
        index.add_file(Path::new("bar.txt"), b"HELLO WORLD");

        let trigrams = extract_trigrams_lowercase(b"hello");
        let results = index.query(&trigrams);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_index_save_load() {
        let mut index = TrigramIndex::new(false);
        index.add_file(Path::new("foo.txt"), b"hello world");
        index.add_file(Path::new("bar.txt"), b"goodbye world");

        let temp_dir = tempfile::tempdir().unwrap();
        let index_path = temp_dir.path().join("trigrams.bin");

        index.save(&index_path).unwrap();
        let loaded = TrigramIndex::load(&index_path).unwrap();

        assert_eq!(loaded.file_count(), 2);
        assert_eq!(loaded.trigram_count(), index.trigram_count());

        let trigrams = extract_trigrams(b"hello");
        let results = loaded.query(&trigrams);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_index_remove_file() {
        let mut index = TrigramIndex::new(false);
        index.add_file(Path::new("foo.txt"), b"hello world");
        index.add_file(Path::new("bar.txt"), b"hello there");

        let trigrams = extract_trigrams(b"hello");
        assert_eq!(index.query(&trigrams).len(), 2);

        index.remove_file(Path::new("foo.txt"));

        let results = index.query(&trigrams);
        assert_eq!(results.len(), 1);
        assert!(results.contains(&Path::new("bar.txt")));
    }

    #[test]
    fn test_index_update_file() {
        let mut index = TrigramIndex::new(false);
        index.add_file(Path::new("foo.txt"), b"hello world");

        let hello_trigrams = extract_trigrams(b"hello");
        assert_eq!(index.query(&hello_trigrams).len(), 1);

        // Update the file
        index.add_file(Path::new("foo.txt"), b"goodbye world");

        // Old content should not match
        assert_eq!(index.query(&hello_trigrams).len(), 0);

        // New content should match
        let goodbye_trigrams = extract_trigrams(b"goodbye");
        assert_eq!(index.query(&goodbye_trigrams).len(), 1);
    }

    #[test]
    fn test_index_stats() {
        let mut index = TrigramIndex::new(false);
        index.add_file(Path::new("foo.txt"), b"hello");
        index.add_file(Path::new("bar.txt"), b"hello");

        let stats = index.stats();
        assert_eq!(stats.file_count, 2);
        assert_eq!(stats.trigram_count, 3); // hel, ell, llo
    }

    #[test]
    fn test_build_from_files() {
        let files = vec![
            (PathBuf::from("foo.txt"), b"hello world".to_vec()),
            (PathBuf::from("bar.txt"), b"goodbye world".to_vec()),
        ];

        let index = TrigramIndex::build_from_files(files, false).unwrap();
        assert_eq!(index.file_count(), 2);

        let trigrams = extract_trigrams(b"world");
        assert_eq!(index.query(&trigrams).len(), 2);
    }
}
