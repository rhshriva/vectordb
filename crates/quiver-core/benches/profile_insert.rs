//! Standalone binary for profiling HNSW insert throughput.
//!
//! Usage:
//!   cargo bench -p quiver-core --bench profile_insert
//!   samply record target/release/deps/profile_insert-*

use quiver_core::distance::Metric;
use quiver_core::index::hnsw::{HnswConfig, HnswIndex};
use quiver_core::index::VectorIndex;
use rand::Rng;
use std::time::Instant;

const NUM_VECTORS: usize = 10_000;
const DIMENSIONS: usize = 128;

fn generate_random_vectors(n: usize, dim: usize) -> Vec<(u64, Vec<f32>)> {
    let mut rng = rand::thread_rng();
    (0..n)
        .map(|i| {
            let vec: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
            (i as u64, vec)
        })
        .collect()
}

fn main() {
    let vectors = generate_random_vectors(NUM_VECTORS, DIMENSIONS);

    // ── Run 1: Overall throughput ─────────────────────────────────────
    {
        let config = HnswConfig {
            ef_construction: 200,
            ef_search: 50,
            m: 12,
        };
        let mut index = HnswIndex::new(DIMENSIONS, Metric::L2, config);

        eprintln!("=== Overall Insert Throughput ===");
        let start = Instant::now();
        for (id, vec) in &vectors {
            index.add(*id, vec).unwrap();
        }
        let elapsed = start.elapsed();
        eprintln!("{} vectors in {:.2?} — {:.0} vec/s\n", NUM_VECTORS, elapsed, NUM_VECTORS as f64 / elapsed.as_secs_f64());
    }

    // ── Run 2: Instrumented insert (measures each phase) ─────────────
    {
        let config = HnswConfig {
            ef_construction: 200,
            ef_search: 50,
            m: 12,
        };
        let mut index = HnswIndex::new(DIMENSIONS, Metric::L2, config);

        eprintln!("=== Instrumented Insert (phase timing) ===");
        let start = Instant::now();
        let stats = index.add_batch_instrumented(&vectors);
        let elapsed = start.elapsed();

        eprintln!("Total:           {:.2?} ({:.0} vec/s)", elapsed, NUM_VECTORS as f64 / elapsed.as_secs_f64());
        eprintln!("  random_level:  {:.2?} ({:.1}%)", stats.random_level, pct(stats.random_level, elapsed));
        eprintln!("  clone_vec:     {:.2?} ({:.1}%)", stats.clone_vec, pct(stats.clone_vec, elapsed));
        eprintln!("  greedy_desc:   {:.2?} ({:.1}%)", stats.greedy_descent, pct(stats.greedy_descent, elapsed));
        eprintln!("  search_layer:  {:.2?} ({:.1}%)", stats.search_layer, pct(stats.search_layer, elapsed));
        eprintln!("    ├─ distance:   {:.2?} ({:.1}%)", stats.sl_distance, pct(stats.sl_distance, elapsed));
        eprintln!("    ├─ hash_lookup:{:.2?} ({:.1}%)", stats.sl_hash_lookup, pct(stats.sl_hash_lookup, elapsed));
        eprintln!("    └─ heap_ops:   {:.2?} ({:.1}%)", stats.sl_heap_ops, pct(stats.sl_heap_ops, elapsed));
        eprintln!("  set_neighbors: {:.2?} ({:.1}%)", stats.set_neighbors, pct(stats.set_neighbors, elapsed));
        eprintln!("  back_edges:    {:.2?} ({:.1}%)", stats.back_edges, pct(stats.back_edges, elapsed));
        eprintln!("  pruning:       {:.2?} ({:.1}%)", stats.pruning, pct(stats.pruning, elapsed));
        eprintln!("  node_insert:   {:.2?} ({:.1}%)", stats.node_insert, pct(stats.node_insert, elapsed));
    }

    // ── Run 3: Scaling behavior ──────────────────────────────────────
    {
        let config = HnswConfig {
            ef_construction: 200,
            ef_search: 50,
            m: 12,
        };
        let mut index = HnswIndex::new(DIMENSIONS, Metric::L2, config);

        eprintln!("\n=== Insert Scaling (throughput at different sizes) ===");
        let checkpoints = [100, 500, 1000, 2000, 5000, 10000];
        let mut last_checkpoint = 0;
        let mut last_time = Instant::now();

        for (id, vec) in &vectors {
            index.add(*id, vec).unwrap();
            let count = *id as usize + 1;
            if checkpoints.contains(&count) {
                let elapsed = last_time.elapsed();
                let batch = count - last_checkpoint;
                eprintln!("  {:>5} vectors: {:.0} vec/s (batch of {})", count, batch as f64 / elapsed.as_secs_f64(), batch);
                last_checkpoint = count;
                last_time = Instant::now();
            }
        }
    }
}

fn pct(part: std::time::Duration, total: std::time::Duration) -> f64 {
    100.0 * part.as_secs_f64() / total.as_secs_f64()
}
