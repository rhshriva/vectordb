//! Distance metrics and SIMD-accelerated kernel implementations.
//!
//! # Runtime dispatch
//!
//! The public functions (`l2_squared`, `dot`, …) select the fastest available
//! implementation at **runtime**:
//!
//! | Platform  | Feature detected | Implementation         |
//! |-----------|-----------------|------------------------|
//! | x86-64    | AVX2 + FMA      | 256-bit AVX2 + FMA      |
//! | x86-64    | (fallback)      | scalar                  |
//! | AArch64   | (always)        | 128-bit NEON            |
//! | other     | —               | scalar                  |
//!
//! All paths produce identical results within floating-point rounding.

/// Supported distance / similarity metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Metric {
    /// Euclidean distance (L2). Lower = more similar.
    L2,
    /// Cosine similarity converted to distance: `1 - cosine_sim`. Lower = more similar.
    Cosine,
    /// Inner / dot-product distance: `-dot(a, b)`. Lower = more similar.
    DotProduct,
}

impl Metric {
    /// Compute the distance between two equal-length vectors.
    ///
    /// # Panics
    /// Panics if `a.len() != b.len()`.
    #[inline]
    pub fn distance(self, a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "vector dimension mismatch");
        match self {
            Metric::L2 => l2(a, b),
            Metric::Cosine => cosine_distance(a, b),
            Metric::DotProduct => dot_product_distance(a, b),
        }
    }

    /// Ordering-preserving distance for internal hot paths (HNSW).
    ///
    /// Returns a value whose **ordering** matches `distance()` but may differ
    /// in absolute magnitude:
    /// - **L2**: returns `l2_squared(a, b)` (no sqrt — sqrt is monotonic so ordering is identical)
    /// - **Cosine / DotProduct**: identical to `distance()`
    ///
    /// Uses `debug_assert_eq` instead of `assert_eq` for the dimension check,
    /// so the check is elided in release builds (saving ~1 branch per call).
    ///
    /// Call [`Metric::fixup_distance`] to convert back to true distance when
    /// returning results to the caller.
    #[inline]
    pub fn distance_ord(self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "vector dimension mismatch");
        match self {
            Metric::L2 => l2_squared(a, b),
            Metric::Cosine => cosine_distance(a, b),
            Metric::DotProduct => dot_product_distance(a, b),
        }
    }

    /// Compute 4 ordering-preserving distances simultaneously (ILP).
    ///
    /// Loads the query vector once and computes 4 distances using independent
    /// accumulators for better CPU utilization.
    #[inline]
    pub fn distance_ord_batch4(self, query: &[f32], vecs: [&[f32]; 4]) -> [f32; 4] {
        match self {
            Metric::L2 => l2_squared_batch4(query, vecs),
            Metric::DotProduct => {
                let dots = dot_batch4(query, vecs);
                [-dots[0], -dots[1], -dots[2], -dots[3]]
            }
            Metric::Cosine => {
                // Cosine needs norms, fall back to individual calls
                [
                    cosine_distance(query, vecs[0]),
                    cosine_distance(query, vecs[1]),
                    cosine_distance(query, vecs[2]),
                    cosine_distance(query, vecs[3]),
                ]
            }
        }
    }

    /// Convert an ordering distance (from [`Metric::distance_ord`]) back to
    /// the true distance that users expect.
    ///
    /// - **L2**: applies `sqrt()` (since `distance_ord` returned squared L2)
    /// - **Cosine / DotProduct**: no-op (values are already correct)
    #[inline]
    pub fn fixup_distance(self, d: f32) -> f32 {
        match self {
            Metric::L2 => d.sqrt(),
            _ => d,
        }
    }
}

// ── Public API ─────────────────────────────────────────────────────────────────

/// Squared Euclidean distance (no sqrt). Dispatches to the fastest SIMD path.
#[inline]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    dispatch_l2_squared(a, b)
}

/// Euclidean (L2) distance.
#[inline]
pub fn l2(a: &[f32], b: &[f32]) -> f32 {
    l2_squared(a, b).sqrt()
}

/// Raw dot product. Dispatches to the fastest SIMD path.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    dispatch_dot(a, b)
}

/// Dot-product distance: `-dot(a, b)`.
/// Negated so that "lower = more similar" is consistent with other metrics.
#[inline]
pub fn dot_product_distance(a: &[f32], b: &[f32]) -> f32 {
    -dot(a, b)
}

/// Cosine distance: `1 - cosine_similarity(a, b)`.
/// Returns 0 for identical vectors, 2 for opposite vectors.
///
/// Cosine internally calls `dot` twice (once for the cross product, once each
/// for the norms), so it benefits from the SIMD `dot` kernel automatically.
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot_ab = dot(a, b);
    let norm_a = dot(a, a).sqrt();
    let norm_b = dot(b, b).sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0; // treat zero-vectors as maximally dissimilar
    }
    1.0 - (dot_ab / (norm_a * norm_b))
}

/// Normalize a vector in-place to unit length (L2 norm = 1).
/// Useful before inserting into a dot-product index to convert it to cosine search.
pub fn normalize(v: &mut [f32]) {
    let norm = dot(v, v).sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

// ── Batch-4 distance (ILP) ──────────────────────────────────────────────────

/// Compute 4 squared L2 distances from `query` to 4 vectors simultaneously.
///
/// Loads query once per chunk and uses 4 independent accumulators to enable
/// instruction-level parallelism (ILP).
#[inline]
pub fn l2_squared_batch4(query: &[f32], vecs: [&[f32]; 4]) -> [f32; 4] {
    dispatch_l2_squared_batch4(query, vecs)
}

/// Compute 4 dot products from `query` to 4 vectors simultaneously.
#[inline]
pub fn dot_batch4(query: &[f32], vecs: [&[f32]; 4]) -> [f32; 4] {
    dispatch_dot_batch4(query, vecs)
}

// ── Runtime dispatch ───────────────────────────────────────────────────────────

#[inline(always)]
fn dispatch_l2_squared(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: avx2 + fma confirmed available via CPUID.
            return unsafe { x86_64::l2_squared_avx2_fma(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is part of the AArch64 base ISA; always present.
        return unsafe { aarch64::l2_squared_neon(a, b) };
    }
    #[allow(unreachable_code)]
    scalar::l2_squared(a, b)
}

#[inline(always)]
fn dispatch_dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: avx2 + fma confirmed available via CPUID.
            return unsafe { x86_64::dot_avx2_fma(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is part of the AArch64 base ISA; always present.
        return unsafe { aarch64::dot_neon(a, b) };
    }
    #[allow(unreachable_code)]
    scalar::dot(a, b)
}

#[inline(always)]
fn dispatch_l2_squared_batch4(query: &[f32], vecs: [&[f32]; 4]) -> [f32; 4] {
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { aarch64::l2_squared_batch4_neon(query, vecs) };
    }
    // Fallback: 4 individual calls (still benefits from query cache locality)
    #[allow(unreachable_code)]
    [
        scalar::l2_squared(query, vecs[0]),
        scalar::l2_squared(query, vecs[1]),
        scalar::l2_squared(query, vecs[2]),
        scalar::l2_squared(query, vecs[3]),
    ]
}

#[inline(always)]
fn dispatch_dot_batch4(query: &[f32], vecs: [&[f32]; 4]) -> [f32; 4] {
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { aarch64::dot_batch4_neon(query, vecs) };
    }
    #[allow(unreachable_code)]
    [
        scalar::dot(query, vecs[0]),
        scalar::dot(query, vecs[1]),
        scalar::dot(query, vecs[2]),
        scalar::dot(query, vecs[3]),
    ]
}

// ── Scalar fallback ────────────────────────────────────────────────────────────

mod scalar {
    #[inline]
    pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let d = x - y;
                d * d
            })
            .sum()
    }

    #[inline]
    pub fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
}

// ── x86-64 AVX2 + FMA kernels ──────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
mod x86_64 {
    use std::arch::x86_64::*;

    /// Squared L2 distance using 256-bit AVX2 + FMA.
    ///
    /// Processes 8 floats per iteration with fused multiply-add:
    /// `acc = (a[i] - b[i])² + acc`
    ///
    /// # Safety
    /// Caller must ensure the CPU supports `avx2` and `fma`.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn l2_squared_avx2_fma(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let chunks = n / 8;
        let mut acc = _mm256_setzero_ps();

        for i in 0..chunks {
            let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            let diff = _mm256_sub_ps(va, vb);
            // acc = diff * diff + acc  (fused multiply-add)
            acc = _mm256_fmadd_ps(diff, diff, acc);
        }

        // Horizontal sum: reduce 8 lanes to 1 scalar.
        let hi = _mm256_extractf128_ps(acc, 1);      // upper 128 bits
        let lo = _mm256_castps256_ps128(acc);         // lower 128 bits
        let sum4 = _mm_add_ps(hi, lo);                // 4 lanes
        let shuf = _mm_movehdup_ps(sum4);             // [s1,s1,s3,s3]
        let sum2 = _mm_add_ps(sum4, shuf);            // [s0+s1, *, s2+s3, *]
        let shuf = _mm_movehl_ps(shuf, sum2);         // [s2+s3, *]
        let sum1 = _mm_add_ss(sum2, shuf);            // [s0+s1+s2+s3]
        let mut result = _mm_cvtss_f32(sum1);

        // Scalar tail for remaining elements
        for i in (chunks * 8)..n {
            let d = *a.get_unchecked(i) - *b.get_unchecked(i);
            result += d * d;
        }
        result
    }

    /// Dot product using 256-bit AVX2 + FMA.
    ///
    /// Processes 8 floats per iteration: `acc = a[i] * b[i] + acc`
    ///
    /// # Safety
    /// Caller must ensure the CPU supports `avx2` and `fma`.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_avx2_fma(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let chunks = n / 8;
        let mut acc = _mm256_setzero_ps();

        for i in 0..chunks {
            let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            // acc = va * vb + acc  (fused multiply-add)
            acc = _mm256_fmadd_ps(va, vb, acc);
        }

        // Horizontal sum
        let hi = _mm256_extractf128_ps(acc, 1);
        let lo = _mm256_castps256_ps128(acc);
        let sum4 = _mm_add_ps(hi, lo);
        let shuf = _mm_movehdup_ps(sum4);
        let sum2 = _mm_add_ps(sum4, shuf);
        let shuf = _mm_movehl_ps(shuf, sum2);
        let sum1 = _mm_add_ss(sum2, shuf);
        let mut result = _mm_cvtss_f32(sum1);

        for i in (chunks * 8)..n {
            result += *a.get_unchecked(i) * *b.get_unchecked(i);
        }
        result
    }
}

// ── AArch64 NEON kernels ───────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use std::arch::aarch64::*;

    /// Squared L2 distance using 128-bit NEON.
    ///
    /// Processes 4 floats per iteration with fused multiply-accumulate:
    /// `acc = (a[i] - b[i])² + acc`
    ///
    /// # Safety
    /// NEON is part of the AArch64 base ISA; always available.
    #[target_feature(enable = "neon")]
    pub unsafe fn l2_squared_neon(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let chunks = n / 4;
        let mut acc = vdupq_n_f32(0.0);

        for i in 0..chunks {
            let va = vld1q_f32(a.as_ptr().add(i * 4));
            let vb = vld1q_f32(b.as_ptr().add(i * 4));
            let diff = vsubq_f32(va, vb);
            // acc = diff * diff + acc  (fused multiply-accumulate)
            acc = vmlaq_f32(acc, diff, diff);
        }

        // Horizontal sum across 4 lanes
        let mut result = vaddvq_f32(acc);

        for i in (chunks * 4)..n {
            let d = *a.get_unchecked(i) - *b.get_unchecked(i);
            result += d * d;
        }
        result
    }

    /// Dot product using 128-bit NEON.
    ///
    /// Processes 4 floats per iteration: `acc = a[i] * b[i] + acc`
    ///
    /// # Safety
    /// NEON is part of the AArch64 base ISA; always available.
    #[target_feature(enable = "neon")]
    pub unsafe fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let chunks = n / 4;
        let mut acc = vdupq_n_f32(0.0);

        for i in 0..chunks {
            let va = vld1q_f32(a.as_ptr().add(i * 4));
            let vb = vld1q_f32(b.as_ptr().add(i * 4));
            acc = vmlaq_f32(acc, va, vb);
        }

        let mut result = vaddvq_f32(acc);

        for i in (chunks * 4)..n {
            result += *a.get_unchecked(i) * *b.get_unchecked(i);
        }
        result
    }

    /// Batch-4 squared L2 distance using NEON.
    ///
    /// Loads the query vector once per chunk and computes 4 distances with
    /// independent accumulators for instruction-level parallelism.
    ///
    /// # Safety
    /// NEON is part of the AArch64 base ISA; always available.
    #[target_feature(enable = "neon")]
    pub unsafe fn l2_squared_batch4_neon(query: &[f32], vecs: [&[f32]; 4]) -> [f32; 4] {
        let n = query.len();
        let chunks = n / 4;
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);

        for i in 0..chunks {
            let vq = vld1q_f32(query.as_ptr().add(i * 4));

            let v0 = vld1q_f32(vecs[0].as_ptr().add(i * 4));
            let d0 = vsubq_f32(vq, v0);
            acc0 = vmlaq_f32(acc0, d0, d0);

            let v1 = vld1q_f32(vecs[1].as_ptr().add(i * 4));
            let d1 = vsubq_f32(vq, v1);
            acc1 = vmlaq_f32(acc1, d1, d1);

            let v2 = vld1q_f32(vecs[2].as_ptr().add(i * 4));
            let d2 = vsubq_f32(vq, v2);
            acc2 = vmlaq_f32(acc2, d2, d2);

            let v3 = vld1q_f32(vecs[3].as_ptr().add(i * 4));
            let d3 = vsubq_f32(vq, v3);
            acc3 = vmlaq_f32(acc3, d3, d3);
        }

        let mut result = [vaddvq_f32(acc0), vaddvq_f32(acc1), vaddvq_f32(acc2), vaddvq_f32(acc3)];

        for i in (chunks * 4)..n {
            let q = *query.get_unchecked(i);
            let d0 = q - *vecs[0].get_unchecked(i); result[0] += d0 * d0;
            let d1 = q - *vecs[1].get_unchecked(i); result[1] += d1 * d1;
            let d2 = q - *vecs[2].get_unchecked(i); result[2] += d2 * d2;
            let d3 = q - *vecs[3].get_unchecked(i); result[3] += d3 * d3;
        }
        result
    }

    /// Batch-4 dot product using NEON.
    ///
    /// # Safety
    /// NEON is part of the AArch64 base ISA; always available.
    #[target_feature(enable = "neon")]
    pub unsafe fn dot_batch4_neon(query: &[f32], vecs: [&[f32]; 4]) -> [f32; 4] {
        let n = query.len();
        let chunks = n / 4;
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);

        for i in 0..chunks {
            let vq = vld1q_f32(query.as_ptr().add(i * 4));

            let v0 = vld1q_f32(vecs[0].as_ptr().add(i * 4));
            acc0 = vmlaq_f32(acc0, vq, v0);

            let v1 = vld1q_f32(vecs[1].as_ptr().add(i * 4));
            acc1 = vmlaq_f32(acc1, vq, v1);

            let v2 = vld1q_f32(vecs[2].as_ptr().add(i * 4));
            acc2 = vmlaq_f32(acc2, vq, v2);

            let v3 = vld1q_f32(vecs[3].as_ptr().add(i * 4));
            acc3 = vmlaq_f32(acc3, vq, v3);
        }

        let mut result = [vaddvq_f32(acc0), vaddvq_f32(acc1), vaddvq_f32(acc2), vaddvq_f32(acc3)];

        for i in (chunks * 4)..n {
            let q = *query.get_unchecked(i);
            result[0] += q * *vecs[0].get_unchecked(i);
            result[1] += q * *vecs[1].get_unchecked(i);
            result[2] += q * *vecs[2].get_unchecked(i);
            result[3] += q * *vecs[3].get_unchecked(i);
        }
        result
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn l2_zero_for_identical() {
        let v = vec![1.0_f32, 2.0, 3.0];
        assert_abs_diff_eq!(l2(&v, &v), 0.0, epsilon = 1e-5);
    }

    #[test]
    fn l2_known_value() {
        let a = vec![0.0_f32, 0.0, 0.0];
        let b = vec![3.0_f32, 4.0, 0.0];
        assert_abs_diff_eq!(l2(&a, &b), 5.0, epsilon = 1e-5);
    }

    #[test]
    fn cosine_identical_vectors() {
        let v = vec![1.0_f32, 0.0, 0.0];
        assert_abs_diff_eq!(cosine_distance(&v, &v), 0.0, epsilon = 1e-5);
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![0.0_f32, 1.0];
        assert_abs_diff_eq!(cosine_distance(&a, &b), 1.0, epsilon = 1e-5);
    }

    #[test]
    fn cosine_opposite_vectors() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![-1.0_f32, 0.0];
        assert_abs_diff_eq!(cosine_distance(&a, &b), 2.0, epsilon = 1e-5);
    }

    #[test]
    fn normalize_unit_length() {
        let mut v = vec![3.0_f32, 4.0];
        normalize(&mut v);
        let norm: f32 = dot(&v, &v).sqrt();
        assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn metric_dispatch() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![0.0_f32, 1.0];
        assert_abs_diff_eq!(Metric::L2.distance(&a, &b), 2.0_f32.sqrt(), epsilon = 1e-5);
        assert_abs_diff_eq!(Metric::Cosine.distance(&a, &b), 1.0, epsilon = 1e-5);
    }

    #[test]
    fn dot_product_known_value() {
        // dot([1,2],[3,4]) = 11 → distance = -11
        let a = vec![1.0_f32, 2.0];
        let b = vec![3.0_f32, 4.0];
        assert_abs_diff_eq!(dot_product_distance(&a, &b), -11.0, epsilon = 1e-5);
    }

    #[test]
    fn dot_product_orthogonal_is_zero() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![0.0_f32, 1.0];
        assert_abs_diff_eq!(dot_product_distance(&a, &b), 0.0, epsilon = 1e-5);
    }

    #[test]
    fn dot_product_metric_dispatch() {
        let a = vec![1.0_f32, 2.0];
        let b = vec![3.0_f32, 4.0];
        assert_abs_diff_eq!(Metric::DotProduct.distance(&a, &b), -11.0, epsilon = 1e-5);
    }

    #[test]
    fn cosine_zero_vector_is_maximally_dissimilar() {
        let zero = vec![0.0_f32, 0.0];
        let unit = vec![1.0_f32, 0.0];
        assert_abs_diff_eq!(cosine_distance(&zero, &unit), 1.0, epsilon = 1e-5);
        assert_abs_diff_eq!(cosine_distance(&unit, &zero), 1.0, epsilon = 1e-5);
    }

    #[test]
    fn normalize_zero_vector_is_noop() {
        let mut v = vec![0.0_f32, 0.0];
        normalize(&mut v);
        assert_eq!(v, vec![0.0, 0.0]);
    }

    /// Verify SIMD kernels agree with the scalar reference on large vectors.
    #[test]
    fn simd_matches_scalar_large_vector() {
        let n = 1536;
        let a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001).sin()).collect();
        let b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001).cos()).collect();

        let scalar_l2 = scalar::l2_squared(&a, &b);
        let scalar_dot = scalar::dot(&a, &b);

        // dispatch_* will call SIMD if available, else scalar — both must agree.
        assert_abs_diff_eq!(dispatch_l2_squared(&a, &b), scalar_l2, epsilon = 1e-2);
        assert_abs_diff_eq!(dispatch_dot(&a, &b), scalar_dot, epsilon = 1e-2);
    }

    /// Verify on a length not divisible by 8 (AVX2 lane width) or 4 (NEON lane width).
    #[test]
    fn simd_handles_non_aligned_length() {
        for n in [1usize, 3, 5, 7, 9, 13, 17, 31, 33] {
            let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();

            let ref_l2 = scalar::l2_squared(&a, &b);
            let ref_dot = scalar::dot(&a, &b);

            let got_l2 = dispatch_l2_squared(&a, &b);
            assert!(
                (got_l2 - ref_l2).abs() < 1e-3,
                "l2_squared mismatch at n={n}: got {got_l2} vs {ref_l2}"
            );
            let got_dot = dispatch_dot(&a, &b);
            assert!(
                (got_dot - ref_dot).abs() < 1e-3,
                "dot mismatch at n={n}: got {got_dot} vs {ref_dot}"
            );
        }
    }

    /// Verify batch-4 distance matches individual calls.
    #[test]
    fn batch4_matches_individual() {
        let n = 128;
        let query: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();
        let vecs: Vec<Vec<f32>> = (0..4)
            .map(|k| (0..n).map(|i| ((i + k * 37) as f32 * 0.01).cos()).collect())
            .collect();

        let batch_l2 = l2_squared_batch4(&query, [&vecs[0], &vecs[1], &vecs[2], &vecs[3]]);
        let batch_dot = dot_batch4(&query, [&vecs[0], &vecs[1], &vecs[2], &vecs[3]]);

        for k in 0..4 {
            let ref_l2 = l2_squared(&query, &vecs[k]);
            assert!(
                (batch_l2[k] - ref_l2).abs() < 1e-2,
                "batch4 l2 mismatch at k={k}: got {} vs {ref_l2}", batch_l2[k]
            );
            let ref_dot = dot(&query, &vecs[k]);
            assert!(
                (batch_dot[k] - ref_dot).abs() < 1e-2,
                "batch4 dot mismatch at k={k}: got {} vs {ref_dot}", batch_dot[k]
            );
        }
    }

    #[test]
    fn batch4_metric_dispatch() {
        let n = 128;
        let query: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();
        let vecs: Vec<Vec<f32>> = (0..4)
            .map(|k| (0..n).map(|i| ((i + k * 37) as f32 * 0.01).cos()).collect())
            .collect();
        let refs = [&vecs[0][..], &vecs[1][..], &vecs[2][..], &vecs[3][..]];

        for metric in [Metric::L2, Metric::DotProduct, Metric::Cosine] {
            let batch = metric.distance_ord_batch4(&query, refs);
            for k in 0..4 {
                let individual = metric.distance_ord(&query, &vecs[k]);
                assert!(
                    (batch[k] - individual).abs() < 1e-2,
                    "batch4 metric {:?} mismatch at k={k}: got {} vs {individual}",
                    metric, batch[k]
                );
            }
        }
    }

    // ── Additional coverage tests ────────────────────────────────────────────

    #[test]
    fn fixup_distance_l2_applies_sqrt() {
        assert_abs_diff_eq!(Metric::L2.fixup_distance(4.0), 2.0, epsilon = 1e-5);
        assert_abs_diff_eq!(Metric::L2.fixup_distance(0.0), 0.0, epsilon = 1e-5);
    }

    #[test]
    fn fixup_distance_cosine_and_dot_are_noop() {
        assert_abs_diff_eq!(Metric::Cosine.fixup_distance(0.5), 0.5, epsilon = 1e-5);
        assert_abs_diff_eq!(Metric::DotProduct.fixup_distance(-3.0), -3.0, epsilon = 1e-5);
    }

    #[test]
    fn distance_ord_l2_returns_squared() {
        let a = vec![0.0_f32, 0.0, 0.0];
        let b = vec![3.0_f32, 4.0, 0.0];
        assert_abs_diff_eq!(Metric::L2.distance_ord(&a, &b), 25.0, epsilon = 1e-3);
    }

    #[test]
    fn distance_ord_cosine_same_as_distance() {
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![4.0_f32, 5.0, 6.0];
        assert_abs_diff_eq!(
            Metric::Cosine.distance_ord(&a, &b),
            Metric::Cosine.distance(&a, &b),
            epsilon = 1e-5
        );
    }

    #[test]
    fn distance_ord_dot_same_as_distance() {
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![4.0_f32, 5.0, 6.0];
        assert_abs_diff_eq!(
            Metric::DotProduct.distance_ord(&a, &b),
            Metric::DotProduct.distance(&a, &b),
            epsilon = 1e-5
        );
    }

    #[test]
    #[should_panic(expected = "vector dimension mismatch")]
    fn metric_distance_panics_on_dimension_mismatch() {
        let a = vec![1.0_f32, 2.0];
        let b = vec![1.0_f32, 2.0, 3.0];
        Metric::L2.distance(&a, &b);
    }

    #[test]
    fn l2_squared_zero_vectors() {
        let z = vec![0.0_f32; 8];
        assert_abs_diff_eq!(l2_squared(&z, &z), 0.0, epsilon = 1e-5);
    }

    #[test]
    fn dot_zero_vectors() {
        let z = vec![0.0_f32; 8];
        assert_abs_diff_eq!(dot(&z, &z), 0.0, epsilon = 1e-5);
    }

    #[test]
    fn l2_squared_identical_vectors() {
        let v = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        assert_abs_diff_eq!(l2_squared(&v, &v), 0.0, epsilon = 1e-5);
    }

    #[test]
    fn dot_self_is_squared_norm() {
        let v = vec![3.0_f32, 4.0];
        assert_abs_diff_eq!(dot(&v, &v), 25.0, epsilon = 1e-5);
    }

    #[test]
    fn cosine_both_zero_vectors() {
        let z = vec![0.0_f32, 0.0];
        assert_abs_diff_eq!(cosine_distance(&z, &z), 1.0, epsilon = 1e-5);
    }

    #[test]
    fn normalize_preserves_direction() {
        let mut v = vec![3.0_f32, 4.0];
        normalize(&mut v);
        assert_abs_diff_eq!(v[0], 0.6, epsilon = 1e-5);
        assert_abs_diff_eq!(v[1], 0.8, epsilon = 1e-5);
    }

    #[test]
    fn l2_large_values() {
        let a = vec![1e10_f32, 1e10];
        let b = vec![-1e10_f32, -1e10];
        let d = l2(&a, &b);
        assert!(d.is_finite());
        assert!(d > 0.0);
    }

    #[test]
    fn dot_product_distance_negates_dot() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![1.0_f32, 0.0];
        assert_abs_diff_eq!(dot_product_distance(&a, &b), -1.0, epsilon = 1e-5);
    }

    #[test]
    fn batch4_with_non_aligned_length() {
        let n = 13;
        let query: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let vecs: Vec<Vec<f32>> = (0..4)
            .map(|k| (0..n).map(|i| (i + k * 5) as f32).collect())
            .collect();

        let batch_l2 = l2_squared_batch4(&query, [&vecs[0], &vecs[1], &vecs[2], &vecs[3]]);
        for k in 0..4 {
            let ref_l2 = l2_squared(&query, &vecs[k]);
            assert!(
                (batch_l2[k] - ref_l2).abs() < 1e-2,
                "batch4 l2 tail mismatch at k={k}"
            );
        }

        let batch_dot = dot_batch4(&query, [&vecs[0], &vecs[1], &vecs[2], &vecs[3]]);
        for k in 0..4 {
            let ref_dot = dot(&query, &vecs[k]);
            assert!(
                (batch_dot[k] - ref_dot).abs() < 1e-2,
                "batch4 dot tail mismatch at k={k}"
            );
        }
    }

    #[test]
    fn batch4_empty_vectors() {
        let query: Vec<f32> = vec![];
        let v: Vec<f32> = vec![];
        let batch = l2_squared_batch4(&query, [&v, &v, &v, &v]);
        assert_eq!(batch, [0.0, 0.0, 0.0, 0.0]);
        let batch = dot_batch4(&query, [&v, &v, &v, &v]);
        assert_eq!(batch, [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn metric_distance_ord_batch4_dot_product_negates() {
        let query = vec![1.0_f32, 2.0, 3.0, 4.0];
        let v = vec![1.0_f32, 1.0, 1.0, 1.0];
        let batch = Metric::DotProduct.distance_ord_batch4(&query, [&v, &v, &v, &v]);
        let expected = -dot(&query, &v);
        for k in 0..4 {
            assert_abs_diff_eq!(batch[k], expected, epsilon = 1e-5);
        }
    }

    #[test]
    fn scalar_l2_squared_directly() {
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![4.0_f32, 5.0, 6.0];
        assert_abs_diff_eq!(scalar::l2_squared(&a, &b), 27.0, epsilon = 1e-5);
    }

    #[test]
    fn scalar_dot_directly() {
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![4.0_f32, 5.0, 6.0];
        assert_abs_diff_eq!(scalar::dot(&a, &b), 32.0, epsilon = 1e-5);
    }

    #[test]
    fn cosine_parallel_scaled_vectors() {
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![2.0_f32, 4.0, 6.0];
        assert_abs_diff_eq!(cosine_distance(&a, &b), 0.0, epsilon = 1e-5);
    }

    #[test]
    fn fixup_roundtrip_l2() {
        let a = vec![0.0_f32, 0.0];
        let b = vec![3.0_f32, 4.0];
        let ord = Metric::L2.distance_ord(&a, &b);
        let fixed = Metric::L2.fixup_distance(ord);
        assert_abs_diff_eq!(fixed, Metric::L2.distance(&a, &b), epsilon = 1e-3);
    }
}
