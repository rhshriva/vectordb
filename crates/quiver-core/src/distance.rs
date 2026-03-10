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
}

/// Squared Euclidean distance.
/// We use squared L2 for comparisons (avoids sqrt) and only sqrt when returning
/// the final distance to callers.
#[inline]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    // rustc auto-vectorizes this loop to SIMD (AVX2 / AVX-512 depending on target-cpu)
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Euclidean (L2) distance.
#[inline]
pub fn l2(a: &[f32], b: &[f32]) -> f32 {
    l2_squared(a, b).sqrt()
}

/// Raw dot product.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Dot-product distance: `-dot(a, b)`.
/// Negated so that "lower = more similar" is consistent with other metrics.
#[inline]
pub fn dot_product_distance(a: &[f32], b: &[f32]) -> f32 {
    -dot(a, b)
}

/// Cosine distance: `1 - cosine_similarity(a, b)`.
/// Returns 0 for identical vectors, 2 for opposite vectors.
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn l2_zero_for_identical() {
        let v = vec![1.0_f32, 2.0, 3.0];
        assert_abs_diff_eq!(l2(&v, &v), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn l2_known_value() {
        let a = vec![0.0_f32, 0.0, 0.0];
        let b = vec![3.0_f32, 4.0, 0.0];
        assert_abs_diff_eq!(l2(&a, &b), 5.0, epsilon = 1e-6);
    }

    #[test]
    fn cosine_identical_vectors() {
        let v = vec![1.0_f32, 0.0, 0.0];
        assert_abs_diff_eq!(cosine_distance(&v, &v), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![0.0_f32, 1.0];
        assert_abs_diff_eq!(cosine_distance(&a, &b), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn cosine_opposite_vectors() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![-1.0_f32, 0.0];
        assert_abs_diff_eq!(cosine_distance(&a, &b), 2.0, epsilon = 1e-6);
    }

    #[test]
    fn normalize_unit_length() {
        let mut v = vec![3.0_f32, 4.0];
        normalize(&mut v);
        let norm: f32 = dot(&v, &v).sqrt();
        assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn metric_dispatch() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![0.0_f32, 1.0];
        assert_abs_diff_eq!(Metric::L2.distance(&a, &b), 2.0_f32.sqrt(), epsilon = 1e-5);
        assert_abs_diff_eq!(Metric::Cosine.distance(&a, &b), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn dot_product_known_value() {
        // dot([1,2],[3,4]) = 11 → distance = -11
        let a = vec![1.0_f32, 2.0];
        let b = vec![3.0_f32, 4.0];
        assert_abs_diff_eq!(dot_product_distance(&a, &b), -11.0, epsilon = 1e-6);
    }

    #[test]
    fn dot_product_orthogonal_is_zero() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![0.0_f32, 1.0];
        assert_abs_diff_eq!(dot_product_distance(&a, &b), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn dot_product_metric_dispatch() {
        let a = vec![1.0_f32, 2.0];
        let b = vec![3.0_f32, 4.0];
        assert_abs_diff_eq!(Metric::DotProduct.distance(&a, &b), -11.0, epsilon = 1e-6);
    }

    #[test]
    fn cosine_zero_vector_is_maximally_dissimilar() {
        // Zero vector returns 1.0, not NaN.
        let zero = vec![0.0_f32, 0.0];
        let unit = vec![1.0_f32, 0.0];
        assert_abs_diff_eq!(cosine_distance(&zero, &unit), 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(cosine_distance(&unit, &zero), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn normalize_zero_vector_is_noop() {
        // Normalising a zero vector must not panic or produce NaN.
        let mut v = vec![0.0_f32, 0.0];
        normalize(&mut v);
        assert_eq!(v, vec![0.0, 0.0]);
    }
}
