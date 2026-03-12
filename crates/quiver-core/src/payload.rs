use serde::{de, Deserialize, Deserializer, Serialize};

/// Arbitrary JSON metadata stored alongside a vector.
pub type Payload = serde_json::Value;

/// A leaf field-level predicate.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct FieldFilter {
    pub field: String,
    pub op: FieldOp,
}

/// Supported comparison operators for payload filtering.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FieldOp {
    #[serde(rename = "$eq")]
    Eq(serde_json::Value),
    #[serde(rename = "$ne")]
    Ne(serde_json::Value),
    #[serde(rename = "$in")]
    In(Vec<serde_json::Value>),
    #[serde(rename = "$gt")]
    Gt(serde_json::Value),
    #[serde(rename = "$gte")]
    Gte(serde_json::Value),
    #[serde(rename = "$lt")]
    Lt(serde_json::Value),
    #[serde(rename = "$lte")]
    Lte(serde_json::Value),
}

/// A filter condition tree for metadata-filtered search.
///
/// Wire formats:
/// - AND: `{"$and": [cond, ...]}`
/// - OR:  `{"$or": [cond, ...]}`
/// - Field: `{"field_name": {"$eq": value}}`
#[derive(Debug, Clone, Serialize, PartialEq)]
pub enum FilterCondition {
    And(Vec<FilterCondition>),
    Or(Vec<FilterCondition>),
    Field(FieldFilter),
}

impl<'de> Deserialize<'de> for FilterCondition {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let map: serde_json::Map<String, serde_json::Value> =
            serde_json::Map::deserialize(d)?;

        if let Some(arr) = map.get("$and") {
            let conditions: Vec<FilterCondition> =
                serde_json::from_value(arr.clone()).map_err(de::Error::custom)?;
            return Ok(FilterCondition::And(conditions));
        }
        if let Some(arr) = map.get("$or") {
            let conditions: Vec<FilterCondition> =
                serde_json::from_value(arr.clone()).map_err(de::Error::custom)?;
            return Ok(FilterCondition::Or(conditions));
        }
        // Field filter: exactly one key = field name, value = op map
        if map.len() == 1 {
            let (field, op_val) = map.into_iter().next().unwrap();
            let op: FieldOp =
                serde_json::from_value(op_val).map_err(de::Error::custom)?;
            return Ok(FilterCondition::Field(FieldFilter { field, op }));
        }
        Err(de::Error::custom(
            "FilterCondition must be $and, $or, or a single field predicate",
        ))
    }
}

/// Evaluate a filter against a payload. Returns `true` if the payload matches.
/// Returns `false` for missing fields or type mismatches (permissive filtering).
pub fn matches_filter(payload: &Payload, filter: &FilterCondition) -> bool {
    match filter {
        FilterCondition::And(conditions) => {
            conditions.iter().all(|c| matches_filter(payload, c))
        }
        FilterCondition::Or(conditions) => {
            conditions.iter().any(|c| matches_filter(payload, c))
        }
        FilterCondition::Field(ff) => eval_field_op(payload, &ff.field, &ff.op),
    }
}

/// Get a value from a payload by dot-notation field path (e.g. "meta.author").
fn get_field<'a>(payload: &'a Payload, field: &str) -> Option<&'a serde_json::Value> {
    let mut current = payload;
    for part in field.split('.') {
        current = current.get(part)?;
    }
    Some(current)
}

fn compare_values(a: &serde_json::Value, b: &serde_json::Value) -> Option<std::cmp::Ordering> {
    match (a, b) {
        (serde_json::Value::Number(x), serde_json::Value::Number(y)) => {
            x.as_f64()?.partial_cmp(&y.as_f64()?)
        }
        (serde_json::Value::String(x), serde_json::Value::String(y)) => Some(x.cmp(y)),
        _ => None,
    }
}

fn eval_field_op(payload: &Payload, field: &str, op: &FieldOp) -> bool {
    let value = match get_field(payload, field) {
        Some(v) => v,
        None => return false,
    };

    match op {
        FieldOp::Eq(expected) => value == expected,
        FieldOp::Ne(expected) => value != expected,
        FieldOp::In(options) => options.iter().any(|opt| value == opt),
        FieldOp::Gt(threshold) => {
            compare_values(value, threshold)
                .map(|ord| ord == std::cmp::Ordering::Greater)
                .unwrap_or(false)
        }
        FieldOp::Gte(threshold) => {
            compare_values(value, threshold)
                .map(|ord| ord != std::cmp::Ordering::Less)
                .unwrap_or(false)
        }
        FieldOp::Lt(threshold) => {
            compare_values(value, threshold)
                .map(|ord| ord == std::cmp::Ordering::Less)
                .unwrap_or(false)
        }
        FieldOp::Lte(threshold) => {
            compare_values(value, threshold)
                .map(|ord| ord != std::cmp::Ordering::Greater)
                .unwrap_or(false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn payload(v: serde_json::Value) -> Payload {
        v
    }

    fn parse_filter(s: &str) -> FilterCondition {
        serde_json::from_str(s).unwrap()
    }

    #[test]
    fn filter_eq_matches() {
        let p = payload(json!({"tag": "news"}));
        let f = parse_filter(r#"{"tag": {"$eq": "news"}}"#);
        assert!(matches_filter(&p, &f));
    }

    #[test]
    fn filter_eq_no_match() {
        let p = payload(json!({"tag": "sports"}));
        let f = parse_filter(r#"{"tag": {"$eq": "news"}}"#);
        assert!(!matches_filter(&p, &f));
    }

    #[test]
    fn filter_ne() {
        let p = payload(json!({"tag": "sports"}));
        let f = parse_filter(r#"{"tag": {"$ne": "news"}}"#);
        assert!(matches_filter(&p, &f));
    }

    #[test]
    fn filter_in_matches() {
        let p = payload(json!({"tag": "news"}));
        let f = parse_filter(r#"{"tag": {"$in": ["news", "sports"]}}"#);
        assert!(matches_filter(&p, &f));
    }

    #[test]
    fn filter_in_no_match() {
        let p = payload(json!({"tag": "tech"}));
        let f = parse_filter(r#"{"tag": {"$in": ["news", "sports"]}}"#);
        assert!(!matches_filter(&p, &f));
    }

    #[test]
    fn filter_gt_match() {
        let p = payload(json!({"score": 0.7}));
        let f = parse_filter(r#"{"score": {"$gt": 0.5}}"#);
        assert!(matches_filter(&p, &f));
    }

    #[test]
    fn filter_gt_no_match() {
        let p = payload(json!({"score": 0.3}));
        let f = parse_filter(r#"{"score": {"$gt": 0.5}}"#);
        assert!(!matches_filter(&p, &f));
    }

    #[test]
    fn filter_gte() {
        let p = payload(json!({"score": 0.5}));
        let f = parse_filter(r#"{"score": {"$gte": 0.5}}"#);
        assert!(matches_filter(&p, &f));
    }

    #[test]
    fn filter_lt_lte() {
        let p = payload(json!({"score": 0.3}));
        assert!(matches_filter(&p, &parse_filter(r#"{"score": {"$lt": 0.5}}"#)));
        assert!(matches_filter(&p, &parse_filter(r#"{"score": {"$lte": 0.3}}"#)));
        assert!(!matches_filter(&p, &parse_filter(r#"{"score": {"$lt": 0.3}}"#)));
    }

    #[test]
    fn filter_and() {
        let p = payload(json!({"tag": "news", "score": 0.8}));
        let f = parse_filter(r#"{"$and": [{"tag": {"$eq": "news"}}, {"score": {"$gte": 0.5}}]}"#);
        assert!(matches_filter(&p, &f));
    }

    #[test]
    fn filter_and_partial_fail() {
        let p = payload(json!({"tag": "news", "score": 0.2}));
        let f = parse_filter(r#"{"$and": [{"tag": {"$eq": "news"}}, {"score": {"$gte": 0.5}}]}"#);
        assert!(!matches_filter(&p, &f));
    }

    #[test]
    fn filter_or() {
        let p = payload(json!({"tag": "sports"}));
        let f = parse_filter(r#"{"$or": [{"tag": {"$eq": "news"}}, {"tag": {"$eq": "sports"}}]}"#);
        assert!(matches_filter(&p, &f));
    }

    #[test]
    fn filter_missing_field_returns_false() {
        let p = payload(json!({"other": "value"}));
        let f = parse_filter(r#"{"tag": {"$eq": "news"}}"#);
        assert!(!matches_filter(&p, &f));
    }

    #[test]
    fn filter_dot_notation() {
        let p = payload(json!({"meta": {"author": "alice"}}));
        let f = parse_filter(r#"{"meta.author": {"$eq": "alice"}}"#);
        assert!(matches_filter(&p, &f));
    }

    #[test]
    fn filter_type_mismatch_returns_false() {
        let p = payload(json!({"score": "high"}));
        let f = parse_filter(r#"{"score": {"$gt": 0.5}}"#);
        assert!(!matches_filter(&p, &f));
    }

    #[test]
    fn filter_deserialization_and_roundtrip() {
        let s = r#"{"$and": [{"tag": {"$eq": "news"}}, {"score": {"$gte": 0.5}}]}"#;
        let f: FilterCondition = serde_json::from_str(s).unwrap();
        assert!(matches!(f, FilterCondition::And(_)));
    }
}
