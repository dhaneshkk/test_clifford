//! Matthews correlation coefficient from a confusion matrix.
pub fn mcc(tp: f64, fp: f64, fn_: f64, tn: f64) -> f64 {
    let d = ((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_)).sqrt();
    if d == 0.0 { 0.0 } else { (tp * tn - fp * fn_) / d }
}

#[cfg(test)]
mod tests {
    #[test]
    fn notebook_ppe_mcc() {
        // notebook report: confusion [[tn,fp],[fn,tp]] = [[451,42],[48,459]] → MCC 0.8200540091801047
        assert!((super::mcc(459.0, 42.0, 48.0, 451.0) - 0.8200540091801047).abs() < 1e-12);
    }
}
