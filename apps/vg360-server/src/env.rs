//! Typed readers for the `VG360_*` environment variables the server parses itself. Each reader is a thin wrapper
//! over a pure `parse_*` helper, which is what the unit tests exercise (the tests never touch the process env).
//! An unset variable (or one that is empty after trimming) is `Ok(None)`; a malformed or out-of-range one is an
//! `Err` naming the variable, so the caller can refuse to start rather than silently use a default.

/// Parse a finite number in `lo..=hi`.
pub fn parse_f64(name: &str, raw: Option<&str>, lo: f64, hi: f64) -> Result<Option<f64>, String> {
    let Some(s) = raw.map(str::trim).filter(|s| !s.is_empty()) else { return Ok(None) };
    let v: f64 = s.parse().map_err(|_| format!("{name}={s:?} is not a number"))?;
    if !v.is_finite() || v < lo || v > hi {
        return Err(format!("{name}={s} is out of range {lo}..={hi}"));
    }
    Ok(Some(v))
}

/// Parse `WxH` (or `WXH`) with each side in `64..=8192`.
pub fn parse_size(name: &str, raw: Option<&str>) -> Result<Option<(usize, usize)>, String> {
    let Some(s) = raw.map(str::trim).filter(|s| !s.is_empty()) else { return Ok(None) };
    let bad = || format!("{name}={s:?} must look like WIDTHxHEIGHT, e.g. 1280x720");
    let (w, h) = s.split_once(['x', 'X']).ok_or_else(bad)?;
    let (w, h): (usize, usize) = (w.trim().parse().map_err(|_| bad())?, h.trim().parse().map_err(|_| bad())?);
    if !(64..=8192).contains(&w) || !(64..=8192).contains(&h) {
        return Err(format!("{name}={s} is out of range: each side must be 64..=8192 pixels"));
    }
    Ok(Some((w, h)))
}

/// Read an `f64` environment variable in `lo..=hi`.
pub fn env_f64(name: &str, lo: f64, hi: f64) -> Result<Option<f64>, String> {
    parse_f64(name, std::env::var(name).ok().as_deref(), lo, hi)
}

/// Read a `WxH` environment variable.
pub fn env_size(name: &str) -> Result<Option<(usize, usize)>, String> {
    parse_size(name, std::env::var(name).ok().as_deref())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unset_or_blank_is_none() {
        assert_eq!(parse_f64("X", None, 0.0, 1.0), Ok(None));
        assert_eq!(parse_f64("X", Some("   "), 0.0, 1.0), Ok(None));
        assert_eq!(parse_size("X", None), Ok(None));
        assert_eq!(parse_size("X", Some("")), Ok(None));
    }

    #[test]
    fn f64_in_range() {
        assert_eq!(parse_f64("VG360_LIVE_HFOV", Some(" 70.5 "), 10.0, 170.0), Ok(Some(70.5)));
        assert_eq!(parse_f64("VG360_LIVE_PITCH", Some("-5"), -5.0, 89.0), Ok(Some(-5.0)));
    }

    #[test]
    fn f64_rejects_garbage_and_range() {
        let e = parse_f64("VG360_LIVE_HFOV", Some("wide"), 10.0, 170.0).unwrap_err();
        assert!(e.contains("VG360_LIVE_HFOV") && e.contains("not a number"), "{e}");
        let e = parse_f64("VG360_LIVE_HFOV", Some("200"), 10.0, 170.0).unwrap_err();
        assert!(e.contains("out of range"), "{e}");
        assert!(parse_f64("X", Some("NaN"), 0.0, 1e9).is_err());
        assert!(parse_f64("X", Some("inf"), 0.0, 1e9).is_err());
    }

    #[test]
    fn size_parses_both_separators() {
        assert_eq!(parse_size("VG360_LIVE_SIZE", Some("1280x720")), Ok(Some((1280, 720))));
        assert_eq!(parse_size("VG360_LIVE_SIZE", Some(" 1920X1080 ")), Ok(Some((1920, 1080))));
    }

    #[test]
    fn size_rejects_malformed_and_out_of_range() {
        for bad in ["1280", "1280*720", "axb", "1280x", "x720"] {
            let e = parse_size("VG360_LIVE_SIZE", Some(bad)).unwrap_err();
            assert!(e.contains("WIDTHxHEIGHT"), "{bad}: {e}");
        }
        let e = parse_size("VG360_LIVE_SIZE", Some("32x720")).unwrap_err();
        assert!(e.contains("out of range"), "{e}");
        assert!(parse_size("VG360_LIVE_SIZE", Some("1280x9000")).is_err());
    }
}
