//! Turning byte counts into what people read: sizes, transfer rates and time left.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Span of recent samples a transfer rate is averaged over: long enough to smooth out bursty
/// connections, short enough to follow a real change of speed.
const RATE_WINDOW: Duration = Duration::from_secs(5);
/// Shortest span that gives a meaningful rate; before that none is reported.
const RATE_MIN_SPAN: Duration = Duration::from_secs(1);

/// Transfer rate over a sliding window of `(time, bytes so far)` samples.
#[derive(Clone, Debug, Default)]
pub struct Rate {
    samples: VecDeque<(Instant, u64)>,
}

impl Rate {
    /// Record that `bytes` have been transferred by `at`. A count that goes backwards (a new
    /// file) starts over.
    pub fn push(&mut self, at: Instant, bytes: u64) {
        if self.samples.back().is_some_and(|&(_, last)| bytes < last) {
            self.samples.clear();
        }
        // The clock starts with the first byte, not while the connection is being set up.
        if self.samples.len() == 1 && self.samples[0].1 == bytes {
            self.samples.clear();
        }
        self.samples.push_back((at, bytes));
        while self.samples.len() > 2 && self.samples.get(1).is_some_and(|&(t, _)| at.duration_since(t) >= RATE_WINDOW) {
            self.samples.pop_front();
        }
    }

    /// Bytes per second, once the samples span [`RATE_MIN_SPAN`].
    #[must_use]
    pub fn bytes_per_sec(&self) -> Option<f64> {
        let (&(start, first), &(end, last)) = (self.samples.front()?, self.samples.back()?);
        let span = end.duration_since(start);
        (span >= RATE_MIN_SPAN).then(|| (last - first) as f64 / span.as_secs_f64())
    }
}

/// Seconds to transfer `remaining` bytes at `bytes_per_sec`, if the rate is usable.
#[must_use]
pub fn time_left(remaining: u64, bytes_per_sec: Option<f64>) -> Option<Duration> {
    let rate = bytes_per_sec.filter(|r| *r >= 1.0)?;
    Some(Duration::from_secs_f64(remaining as f64 / rate))
}

/// Decimal size, as download sizes are usually quoted: `1500` -> `1.5 kB`, `3.5e9` -> `3.5 GB`.
#[must_use]
pub fn human_size(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "kB", "MB", "GB", "TB"];
    const STEP: f64 = 1000.0;
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= STEP && unit < UNITS.len() - 1 {
        value /= STEP;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else if value >= 100.0 {
        format!("{value:.0} {}", UNITS[unit])
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

/// `2.4 MB/s`.
#[must_use]
pub fn human_rate(bytes_per_sec: f64) -> String {
    format!("{}/s", human_size(bytes_per_sec as u64))
}

/// Rounded, friendly duration: `45 s`, `12 min`, `1 h 20 min`.
#[must_use]
pub fn human_duration(duration: Duration) -> String {
    const MINUTE: u64 = 60;
    const HOUR: u64 = 60 * MINUTE;
    let secs = duration.as_secs();
    if secs < MINUTE {
        format!("{} s", secs.max(1))
    } else if secs < HOUR {
        format!("{} min", secs.div_ceil(MINUTE))
    } else {
        let minutes = secs.div_ceil(MINUTE);
        match minutes % MINUTE {
            0 => format!("{} h", minutes / MINUTE),
            rest => format!("{} h {rest} min", minutes / MINUTE),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_are_human_readable() {
        assert_eq!(human_size(512), "512 B");
        assert_eq!(human_size(1500), "1.5 kB");
        assert_eq!(human_size(240_868_705), "241 MB");
        assert_eq!(human_size(3_500_000_000), "3.5 GB");
        assert_eq!(human_rate(2_400_000.0), "2.4 MB/s");
    }

    #[test]
    fn durations_round_up_to_friendly_units() {
        assert_eq!(human_duration(Duration::from_millis(300)), "1 s");
        assert_eq!(human_duration(Duration::from_secs(45)), "45 s");
        assert_eq!(human_duration(Duration::from_secs(61)), "2 min");
        assert_eq!(human_duration(Duration::from_hours(1)), "1 h");
        assert_eq!(human_duration(Duration::from_mins(80)), "1 h 20 min");
    }

    #[test]
    fn rate_averages_over_the_window() {
        let start = Instant::now();
        let mut rate = Rate::default();
        rate.push(start, 0);
        assert_eq!(rate.bytes_per_sec(), None);
        rate.push(start + Duration::from_secs(2), 2_000_000);
        assert_eq!(rate.bytes_per_sec(), Some(1_000_000.0));
        // Old samples leave the window: only the last few seconds count.
        for second in 3..=20 {
            rate.push(start + Duration::from_secs(second), 2_000_000 + (second - 2) * 500_000);
        }
        let recent = rate.bytes_per_sec().unwrap();
        assert!((recent - 500_000.0).abs() < 1.0, "{recent}");
        assert_eq!(time_left(1_000_000, Some(recent)), Some(Duration::from_secs(2)));
    }

    #[test]
    fn rate_restarts_when_the_count_goes_back() {
        let start = Instant::now();
        let mut rate = Rate::default();
        rate.push(start, 5_000_000);
        rate.push(start + Duration::from_secs(2), 0);
        assert_eq!(rate.bytes_per_sec(), None);
        assert_eq!(time_left(10, None), None);
    }
}
