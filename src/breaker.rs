//! A small circuit breaker.
//!
//! Closed: calls go through; failures are counted. After `threshold`
//! failures in a row the breaker opens for `cooldown`, and calls are refused
//! without trying. When the cooldown has passed one call is let through
//! (half-open): success closes the breaker, failure opens it again for
//! another cooldown.

use std::time::{Duration, Instant};

pub struct Breaker {
    threshold: u32,
    cooldown: Duration,
    failures: u32,
    open_until: Option<Instant>,
}

pub enum State {
    Closed,
    /// Refusing calls for this much longer.
    Open(Duration),
    /// Cooldown over; the next call is a probe.
    HalfOpen,
}

impl Breaker {
    pub fn new(threshold: u32, cooldown: Duration) -> Self {
        Self { threshold, cooldown, failures: 0, open_until: None }
    }

    pub fn state(&self) -> State {
        match self.open_until {
            None => State::Closed,
            Some(until) => match until.checked_duration_since(Instant::now()) {
                Some(left) if !left.is_zero() => State::Open(left),
                _ => State::HalfOpen,
            },
        }
    }

    pub fn failures(&self) -> u32 {
        self.failures
    }

    pub fn success(&mut self) {
        self.failures = 0;
        self.open_until = None;
    }

    /// Count a failure. Returns true if this failure (re)opened the breaker.
    pub fn failure(&mut self) -> bool {
        self.failures += 1;
        if self.failures >= self.threshold {
            self.open_until = Some(Instant::now() + self.cooldown);
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opens_after_threshold_and_closes_on_success() {
        let mut b = Breaker::new(3, Duration::from_secs(60));
        assert!(!b.failure());
        assert!(!b.failure());
        assert!(matches!(b.state(), State::Closed));
        assert!(b.failure());
        assert!(matches!(b.state(), State::Open(_)));
        b.success();
        assert!(matches!(b.state(), State::Closed));
        assert_eq!(b.failures(), 0);
    }

    #[test]
    fn half_open_after_cooldown_and_reopens_on_failure() {
        let mut b = Breaker::new(1, Duration::ZERO);
        assert!(b.failure());
        assert!(matches!(b.state(), State::HalfOpen));
        assert!(b.failure()); // probe failed: open again
    }
}
