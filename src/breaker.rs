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
        Self {
            threshold,
            cooldown,
            failures: 0,
            open_until: None,
        }
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

    #[test]
    fn threshold_one_opens_on_first_failure() {
        let mut b = Breaker::new(1, Duration::from_secs(60));
        assert!(matches!(b.state(), State::Closed));
        assert!(b.failure());
        assert!(matches!(b.state(), State::Open(_)));
        assert_eq!(b.failures(), 1);
    }

    #[test]
    fn failures_keep_counting_past_threshold() {
        let mut b = Breaker::new(2, Duration::from_secs(60));
        assert!(!b.failure());
        assert!(b.failure());
        // Every further failure reopens (and reports it) and is counted.
        assert!(b.failure());
        assert!(b.failure());
        assert_eq!(b.failures(), 4);
        assert!(matches!(b.state(), State::Open(_)));
    }

    #[test]
    fn success_after_half_open_closes() {
        let mut b = Breaker::new(2, Duration::ZERO);
        b.failure();
        b.failure();
        assert!(matches!(b.state(), State::HalfOpen));
        b.success();
        assert!(matches!(b.state(), State::Closed));
        assert_eq!(b.failures(), 0);
        // Closed again means a fresh count: one failure does not reopen.
        assert!(!b.failure());
        assert!(matches!(b.state(), State::Closed));
    }

    #[test]
    fn open_remaining_is_at_most_cooldown() {
        let cooldown = Duration::from_secs(300);
        let mut b = Breaker::new(1, cooldown);
        b.failure();
        match b.state() {
            State::Open(left) => {
                assert!(left <= cooldown);
                assert!(left > cooldown - Duration::from_secs(5));
            }
            _ => panic!("expected open"),
        }
    }

    #[test]
    fn reopening_restarts_the_cooldown() {
        let mut b = Breaker::new(1, Duration::from_millis(30));
        b.failure();
        std::thread::sleep(Duration::from_millis(40));
        assert!(matches!(b.state(), State::HalfOpen));
        assert!(b.failure());
        assert!(matches!(b.state(), State::Open(_)));
    }

    #[test]
    fn zero_threshold_opens_on_first_failure() {
        // Degenerate config: behaves like threshold 1.
        let mut b = Breaker::new(0, Duration::from_secs(60));
        assert!(matches!(b.state(), State::Closed));
        assert!(b.failure());
    }
}
