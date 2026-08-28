from scripts.release_train_7_soak import (
    QUALIFYING_DURATION_SECONDS,
    _sample_memory,
    qualification_status,
)


def test_soak_status_requires_full_monitored_two_hours() -> None:
    assert qualification_status(QUALIFYING_DURATION_SECONDS - 0.001) == (
        "NON_QUALIFYING_SHORT_SOAK"
    )
    assert qualification_status(QUALIFYING_DURATION_SECONDS) == (
        "PASS_TWO_HOUR_PACKAGED_SOAK"
    )


def test_soak_memory_summary_uses_first_last_and_peak() -> None:
    samples = [
        {"desktop_rss_kib": 100},
        {"desktop_rss_kib": None},
        {"desktop_rss_kib": 150},
        {"desktop_rss_kib": 120},
    ]
    assert _sample_memory(samples, "desktop_rss_kib") == {
        "start_kib": 100,
        "end_kib": 120,
        "peak_kib": 150,
    }
