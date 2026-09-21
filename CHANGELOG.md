# Changelog

## Unreleased

- Add an independently owned, non-streaming conversation pipeline with guards before speech synthesis/audio enqueue, tool dispatch, and motion dispatch.
- Interrupt pending work on local hard stop; hold or block stops the rest of a turn.
- Keep motion disabled by default; effectful tools require explicit registration and a separate owner-approval port for exact canonical arguments before dispatch.
- Test pre-output ordering, failure paths, stop interruption, wheel contents, and isolated wheel import.
- Add an offline ordering example and citation metadata.
- Add a bounded, opt-in Reachy Mini 1.10 media adapter with fake-backend tests; no robot validation yet.
