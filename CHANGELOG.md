# Changelog

## Unreleased

- Add an independently owned, non-streaming conversation pipeline with guards before speech synthesis/audio enqueue, tool dispatch, and motion dispatch.
- Interrupt pending work on local hard stop; hold or block stops the rest of a turn.
- Keep effectful tools and motion disabled by default pending trusted adapters and hardware validation.
- Test pre-output ordering, failure paths, stop interruption, wheel contents, and isolated wheel import.
