# Changelog

## Unreleased

- Add an independently owned, non-streaming conversation pipeline with guards before speech synthesis/audio enqueue, tool dispatch, and motion dispatch.
- Interrupt pending work on local hard stop; hold or block stops the rest of a turn.
- Keep motion disabled by default; effectful tools require explicit registration and a separate owner-approval port for exact canonical arguments before dispatch.
- Test pre-output ordering, failure paths, stop interruption, wheel contents, and isolated wheel import.
- Add an offline ordering example and citation metadata.
- Add a bounded, opt-in Reachy Mini 1.10 media adapter with fake-backend tests; no robot validation yet.
- Bind motion proposals to the exact target seen by the guard, then validate conservative application caps in an opt-in SDK motion adapter; add a composite audio/motion stop port and fake-client tests.
- Add bounded owned microphone segmentation and optional local-only faster-whisper ASR adapter, with synthetic PCM/fake-model capture-to-output ordering tests; no live recognition or robot validation.
- Add a loopback-only, non-streaming Ollama proposal planner with strict typed JSON decoding, an async TypeSafe guard port, and an opt-in offline eSpeak/FFmpeg PCM synthesizer; fixture tests and CI exercise only software ordering, not robot behavior.
- Add a versioned atomic policy store, five synthetic dispatch-free policy-preview actions, and a privacy-minimized verdict ledger with legacy migration and JSONL export; carry numeric Jev judgments into ledger rows.
