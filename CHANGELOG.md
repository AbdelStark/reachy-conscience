# Changelog

## Unreleased

- Keep one asyncio event loop for an operator-paced voice-host session, including sign actions and terminal stop; multi-turn and failure tests verify loop identity and cleanup.
- Expire read-only Reflex hints from the relay timestamp's remaining 1.5-second freshness budget, rather than granting a second full local window after delayed receipt.
- Add an opt-in, read-only Reflex event monitor with an exact loopback subscriber token, strict text-free schema/sequence/freshness checks, short-lived hints, and an operator `e` inspection command. No hint opens capture or drives output; fake WebSocket and actual Reflex-relay software tests are not joined-app or robot validation.
- Join a cancelled microphone task even when an external stop's output-halt request fails; a fake-port race test holds capture cleanup open to verify stop cannot return early. This does not establish physical silence.
- Add a self-checking no-robot owned voice-session walkthrough showing approved speech, pre-synthesis blocked output, and direct local stop across separate fake ports; CI runs it on both supported test Pythons.
- Add an opt-in, text-free loopback speaking assertion for Reflex after owned audio enqueue, with short heartbeats and operator-declared quiet; it is not an SDK playback receipt, guard, or hardware-validated turn-taking integration.
- Add a separately enabled operator-triggered sign response path: inbound sign guard, one speech-only local planner proposal, sign-instruction-following output judgment, then owned TTS/audio. Tool, motion, and hard-stop paths remain unavailable from signs; fixture tests do not establish live OCR or robot behavior.
- Add opt-in one-frame Reachy camera-sign screening with local Tesseract OCR, bounded JPEG/text/time limits, a source-labeled inbound guard record and `conscience.verdict@3` ledger row, and no planner/output path. Signs cannot trigger the speech hard-stop; synthetic JPEG and fake SDK tests do not validate on-robot camera or OCR quality.
- Enforce four-proposal batch and bounded planner/TTS waits at the owned pipeline boundary, independently of the concrete adapters; reject malformed batches before partial output and hold stalled operations without enqueueing audio.
- Recognize short direct, polite or wake-word-prefixed stop commands locally before Jev, while excluding negated, quoted, or embedded uses; fake-session tests cover the guarded-output bypass without claiming spoken-stop availability during playback.
- Require an opt-in motion-context provider with bounded collection and post-guard freshness checks; deterministically hold missing, stale, critical-battery, hot-motor, and larger-motion-near-person states before model or SDK dispatch. The voice host still has no live sensor provider and keeps motion disabled.
- Arm owned speaker playback lazily only when approved speech reaches its audio sink; blocked, ignored, and no-input turns never call `start_playing()`. Fence stop against in-flight arming/enqueue and request a follow-up halt after a race; fake-port and SDK-boundary tests only.
- Keep inbound route confidences in dedicated route metadata, not the ledger's top-three safety-judgment probabilities; add an export regression test.
- Batch inbound Choice routing with the guard judgment; make the operator-paced host fail closed on missing or uncertain routes, handle only ignore and stop in code, and hold unsupported look/quiet/sleep commands. Record validated route proposals in the text-minimized verdict ledger without transcript text; fixture tests only.
- Add a packaged 20-case self-authored spoken/sign-style red-team corpus, dispatch-free guard runner, verdict counts and observed p95, and an explicitly confirmed owner-console run; no live model results or safety claims.
- Wire the fixed-destination local-note tool into `conscience-voice` behind `--enable-local-notes`, with host-required guard confirmation and the same one-shot broker in the owner console; default remains tool-free and motion stays disabled.
- Hold uncertain audience, privacy, hostile-tone, request-match, and motion-safety judgments before any owned sink; block high-confidence hostile tone and snapshot policy across each Jev request and verdict.
- Add an explicitly operator-paced `conscience-voice` host that composes owned capture, local ASR/planner, Jev guard, offline TTS, SDK audio, policy console, and ledger. Tools/motion remain disabled, with no automatic next listen or hardware claim.
- Request an owned playback queue flush before a later microphone capture; a stop racing with that flush cannot reopen capture. The SDK still provides no playback-complete acknowledgement.
- Add an opt-in, fixed-destination local-note tool as a real effectful sink, with exact-argument broker integration tests for approval, denial, hard stop, and private-file checks; no robot or external service is involved.
- Request a pinned-SDK `clear_player()` queue flush before playback stop, fail closed when the flush API is missing, and test flush failure/stop ordering without a robot.
- Install the newly built wheel into a fresh isolated CI environment before its import smoke, avoiding reuse of an older same-version wheel cache.
- Add a loopback-only, bearer-authenticated owner policy and ledger console with a static browser shell, atomic policy writes, summary-free export, and an explicitly enabled, dispatch-free five-case preview; the standalone command connects to no robot or effectful tool.
- Add a one-shot owner-approval broker and authenticated browser review for exact tool arguments, with expiry, replay rejection, console-shutdown cancellation, and hard-stop invalidation; still no robot host.
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
- Add a one-turn owned voice lifecycle host that serializes capture, final ASR, playback arming, and the guarded pipeline, with an external stop path and fake-port race tests; no continuous listening or robot validation.
