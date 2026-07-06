# [DRAFT — 待用户决定是否提交到 pollen-robotics/reachy_mini]

**Title:** Daemon crashes intermittently (exit code 116) — every `no_media` client forces a full media-pipeline release/rebuild cycle

## Environment

- Robot: Reachy Mini **Lite** (USB), motors on COM3
- OS: Windows 11 Home (10.0.26200)
- `reachy_mini` 1.7.3 (installed via the official "Reachy Mini Control" desktop bootstrapper venv)
- Daemon launched with defaults: `reachy-mini-daemon.exe` (localhost-only, wake-up-on-start)

## Symptom

The daemon process dies intermittently with **exit code 116**, typically after it has been
running for a while (~45+ min) and has gone through several media release/re-acquire
cycles. The last log line before one historical crash was `Re-acquiring media hardware...`;
clients then see `ConnectionReset` on `/api/media/acquire`.

We also observed a **second failure mode**: after force-killing a previous daemon instance
and immediately starting a new one, the new daemon logs `Daemon started successfully`,
then dies silently (~1 min later, exit 116, no error lines in the log) with **no clients
connected at all**. Waiting a few seconds between kill and restart avoids this — it looks
like a race on WASAPI device release.

## Root-cause analysis (what we found reading the SDK)

1. **Every `media_backend="no_media"` client triggers a daemon-side media teardown/rebuild.**
   In `reachy_mini.py` (`_configure_mediamanager`, around line 288), a client that asks for
   `no_media` calls `self.release_media()` — i.e. it tells the daemon to stop its
   `GstMediaServer` ("so camera/mic are available for direct access"), even if the client
   never intends to touch the camera or microphone (motion-only scripts).
   On `__exit__` (around line 181) the client calls `acquire_media()`, and
   `GstMediaServer.start()` **rebuilds the entire pipeline from scratch** (camera source,
   WASAPI elements, `webrtcsink`, embedded Rust signalling).

   So a workflow of small motion-only scripts (very common during development) puts the
   daemon through a continuous stop/rebuild churn it was probably never designed for.

2. **The rebuild path is visibly fragile.** After ~38 release/re-acquire cycles in one
   daemon instance (2 h uptime) we logged:

   ```
   reachy_mini.media.media_server - ERROR - Error: gst-stream-error-quark: GStreamer
   encountered a general stream error. (1) net\webrtc\src\webrtcsink\imp.rs(2491):
   gstrswebrtc::webrtcsink::imp::BaseWebRTCSink::connect_signaller::{{closure}}::{{closure}} ():
   /GstPipeline:reachymini_webrtc_sender/GstWebRTCSink:webrtcsink24:
   ```

   Note the element name `webrtcsink24` — each rebuild creates a new sink element. The
   crash itself is probabilistic: a scripted 30-cycle connect/disconnect loop on an aged
   instance did *not* crash it, but multiple real-world sessions did (historically after
   ~45 min + several cycles).

## Suggestions

1. `no_media` clients should **not** force the daemon to release/rebuild its media
   pipeline by default — motion-only clients don't need the camera/mic at all. An explicit
   opt-in (`release_media=True` or calling `release_media()` manually) would preserve the
   direct-access use case without the churn. (Our current workaround is monkey-patching
   `ReachyMini.release_media` to a no-op for motion-only scripts — works, zero cycles.)
2. Harden `GstMediaServer.start()`/`stop()` against repeated cycles (error recovery /
   supervision instead of process death), or reuse the pipeline rather than rebuilding.
3. Serialize/guard daemon startup against a previous instance's WASAPI handles not yet
   released (failure mode 2 above).

Happy to provide full daemon logs or run instrumented builds on this machine.
