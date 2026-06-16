# Local TTS Candidate Scan - 2026-06-16

Goal: identify current local, state-of-the-art TTS candidates for voice quality
and voice cloning tests. This project intentionally excludes older baseline
engines unless we later need a historical comparison point.

## Priority 1

| Candidate | Hugging Face / project | Rationale |
| --- | --- | --- |
| Qwen3-TTS 1.7B CustomVoice | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | Mature current local candidate with custom voice, voice clone/design APIs, and strong adoption. |
| VoxCPM2 | `openbmb/VoxCPM2` | 2B tokenizer-free diffusion-autoregressive TTS, 30 languages, 48 kHz output, controllable cloning. |
| dots.tts soar | `rednote-hilab/dots.tts-soar` | 2B fully continuous AR TTS; local CLI/Python API, continuation voice cloning, 48 kHz output, Apache 2.0, and strong published cloning/naturalness claims. |
| IndexTTS2 | `IndexTeam/IndexTTS-2` | Expressive zero-shot cloning, emotion control, and duration control. |

## Priority 2

| Candidate | Hugging Face / project | Rationale |
| --- | --- | --- |
| Fish Speech S2 Pro | `fishaudio/s2-pro` | High-end expressive comparison model; verify license and local install constraints. |
| OmniVoice | `k2-fsa/OmniVoice` | New multilingual zero-shot model with voice cloning/design claims. |
| MOSS-TTS v1.5 | `OpenMOSS-Team/MOSS-TTS-v1.5` | Recent long-form/cloning candidate with pronunciation control claims. |
| dots.tts mf | `rednote-hilab/dots.tts-mf` | MeanFlow-distilled dots.tts variant for low-latency/few-step inference; compare against soar after quality baseline. |
| Fun-CosyVoice3 0.5B 2512 | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` | Current small streaming zero-shot cloning candidate with Apache 2.0 tag, strong adoption, multilingual support, and good benchmark positioning. |
| GLM-TTS | `zai-org/GLM-TTS` | MIT Chinese/English zero-shot cloning with emotion control, phoneme control, and streaming inference claims. |
| FireRedTTS2 | `FireRedTeam/FireRedTTS2` | Long-form streaming multi-speaker TTS with cross-lingual/code-switching zero-shot cloning claims and very low first-packet latency claim. |
| VoxCPM 0.5B | `openbmb/VoxCPM-0.5B` | Smaller Apache 2.0 VoxCPM model; worthwhile latency comparison against VoxCPM2, especially if choppiness/first-audio latency remains a concern. |

## Watch List

| Candidate | Hugging Face / project | Notes |
| --- | --- | --- |
| Higgs Audio v3 TTS 4B | `bosonai/higgs-audio-v3-tts-4b` | Recently updated and high-interest; needs capability verification. |
| ZONOS2 | `Zyphra/ZONOS2` | Very recent HF activity; needs local capability/install check. |
| Supertone supertonic-3 | `Supertone/supertonic-3` | Current HF activity; verify whether local/open use fits. |
| LongCat-AudioDiT-3.5B | various mirrors/derivatives | Interesting claims, but needs source/license validation before serious work. |
| Marco-Voice | `AIDC-AI/Marco-Voice` | Apache 2.0, CosyVoice-compatible, voice cloning plus emotion control; low adoption so treat as watch-list. |
| VibeVoice 1.5B | `microsoft/VibeVoice-1.5B` | Popular MIT long-form multi-speaker candidate; verify current runnable TTS/cloning code path because Microsoft temporarily removed TTS code after misuse reports. |
| VibeVoice Realtime 0.5B | `microsoft/VibeVoice-Realtime-0.5B` | Excellent realtime TTS adoption and latency claims, but current realtime variant is fixed/single-speaker rather than arbitrary WAV cloning. |
| MioTTS 0.4B / 2.6B | `Aratako/MioTTS-*` | English/Japanese zero-shot cloning family with strong speed claims; regional language scope but technically interesting. |
| Irodori-TTS 500M v3 | `Aratako/Irodori-TTS-500M-v3` | Recent Japanese zero-shot cloning model with emoji-based style/effect control. |
| VieNeu-TTS v2 Turbo | `pnnbao-ump/VieNeu-TTS-v2-Turbo` | Current Vietnamese/English instant zero-shot cloning candidate; regional but low-latency focused. |
| OpenF5 TTS Base | `mrfakename/OpenF5-TTS-Base` | Apache 2.0 F5-style alpha; lower priority because the model card says it trails official non-commercial F5-TTS. |
| Spark-TTS 0.5B | `HKUSTAudio/Spark-TTS-0.5B` | Zero-shot cloning, but license changed to CC BY-NC-SA; keep as a reference-only candidate. |

## AISpeechApp Backend Support Status

| Candidate | Support status | Current local readiness |
| --- | --- | --- |
| VoxCPM2 | GUI streaming + batch synthesis | Runnable locally; current best-integrated voice cloning path. |
| Qwen3-TTS CustomVoice | Batch synthesis | Runnable if `qwen_tts` package/model are installed. |
| dots.tts soar / mf | Batch synthesis | Runnable through the separate dots environment used for prior samples. |
| IndexTTS2 | Batch synthesis | Backend path exists; install/runtime remains fragile. |
| OmniVoice | Batch WAV/MP3 synthesis added 2026-06-16 | Metadata OK; local `.venv` currently needs `omnivoice` install and model download. Uses documented `OmniVoice.from_pretrained(...).generate(..., ref_audio, ref_text)` cloning API with `voice_refs/first_impression.wav` and `.txt`. |
| VibeVoice 1.5B | Experimental batch WAV/MP3 synthesis added 2026-06-16 | Metadata/imports OK; model not cached. Uses HF Transformers `text-to-speech` pipeline because official Microsoft repo says TTS code was removed. Treat arbitrary WAV cloning as unverified until the pipeline/API exposes a reference voice hook. |

Batch MP3 output is handled by writing a temporary WAV and encoding with
FFmpeg/libmp3lame. The unit suite verifies both WAV and MP3 output paths without
requiring heavyweight model downloads.

## 2026-06-16 Refresh Findings

| Candidate | Last modified on HF | Current disposition |
| --- | ---: | --- |
| Fun-CosyVoice3 0.5B 2512 | 2026-02-03 | Add as Priority 2. Strong current local candidate; official repo shows streaming support and benchmark table includes CosyVoice3, GLM-TTS, VoxCPM, IndexTTS2, etc. |
| GLM-TTS | 2026-01-12 | Add as Priority 2. Zero-shot cloning, streaming, emotion and phoneme controls; bilingual Chinese/English. |
| FireRedTTS2 | 2025-09-17 | Add as Priority 2. Less fresh than CosyVoice3/GLM, but its long-form/multi-speaker streaming use case is relevant. |
| VoxCPM 0.5B | 2025-09-19 | Add as Priority 2. Smaller sibling to VoxCPM2; useful if we need a faster/lighter VoxCPM baseline. |
| Marco-Voice | 2025-12-03 | Add to watch list. Apache 2.0 and emotion cloning, but low usage metrics. |
| VibeVoice 1.5B | 2026-01-22 | Add to watch list. Very high adoption and MIT, but verify live cloning path before spending setup time. |
| VibeVoice Realtime 0.5B | 2025-12-12 | Add to watch list as realtime TTS, not primary voice cloning. Its realtime model is currently single-speaker/fixed voice prompt oriented. |
| MioTTS 0.4B / 2.6B | 2026-02-10 | Add to watch list. Very fast English/Japanese cloning candidates. |
| Irodori-TTS 500M v3 | 2026-05-12 | Add to watch list. Very recent, but Japanese-only scope. |
| VieNeu-TTS v2 Turbo | 2026-04-01 | Add to watch list. Strong regional Vietnamese/English latency/cloning angle. |
| OpenF5 TTS Base | 2025-05-17 | Add low-priority watch. Permissive F5-style option, but explicitly alpha/inferior to official NC F5. |
| Spark-TTS 0.5B | 2025-03-07 | Reference only due to CC BY-NC-SA license. |

## dots.tts Must-Have Check

| Requirement | Current finding |
| --- | --- |
| Voice cloning | Yes. HF quick start documents continuation voice cloning with `--prompt-audio` and exact `--prompt-text`. |
| Local runnable | Yes. The model card documents local install from `git+https://github.com/rednote-hilab/dots.tts.git`, plus CLI and Python API usage. |
| License | Reported as Apache 2.0 in the technical report/release materials. Verify model card/license file during install before product use. |
| Stable Python API | Promising. HF quick start shows `from dots_tts.runtime import DotsTtsRuntime` and `DotsTtsRuntime.from_pretrained(...)`. Treat as new but testable. |
| Windows fit | Unknown. Install instructions use conda Python 3.10 and pinned constraints; our smoke should try a separate backend venv if needed. |
| Quality claim | Strong enough to test. Official material claims best average Seed-TTS-Eval performance and high speaker similarity across multilingual benchmarks. |
| Latency | `dots.tts-mf` is the few-step/low-latency variant; technical report claims low first-packet latency. Needs local timing. |

## Source Notes

- Hugging Face TTS model listing showed recent/high-activity models including
  `bosonai/higgs-audio-v3-tts-4b`, `Zyphra/ZONOS2`, `openbmb/VoxCPM2`,
  `OpenMOSS-Team/MOSS-TTS-v1.5`, `k2-fsa/OmniVoice`, Qwen3-TTS,
  Fish Speech S2 Pro, and IndexTTS2.
- vLLM-Omni TTS examples list current locally servable TTS pipelines such as
  Fish Speech S2 Pro, CosyVoice3, GLM-TTS, and Ming-omni-tts. These are useful
  references for later serving/runtime experiments, but this harness starts
  with direct local smoke tests.
- `dots.tts-soar` was promoted from watch list to priority 1 after verifying
  the model card supports the core requirements we care about.
- 2026-06-16 refresh added CosyVoice3, GLM-TTS, FireRedTTS2, VoxCPM 0.5B,
  Marco-Voice, VibeVoice, MioTTS, Irodori-TTS, VieNeu-TTS, OpenF5, and Spark-TTS
  to the candidate database with priority tiers.
