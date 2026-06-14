# Local TTS Candidate Scan - 2026-06-14

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

## Watch List

| Candidate | Hugging Face / project | Notes |
| --- | --- | --- |
| Higgs Audio v3 TTS 4B | `bosonai/higgs-audio-v3-tts-4b` | Recently updated and high-interest; needs capability verification. |
| ZONOS2 | `Zyphra/ZONOS2` | Very recent HF activity; needs local capability/install check. |
| Supertone supertonic-3 | `Supertone/supertonic-3` | Current HF activity; verify whether local/open use fits. |
| LongCat-AudioDiT-3.5B | various mirrors/derivatives | Interesting claims, but needs source/license validation before serious work. |

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
