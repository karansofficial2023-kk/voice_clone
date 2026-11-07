import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PRETRAINED_DIR = os.path.join(ROOT_DIR, "pretrained_models")
os.makedirs(PRETRAINED_DIR, exist_ok=True)

print("=" * 65)
print("  CosyVoice 2 & Dhee Pravega-TTS - 1-Click Model Downloader")
print("=" * 65)

# 1. Download CosyVoice 2-0.5B (Expressive English / Multilingual Engine)
cosy2_dir = os.path.join(PRETRAINED_DIR, "CosyVoice2-0.5B")
print(f"\n[1/2] Checking / Downloading CosyVoice 2-0.5B to: {cosy2_dir}...")
if os.path.exists(os.path.join(cosy2_dir, "llm.pt")) and os.path.exists(os.path.join(cosy2_dir, "flow.pt")):
    print("✅ CosyVoice 2-0.5B is already downloaded!")
else:
    t0 = time.time()
    try:
        from modelscope import snapshot_download as ms_download
        print("Downloading from ModelScope (iic/CosyVoice2-0.5B)...")
        ms_download('iic/CosyVoice2-0.5B', local_dir=cosy2_dir)
        print(f"✅ Successfully downloaded CosyVoice 2-0.5B in {time.time()-t0:.1f}s!")
    except Exception as e1:
        print(f"ModelScope download failed: {e1}. Falling back to HuggingFace...")
        from huggingface_hub import snapshot_download as hf_download
        hf_download(repo_id="FunAudioLLM/CosyVoice2-0.5B", local_dir=cosy2_dir, local_dir_use_symlinks=False, resume_download=True)
        print(f"✅ Successfully downloaded CosyVoice 2-0.5B from HuggingFace in {time.time()-t0:.1f}s!")

# 2. Download Dhee Pravega-TTS (Native Indic Speech Engine)
pravega_dir = os.path.join(PRETRAINED_DIR, "dhee-pravega-tts")
print(f"\n[2/2] Checking / Downloading Dhee Pravega-TTS to: {pravega_dir}...")
if os.path.exists(os.path.join(pravega_dir, "llm.pt")) and os.path.exists(os.path.join(pravega_dir, "flow.pt")):
    print("✅ Dhee Pravega-TTS is already downloaded!")
else:
    t0 = time.time()
    from huggingface_hub import snapshot_download as hf_download
    print("Downloading from HuggingFace (dheeyantra/dhee-pravega-tts)...")
    hf_download(
        repo_id="dheeyantra/dhee-pravega-tts",
        local_dir=pravega_dir,
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print(f"✅ Successfully downloaded Dhee Pravega-TTS in {time.time()-t0:.1f}s!")

print("\n" + "=" * 65)
print("🎉 ALL MODELS ARE INSTALLED AND READY!")
print("Launch WebUI with: python webui.py --port 50000")
print("=" * 65)

