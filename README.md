# 🎙️ CosyVoice 2 & Dhee Pravega-TTS — Dual Engine Voice Cloning & Native Indic TTS

An advanced, production-ready speech synthesis and zero-shot voice cloning platform combining **CosyVoice 2.0** (Expressive Multilingual & English Voice Cloning with Natural Emotion Control) and **Dhee Pravega-TTS** (Direct Native Script Engine for 11+ Indian Languages including Tamil, Hindi, Telugu, Kannada, Malayalam, and Bengali).

---

## 🌟 Key Features

1. **Dual Speech Engines:**
   - **CosyVoice 2-0.5B**: SOTA expressive voice cloning, natural language emotion instructions (`[laughter]`, `[sigh]`, `[breath]`, `<strong>`), and cross-lingual synthesis.
   - **Dhee Pravega-TTS**: Native Unicode Indic TTS engine (CosyVoice 3 architecture fine-tuned by Dheeyantra Research Labs on Indic speech datasets). Synthesizes direct native scripts for **தமிழ் (Tamil)**, **हिन्दी (Hindi)**, **తెలుగు (Telugu)**, Kannada, Malayalam, Bengali, Gujarati, Punjabi, Marathi, Assamese, and Odia without transliteration.
2. **Pre-Loaded Emotional Cloned Voices Library (`Voices/`):**
   - 🌺 **Tamil**: `Tamil - Nila (ElevenLabs Emotional)`
   - 🦚 **Hindi**: `Hindi - Swara (ElevenLabs Emotional)`
   - 🌼 **Telugu**: `Telugu - Mouni (ElevenLabs Emotional)`
   - 🎙️ **English**: `English - Bangalore Lady Voice 1`, `English - Bangalore Voice 2`, `English - Neeraj Voice`, `English - Tripti (Kids Storyteller)`
3. **✨ Local Ollama AI Auto-Tagging:**
   - Single-click emotion detection and instruction formatting using local Ollama (`llama3.1:8b` or `qwen3:8b`).
4. **Adaptive Duration & Clean Ending Fix:**
   - Fixes the trailing hallucination/repetition bug in Indic speech by implementing adaptive subword ratio scaling (`min_token_text_ratio = 0.2`).
5. **Interactive WebUI (Gradio 5):**
   - Clean, modern dual-engine interface with automatic language routing, locked reference voice library, emotion presets, and 1-click synthesis.

---

## 🚀 Quick Start: Setup on a New System

### Hardware Requirements
- **OS:** Windows 10/11 (64-bit) or Linux (Ubuntu 20.04/22.04)
- **GPU:** NVIDIA GPU with at least 6 GB VRAM (RTX 3060, 4060, 3080, 4080, 4090, A100, etc.)
- **CUDA:** CUDA 12.1 or compatible NVIDIA driver
- **Disk Space:** ~15 GB free space (for conda env + models)

---

### Step 1: Clone the Repository (with Submodules)

```bash
git clone --recursive https://github.com/karansofficial2023-kk/voice_clone.git
cd voice_clone
```

> **Note:** The `--recursive` flag ensures `third_party/Matcha-TTS` is automatically cloned. If you cloned without `--recursive`, run:
> ```bash
> git submodule update --init --recursive
> ```

---

### Step 2: Environment Setup

#### Option A: Windows 1-Click Setup
Simply double-click `setup_new_pc.bat` (or run it in Command Prompt / PowerShell).
It will automatically:
1. Create the Conda environment `cosyvoice2` (Python 3.10)
2. Install PyTorch with CUDA 12.1
3. Install all required dependencies
4. Download the pretrained models automatically

#### Option B: Manual Setup (Windows & Linux)

1. **Install Conda / Miniforge:**
   If you don't have Conda, install [Miniforge](https://github.com/conda-forge/miniforge/releases) or Miniconda.

2. **Create and Activate Environment:**
   ```bash
   conda create -n cosyvoice2 python=3.10 -y
   conda activate cosyvoice2
   ```

3. **Install PyTorch with CUDA 12.1 Support:**
   ```bash
   pip install torch==2.3.1+cu121 torchaudio==2.3.1+cu121 torchvision==0.18.1+cu121 --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

### Step 3: Download Pretrained Models (1-Click)

Run the included automated model downloader:

```bash
python download_all_models.py
```

This will download:
- `pretrained_models/CosyVoice2-0.5B` (via ModelScope / HuggingFace)
- `pretrained_models/dhee-pravega-tts` (via HuggingFace `dheeyantra/dhee-pravega-tts`)

---

### Step 4 (Optional): Local Ollama for AI Auto-Tagging

For the `✨ AI Auto-Tag` feature in the WebUI:
1. Download and install [Ollama](https://ollama.ai/).
2. Pull the Llama 3.1 8B model:
   ```bash
   ollama pull llama3.1:8b
   ```
*(If Ollama is not installed, the WebUI will use the built-in heuristic emotion analyzer automatically).*

---

### Step 5: Launch the WebUI

```bash
python webui.py --port 50000
```
Or on Windows:
```cmd
run_webui.bat
```

Open your browser and navigate to:
**`http://localhost:50000`**

---

## 🎙️ Adding Your Own Custom Cloned Voice

To add a new voice to the library dropdown so it is always available:
1. Record or obtain a clean **5 to 15-second** reference audio clip (`.mp3` or `.wav`).
2. Write the exact transcript of what was spoken in that clip into a `.txt` file with the exact same base name.
3. Place both files in the `Voices/` directory:
   ```text
   Voices/
   ├── My_New_Speaker.mp3
   └── My_New_Speaker.txt
   ```
4. Click **🔄 Refresh Voices Library** in the WebUI. Your voice will instantly appear in the library dropdown!

---

## 🛠️ Project Structure

```text
voice_clone/
├── cosyvoice/                   # Core CosyVoice TTS library (patched for Indic & CV3)
│   ├── cli/
│   │   ├── cosyvoice.py         # Multi-model inference & <|endofprompt|> routing
│   │   ├── frontend.py          # Multilingual text normalization & tokenization
│   │   └── model.py             # CosyVoiceModel & CosyVoice3Model definitions
│   └── llm/
│       └── llm.py               # Text-to-speech token LLM with adaptive Indic ratio
├── Voices/                      # Cloned reference audio clips & transcripts
│   ├── Tamil - Nila (ElevenLabs Emotional).mp3 / .txt
│   ├── Hindi - Swara (ElevenLabs Emotional).mp3 / .txt
│   ├── Telugu - Mouni (ElevenLabs Emotional).mp3 / .txt
│   └── English Voices...
├── third_party/
│   └── Matcha-TTS/              # Flow matching vocoder submodule
├── download_all_models.py       # 1-Click downloader for both engines
├── webui.py                     # Enhanced Dual-Engine Gradio WebUI
├── setup_new_pc.bat             # 1-Click Windows setup script
├── run_webui.bat                # 1-Click Windows launcher script
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## 📜 Licences & Acknowledgements

- **CosyVoice 2 / 3**: Alibaba FunAudioLLM (Apache-2.0 License).
- **Dhee Pravega-TTS**: Developed by Dheeyantra Research Labs (Licensed for research use; parts of Indic dataset derived from IIT-Madras Indic TTS database).
- **Matcha-TTS**: Shivam Mehta (Apache-2.0 License).
