# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Liu Yue)
#               2026 Enhanced with Dhee Pravega-TTS Native Indic Engine
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import re
import argparse
import json
import requests
import gradio as gr
import numpy as np
import torch
import torchaudio
import random

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([ROOT_DIR, f'{ROOT_DIR}/third_party/Matcha-TTS'])
from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.common import set_all_random_seed

# Regex matching Indic scripts (Devanagari, Bengali, Gurmukhi, Gujarati, Odia, Tamil, Telugu, Kannada, Malayalam)
INDIC_PATTERN = re.compile(r'[\u0900-\u0D7F]')


def is_indic_request(tts_text: str, prompt_text: str = '', prompt_wav: str = '') -> bool:
    """Detects whether synthesis text, prompt text, or prompt audio is for an Indic language."""
    if tts_text and re.search(INDIC_PATTERN, tts_text):
        return True
    if prompt_text and re.search(INDIC_PATTERN, prompt_text):
        return True
    if prompt_wav:
        base_lower = os.path.basename(prompt_wav).lower()
        if any(lang in base_lower for lang in ['tamil', 'telugu', 'hindi', 'malayalam', 'kannada', 'bengali', 'marathi', 'gujarati', 'punjabi']):
            return True
    return False


inference_mode_list = ['Pretrained Voice', '3s Rapid Voice Clone', 'Cross-Lingual Clone', 'Natural Language Instruction']
instruct_dict = {
    'Pretrained Voice': '1. Select a pretrained speaker from the dropdown.\n2. Click the "Generate Audio" button.',
    '3s Rapid Voice Clone': '1. Upload or select reference audio (<= 30s, >= 16kHz).\n2. Enter or verify prompt text matching the audio.\n3. Click "Generate Audio".',
    'Cross-Lingual Clone': '1. Upload or select prompt audio (<= 30s, >= 16kHz).\n2. Enter synthesis text in any target language.\n3. Click "Generate Audio".',
    'Natural Language Instruction': '1. Select an emotion preset or enter emotion instruction (English voices).\n2. For Indic voices, select emotional reference audio (Nila, Swara, Mouni).\n3. Click "Generate Audio".'
}
stream_mode_list = [('No', False), ('Yes', True)]
prompt_sr = 16000

VOICES_DIR = os.path.join(ROOT_DIR, 'Voices')


def get_saved_voices():
    voices = {}
    if os.path.exists(VOICES_DIR):
        for fname in sorted(os.listdir(VOICES_DIR)):
            if fname.lower().endswith(('.mp3', '.wav', '.flac', '.m4a')):
                base_name = os.path.splitext(fname)[0]
                audio_path = os.path.join(VOICES_DIR, fname)
                txt_path = os.path.join(VOICES_DIR, base_name + '.txt')
                transcript = ""
                if os.path.exists(txt_path):
                    try:
                        with open(txt_path, 'r', encoding='utf-8') as f:
                            transcript = f.read().strip()
                    except Exception as e:
                        logging.warning(f"Failed to read {txt_path}: {e}")
                voices[base_name] = {
                    "audio_path": audio_path,
                    "transcript": transcript
                }
    return voices


def auto_tag_text_with_llm(text):
    if not text or text.strip() == '':
        gr.Warning("Please enter text in 'Text to Synthesize' first.")
        return text, ""

    ollama_url = "http://127.0.0.1:11434/api/generate"
    prompt = f"""You are an expert TTS audio director for CosyVoice 2.
Analyze the following text for speech synthesis:
Text: "{text}"

Tasks:
1. Detect dominant emotion: "Happy", "Sad", "Angry", "Soft Whisper", "Loud", or "Storyteller".
2. Select instruction prompt for CosyVoice 2:
   - Happy: "You are a helpful assistant. 请非常开心地说一句话。<|endofprompt|>"
   - Sad: "You are a helpful assistant. 请非常伤心地说一句话。<|endofprompt|>"
   - Angry: "You are a helpful assistant. 请非常生气地说一句话。<|endofprompt|>"
   - Soft Whisper: "You are a helpful assistant. Please say a sentence in a very soft voice.<|endofprompt|>"
   - Loud: "You are a helpful assistant. Please say a sentence as loudly as possible.<|endofprompt|>"
   - Storyteller: "You are a helpful assistant. Speak with a warm, gentle storytelling tone.<|endofprompt|>"
3. Enhance the text with natural word-level tags for expressive speech:
   - <strong>word</strong> around 1 to 3 key dramatic or emphatic words
   - [laughter] at places where the speaker is amused or smiling
   - [sigh] when expressing sadness, weariness, or relief
   - [breath] for a natural inhalation pause before key clauses
   CRITICAL: Keep the exact original text verbatim! Do NOT repeat, duplicate, or alter any words or sentences. Do not append tags at the very end.

Return JSON only:
{{
  "emotion": "...",
  "instruction": "...",
  "tagged_text": "..."
}}
"""
    for model_name in ["llama3.1:8b", "qwen3:8b"]:
        try:
            res = requests.post(ollama_url, json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "format": "json"
            }, timeout=20)
            if res.status_code == 200:
                result = json.loads(res.json().get('response', '{}'))
                tagged = result.get('tagged_text', text)
                instruction = result.get('instruction', '')
                emo = result.get('emotion', 'Detected')
                if instruction != '' and '<|endofprompt|>' not in instruction:
                    instruction += '<|endofprompt|>'
                gr.Info(f"✨ AI Auto-Tagged as [{emo}] using local Ollama ({model_name})!")
                return tagged, instruction
        except Exception as e:
            logging.warning(f"Ollama auto-tag with {model_name} failed: {e}")

    # Fallback heuristic
    tagged = text
    lower = text.lower()
    if any(w in lower for w in ['happy', 'glee', 'smile', 'joy', 'celebrat', 'won', 'exciting', 'danced']):
        instruction = "You are a helpful assistant. 请非常开心地说一句话。<|endofprompt|>"
        tagged = tagged.replace('!', '! [laughter] ')
        emo = "Happy"
    elif any(w in lower for w in ['sad', 'crying', 'lost', 'tears', 'alone', 'heartbroken', 'sorrow']):
        instruction = "You are a helpful assistant. 请非常伤心地说一句话。<|endofprompt|>"
        tagged = '[sigh] ' + tagged
        emo = "Sad"
    elif any(w in lower for w in ['angry', 'furious', 'hate', 'shouted', 'stop', 'strict']):
        instruction = "You are a helpful assistant. 请非常生气地说一句话。<|endofprompt|>"
        emo = "Angry"
    elif any(w in lower for w in ['soft', 'whisper', 'night', 'sleep', 'quiet', 'gentle']):
        instruction = "You are a helpful assistant. Please say a sentence in a very soft voice.<|endofprompt|>"
        emo = "Soft Whisper"
    else:
        instruction = "You are a helpful assistant. Speak with a warm, gentle storytelling tone.<|endofprompt|>"
        emo = "Storyteller"
    gr.Info(f"✨ Auto-tagged as [{emo}] using heuristic analyzer.")
    return tagged, instruction


def generate_seed():
    seed = random.randint(1, 100000000)
    return {
        "__type__": "update",
        "value": seed
    }


def change_instruction(mode_checkbox_group):
    return instruct_dict[mode_checkbox_group]


def generate_audio(tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                   seed, stream, speed):
    if prompt_wav_upload is not None:
        prompt_wav = prompt_wav_upload
    elif prompt_wav_record is not None:
        prompt_wav = prompt_wav_record
    else:
        prompt_wav = None

    # Determine whether this is an Indic request (Tamil, Telugu, Hindi, etc.)
    use_indic = is_indic_request(tts_text, prompt_text or '', prompt_wav or '') and cosyvoice_indic is not None
    active_cosyvoice = cosyvoice_indic if use_indic else cosyvoice_en

    if use_indic:
        logging.info("🇮🇳 Routing to Dhee Pravega-TTS (Native Indic Speech Engine)")
        gr.Info("🇮🇳 Synthesizing with Dhee Pravega-TTS (Direct Native Script Engine)")
    else:
        logging.info("🎙️ Routing to CosyVoice 2 (Expressive English / Multilingual Engine)")

    default_data = np.zeros(active_cosyvoice.sample_rate)

    if prompt_wav is not None:
        audio_info = torchaudio.info(prompt_wav)
        duration = audio_info.num_frames / audio_info.sample_rate
        if duration > 30.0:
            gr.Warning(f'Prompt audio is {duration:.1f} seconds long! Reference audio must be 30 seconds or shorter (5 to 15 seconds is ideal). Please trim your audio file and try again.')
            yield (active_cosyvoice.sample_rate, default_data)
            return
        if audio_info.sample_rate < prompt_sr:
            gr.Warning(f'Prompt audio sample rate {audio_info.sample_rate}Hz is below required {prompt_sr}Hz.')
            yield (active_cosyvoice.sample_rate, default_data)
            return

    # Sanitize formatting tags
    tts_text = re.sub(r'<br\s*/?>', ' ', tts_text, flags=re.IGNORECASE)
    tts_text = re.sub(r'</?(?:b|em)>', lambda m: '<strong>' if '/' not in m.group() else '</strong>', tts_text, flags=re.IGNORECASE)
    tts_text = re.sub(r'<(?!/?(?:strong|laughter)\b)[^>]+>', ' ', tts_text)
    if '<strong>' in tts_text and '</strong>' not in tts_text:
        tts_text = tts_text + '</strong>'
    tts_text = re.sub(r'[ \t]+', ' ', tts_text).strip()

    if use_indic:
        # For Native Indic speech engine (Dhee Pravega-TTS), strip special event tokens/tags
        # that disrupt native phonetic sequence and can cause trailing phonetic artifacts
        tts_text = re.sub(r'\[(?:laughter|sigh|breath|whisper|cough)\]', ' ', tts_text, flags=re.IGNORECASE)
        tts_text = re.sub(r'</?(?:strong|b|em|laughter|sigh|breath)>', ' ', tts_text, flags=re.IGNORECASE)
        tts_text = re.sub(r'<[^>]+>', ' ', tts_text)
        tts_text = re.sub(r'[ \t]+', ' ', tts_text).strip()

    # Format instruction if provided
    formatted_instruct = instruct_text.strip() if instruct_text else ""
    if formatted_instruct != "":
        if '<|endofprompt|>' not in formatted_instruct:
            formatted_instruct = formatted_instruct + '<|endofprompt|>'
        if not formatted_instruct.startswith('You are a helpful assistant') and not formatted_instruct.startswith('<|'):
            formatted_instruct = 'You are a helpful assistant. ' + formatted_instruct

    audio_list = []

    if mode_checkbox_group == 'Pretrained Voice':
        if instruct_text != '' or prompt_wav is not None or prompt_text != '':
            gr.Info('In Pretrained Voice mode, prompt audio, prompt text, and instruction text are ignored.')
        if sft_dropdown == '':
            gr.Warning('No pretrained speakers available.')
            yield (active_cosyvoice.sample_rate, default_data)
            return
        logging.info('Processing pretrained voice inference request')
        set_all_random_seed(seed)
        for i in cosyvoice_en.inference_sft(tts_text, sft_dropdown, stream=stream, speed=speed):
            speech = i['tts_speech'].numpy().flatten()
            audio_list.append(speech)
            if stream:
                yield (cosyvoice_en.sample_rate, speech)

    elif mode_checkbox_group == 'Cross-Lingual Clone':
        if instruct_text != '':
            gr.Info('In Cross-Lingual Clone mode, instruction text is ignored.')
        if prompt_wav is None:
            gr.Warning('Please provide prompt audio for Cross-Lingual Clone mode.')
            yield (active_cosyvoice.sample_rate, default_data)
            return
        logging.info('Processing cross-lingual voice clone request')
        set_all_random_seed(seed)
        for i in active_cosyvoice.inference_cross_lingual(tts_text, prompt_wav, stream=stream, speed=speed):
            speech = i['tts_speech'].numpy().flatten()
            audio_list.append(speech)
            if stream:
                yield (active_cosyvoice.sample_rate, speech)

    elif mode_checkbox_group == '3s Rapid Voice Clone':
        if prompt_wav is None:
            gr.Warning('Prompt audio is missing. Please select a voice from the library or upload audio.')
            yield (active_cosyvoice.sample_rate, default_data)
            return
        set_all_random_seed(seed)
        if formatted_instruct != "" and not use_indic:
            logging.info('Processing 3s voice clone WITH emotion instruction via instruct2')
            for i in active_cosyvoice.inference_instruct2(tts_text, formatted_instruct, prompt_wav, stream=stream, speed=speed):
                speech = i['tts_speech'].numpy().flatten()
                audio_list.append(speech)
                if stream:
                    yield (active_cosyvoice.sample_rate, speech)
        else:
            if not prompt_text or prompt_text.strip() == '':
                gr.Warning('Prompt transcript is empty. You can type or edit the words spoken in the prompt audio.')
                yield (active_cosyvoice.sample_rate, default_data)
                return
            logging.info('Processing standard 3s zero-shot voice clone request')
            for i in active_cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_wav, stream=stream, speed=speed):
                speech = i['tts_speech'].numpy().flatten()
                audio_list.append(speech)
                if stream:
                    yield (active_cosyvoice.sample_rate, speech)

    else:  # Natural Language Instruction
        set_all_random_seed(seed)
        if use_indic:
            if prompt_wav is None:
                gr.Warning('Please select an emotional reference voice from the library (Nila, Swara, Mouni).')
                yield (active_cosyvoice.sample_rate, default_data)
                return
            for i in active_cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_wav, stream=stream, speed=speed):
                speech = i['tts_speech'].numpy().flatten()
                audio_list.append(speech)
                if stream:
                    yield (active_cosyvoice.sample_rate, speech)
        else:
            if formatted_instruct == "":
                gr.Warning('Please enter an instruction prompt, select an emotion preset, or click ✨ AI Auto-Tag.')
                yield (active_cosyvoice.sample_rate, default_data)
                return
            if prompt_wav is None:
                default_prompt = os.path.join(ROOT_DIR, 'asset', 'zero_shot_prompt.wav')
                if os.path.exists(default_prompt):
                    prompt_wav = default_prompt
                else:
                    gr.Warning('Please upload prompt audio or select a voice from the library.')
                    yield (active_cosyvoice.sample_rate, default_data)
                    return
            logging.info('Processing natural language instruction voice clone request')
            for i in active_cosyvoice.inference_instruct2(tts_text, formatted_instruct, prompt_wav, stream=stream, speed=speed):
                speech = i['tts_speech'].numpy().flatten()
                audio_list.append(speech)
                if stream:
                    yield (active_cosyvoice.sample_rate, speech)

    if not stream and len(audio_list) > 0:
        yield (active_cosyvoice.sample_rate, np.concatenate(audio_list))


def main():
    with gr.Blocks(title="CosyVoice 2 & Dhee Pravega-TTS — Native Indic & Expressive Speech") as demo:
        gr.Markdown("### 🎙️ [CosyVoice 2 & Dhee Pravega-TTS](https://github.com/FunAudioLLM/CosyVoice) — Expressive Multilingual & Native Indic TTS")
        gr.Markdown("Supports direct native scripts for **தமிழ் (Tamil)**, **हिन्दी (Hindi)**, **తెలుగు (Telugu)**, Kannada, Malayalam, and Expressive **English** voice cloning.")

        with gr.Row():
            tts_text = gr.Textbox(
                label="Text to Synthesize (Supports English, தமிழ், हिन्दी, తెలుగు, etc. — Native Scripts Supported Directly)",
                lines=3,
                value="Welcome! Once upon a time, in a peaceful green forest, a little bird taught everyone the true value of kindness.",
                scale=4
            )
            with gr.Column(scale=1):
                auto_tag_btn = gr.Button("✨ AI Auto-Tag\n(Ollama)", variant="secondary", size="sm")

        with gr.Row():
            mode_checkbox_group = gr.Radio(choices=inference_mode_list, label='Inference Mode', value=inference_mode_list[0])
            instruction_text = gr.Text(label="Step-by-Step Instructions", value=instruct_dict[inference_mode_list[0]], scale=0.5)
            sft_dropdown = gr.Dropdown(choices=sft_spk, label='Pretrained Speaker (CosyVoice 2)', value=sft_spk[0], scale=0.25)
            stream = gr.Radio(choices=stream_mode_list, label='Streaming Mode', value=stream_mode_list[0][1])
            speed = gr.Number(value=1.0, label="Speaking Speed (Non-streaming only)", minimum=0.5, maximum=2.0, step=0.1)
            with gr.Column(scale=0.25):
                seed_button = gr.Button(value="\U0001F3B2 Randomize")
                seed = gr.Number(value=0, label="Inference Seed")

        saved_voices = get_saved_voices()
        saved_voice_names = ["(Custom Upload / Recording)"] + list(saved_voices.keys())

        with gr.Row():
            saved_voice_dropdown = gr.Dropdown(
                choices=saved_voice_names,
                value=saved_voice_names[0],
                label="🎙️ Cloned Voices Library (Select Tamil - Nila, Hindi - Swara, Telugu - Mouni, or English Voices to Auto-Load)",
                scale=0.75
            )
            refresh_voices_btn = gr.Button("🔄 Refresh Voices Library", scale=0.25)

        with gr.Row():
            prompt_wav_upload = gr.Audio(sources='upload', type='filepath', label='Upload Prompt Audio (WAV/MP3, >= 16kHz)')
            prompt_wav_record = gr.Audio(sources='microphone', type='filepath', label='Record Prompt Audio')
        prompt_text = gr.Textbox(
            label="Prompt Audio Transcript (What is spoken in the prompt audio)",
            lines=2,
            placeholder="Select a voice from the library (locked read-only) or choose Custom Upload to type...",
            value='',
            interactive=True
        )
        with gr.Row():
            emotion_preset = gr.Dropdown(
                choices=[
                    "Custom (Type below)",
                    "✨ Auto-Tag with Local Ollama (Word-Level Tags + Emotion)",
                    "✨ AI Auto-Tagged (Preset Frozen)",
                    "😊 Happy / Joyful",
                    "😢 Sad / Melancholic",
                    "😡 Angry / Furious",
                    "🤫 Soft / Gentle Storyteller Whisper",
                    "📢 Loud / Energetic",
                    "🎭 Sarcastic / Amused"
                ],
                label="Emotion Preset (Select to automatically fill instruction)",
                value="Custom (Type below)",
                interactive=True,
                scale=0.35
            )
            unlock_preset_btn = gr.Button("🔓 Unlock Preset", visible=False, scale=0.15, variant="secondary")
            instruct_text = gr.Textbox(
                label="Instruction / Emotion Prompt (or fine-grained tags like [laughter], [sigh], [breath] in text)",
                lines=1,
                placeholder="e.g. Speak with a warm, gentle, maternal storytelling tone, full of wonder and expressive emotion...",
                value='',
                interactive=True,
                scale=0.5
            )

        def on_auto_tag(text, current_mode):
            tagged, instr = auto_tag_text_with_llm(text)
            new_mode = current_mode if current_mode != 'Pretrained Voice' else '3s Rapid Voice Clone'
            return (
                tagged,
                instr,
                gr.update(value="✨ AI Auto-Tagged (Preset Frozen)", interactive=False),
                gr.update(visible=True),
                new_mode
            )

        auto_tag_btn.click(
            fn=on_auto_tag,
            inputs=[tts_text, mode_checkbox_group],
            outputs=[tts_text, instruct_text, emotion_preset, unlock_preset_btn, mode_checkbox_group]
        )

        def on_unlock_preset():
            return gr.update(value="Custom (Type below)", interactive=True), gr.update(visible=False)

        unlock_preset_btn.click(
            fn=on_unlock_preset,
            inputs=[],
            outputs=[emotion_preset, unlock_preset_btn]
        )

        def select_voice(choice, current_mode):
            if not choice or choice.startswith("(Custom"):
                return None, None, gr.update(value="", interactive=True, label="Prompt Audio Transcript (What is spoken in the prompt audio)"), current_mode
            voices = get_saved_voices()
            if choice in voices:
                v = voices[choice]
                new_mode = current_mode
                if current_mode == 'Pretrained Voice':
                    new_mode = '3s Rapid Voice Clone'
                return v["audio_path"], None, gr.update(value=v["transcript"], interactive=False, label=f"Prompt Audio Transcript (🔒 Locked / Non-Editable — {choice})"), new_mode
            return None, None, gr.update(value="", interactive=True, label="Prompt Audio Transcript (What is spoken in the prompt audio)"), current_mode

        def refresh_voices_list():
            voices = get_saved_voices()
            names = ["(Custom Upload / Recording)"] + list(voices.keys())
            return gr.update(choices=names, value=names[0])

        saved_voice_dropdown.change(
            fn=select_voice,
            inputs=[saved_voice_dropdown, mode_checkbox_group],
            outputs=[prompt_wav_upload, prompt_wav_record, prompt_text, mode_checkbox_group]
        )
        refresh_voices_btn.click(
            fn=refresh_voices_list,
            inputs=[],
            outputs=[saved_voice_dropdown]
        )

        preset_map = {
            "Custom (Type below)": "",
            "😊 Happy / Joyful": "You are a helpful assistant. 请非常开心地说一句话。<|endofprompt|>",
            "😢 Sad / Melancholic": "You are a helpful assistant. 请非常伤心地说一句话。<|endofprompt|>",
            "😡 Angry / Furious": "You are a helpful assistant. 请非常生气地说一句话。<|endofprompt|>",
            "🤫 Soft / Gentle Storyteller Whisper": "You are a helpful assistant. Please say a sentence in a very soft voice.<|endofprompt|>",
            "📢 Loud / Energetic": "You are a helpful assistant. Please say a sentence as loudly as possible.<|endofprompt|>",
            "🎭 Sarcastic / Amused": "You are a helpful assistant. Please speak with amusement and laughter.<|endofprompt|>"
        }
        def on_preset_change(choice, text, current_mode):
            new_mode = current_mode if current_mode != 'Pretrained Voice' else '3s Rapid Voice Clone'
            if choice == "✨ Auto-Tag with Local Ollama (Word-Level Tags + Emotion)":
                tagged, instr = auto_tag_text_with_llm(text)
                return (
                    tagged,
                    instr,
                    gr.update(value="✨ AI Auto-Tagged (Preset Frozen)", interactive=False),
                    gr.update(visible=True),
                    new_mode
                )
            elif choice == "✨ AI Auto-Tagged (Preset Frozen)":
                return text, gr.update(), gr.update(), gr.update(visible=True), current_mode
            elif choice == "Custom (Type below)":
                return text, "", gr.update(value=choice, interactive=True), gr.update(visible=False), current_mode
            else:
                instr = preset_map.get(choice, "")
                return text, instr, gr.update(value=choice, interactive=True), gr.update(visible=False), new_mode

        emotion_preset.change(
            fn=on_preset_change,
            inputs=[emotion_preset, tts_text, mode_checkbox_group],
            outputs=[tts_text, instruct_text, emotion_preset, unlock_preset_btn, mode_checkbox_group]
        )

        generate_button = gr.Button("Generate Audio", variant="primary")
        audio_output = gr.Audio(label="Synthesized Audio Output", autoplay=True)

        seed_button.click(generate_seed, inputs=[], outputs=seed)
        generate_button.click(
            generate_audio,
            inputs=[tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                    seed, stream, speed],
            outputs=[audio_output]
        )
        mode_checkbox_group.change(fn=change_instruction, inputs=[mode_checkbox_group], outputs=[instruction_text])
    demo.queue(max_size=20, default_concurrency_limit=1)
    demo.launch(server_name='0.0.0.0', server_port=args.port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=50000)
    parser.add_argument('--model_dir', type=str, default='pretrained_models/CosyVoice2-0.5B')
    parser.add_argument('--indic_model_dir', type=str, default='pretrained_models/dhee-pravega-tts')
    args = parser.parse_args()

    logging.info(f"Loading CosyVoice 2 from {args.model_dir}...")
    cosyvoice_en = AutoModel(model_dir=args.model_dir)

    logging.info(f"Loading Dhee Pravega-TTS from {args.indic_model_dir}...")
    try:
        cosyvoice_indic = AutoModel(model_dir=args.indic_model_dir)
        logging.info("✅ Successfully loaded Dhee Pravega-TTS (Native Indic Speech Engine)")
    except Exception as e:
        logging.warning(f"Could not load Indic model: {e}")
        cosyvoice_indic = None

    sft_spk = cosyvoice_en.list_available_spks()
    if len(sft_spk) == 0:
        sft_spk = ['']

    main()
