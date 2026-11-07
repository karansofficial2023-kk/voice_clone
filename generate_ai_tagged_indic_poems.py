import os
import sys
import shutil
import time
from gradio_client import Client, handle_file

sys.stdout.reconfigure(encoding='utf-8')

client = Client("http://localhost:50000/")
print("Connected to CosyVoice 2 & Dhee Pravega-TTS WebUI on http://localhost:50000/")

downloads_dir = "C:/Users/User/Downloads"
artifacts_dir = "C:/Users/User/.gemini/antigravity/brain/011307c8-8cfd-4f2c-8fb0-7cc2a11b5a73"

os.makedirs(downloads_dir, exist_ok=True)
os.makedirs(artifacts_dir, exist_ok=True)

indic_poems = [
    {
        "lang": "Tamil",
        "voice": "Tamil - Nila (ElevenLabs Emotional)",
        "title": "வானில் சிறகடிக்கும் பறவைகள் (Soaring Birds & Dawn)",
        "raw_text": "வானில் சிறகடித்துப் பறக்கும் பறவைகளே, தென்றல் காற்றில் தவழும் இனிய கீதங்களே, காலையின் புதிய விடியலில் மலரும் பூக்களாய், நம் உள்ளத்தில் என்றும் இன்பம் பொங்கட்டும்!",
        "filename": "Tamil_AI_AutoTagged_Poem_Nila.wav",
        "seed": 101
    },
    {
        "lang": "Hindi",
        "voice": "Hindi - Swara (ElevenLabs Emotional)",
        "title": "बहती नदियां और गाती हवाएं (Flowing Rivers & New Hopes)",
        "raw_text": "बहती हुई नदियां और गाती हुई हवाएं, मन में जगाती हैं नई उमंग और आशाएं। हर सुबह लेकर आती है खुशियों का नया पैगाम, सुंदर बने हमारा हर एक पल और मुकाम!",
        "filename": "Hindi_AI_AutoTagged_Poem_Swara.wav",
        "seed": 202
    },
    {
        "lang": "Telugu",
        "voice": "Telugu - Mouni (ElevenLabs Emotional)",
        "title": "చిరుగాలి వీచగా పూదోట (Gentle Breeze & Blossoming Joy)",
        "raw_text": "చిరుగాలి వీచగా పూదోట మురిసెను, విరబూసిన పూలలో అందాలు చిందేను। మనసంతా నిండెను మధురమైన రాగం, ప్రతి క్షణం నింపును ఎనలేని ఆనందం!",
        "filename": "Telugu_AI_AutoTagged_Poem_Mouni.wav",
        "seed": 303
    }
]

print(f"Beginning AI Auto-Tagging & Audio Generation for {len(indic_poems)} Indic Voices...")
results = []
start_all = time.time()

for idx, p in enumerate(indic_poems, 1):
    t0 = time.time()
    print(f"\n===========================================================")
    print(f"[{idx}/{len(indic_poems)}] {p['lang']} | Voice: {p['voice']}")
    print(f"Original Poem:\n  {p['raw_text']}")
    
    # 1. Select Voice
    print(f"Selecting voice: {p['voice']}...")
    voice_res = client.predict(
        choice=p['voice'],
        current_mode="3s Rapid Voice Clone",
        api_name="/select_voice"
    )
    prompt_wav_path = voice_res[0]
    prompt_text_obj = voice_res[2]
    prompt_text = prompt_text_obj.get("value", "") if isinstance(prompt_text_obj, dict) else str(prompt_text_obj)
    mode = voice_res[3]
    
    # 2. AI Auto-Tag with local Ollama
    print("Calling /on_auto_tag (Ollama llama3.1:8b)...")
    tag_res = client.predict(
        text=p['raw_text'],
        current_mode="3s Rapid Voice Clone",
        api_name="/on_auto_tag"
    )
    tagged_text = tag_res[0]
    instruction = tag_res[1]
    preset_status = tag_res[2].get("value", "") if isinstance(tag_res[2], dict) else str(tag_res[2])
    print(f"✨ AI Auto-Tagged Text:\n  {tagged_text}")
    print(f"✨ AI Instruction:\n  {instruction}")
    
    # 3. Generate Audio
    print("Synthesizing with Dhee Pravega-TTS (Native Indic Speech Engine)...")
    audio_res = client.predict(
        tts_text=tagged_text,
        mode_checkbox_group=mode,
        sft_dropdown="",
        prompt_text=prompt_text,
        prompt_wav_upload=handle_file(prompt_wav_path),
        prompt_wav_record=None,
        instruct_text=instruction,
        seed=p['seed'],
        stream=False,
        speed=1.0,
        api_name="/generate_audio"
    )
    
    # 4. Save to Downloads and Artifacts
    download_path = os.path.join(downloads_dir, p['filename'])
    artifact_path = os.path.join(artifacts_dir, p['filename'])
    shutil.copy2(audio_res, download_path)
    shutil.copy2(audio_res, artifact_path)
    
    file_size_kb = os.path.getsize(download_path) / 1024
    elapsed = time.time() - t0
    print(f"✅ Generated in {elapsed:.2f}s | Size: {file_size_kb:.1f} KB")
    print(f"   📂 Downloads: {download_path}")
    print(f"   🔗 Artifact:  {artifact_path}")
    
    results.append({
        "lang": p['lang'],
        "voice": p['voice'],
        "title": p['title'],
        "raw_text": p['raw_text'],
        "tagged_text": tagged_text,
        "instruction": instruction,
        "download_path": download_path,
        "artifact_path": artifact_path,
        "file_size_kb": file_size_kb,
        "duration_sec": elapsed
    })

total_time = time.time() - start_all
print(f"\n===========================================================")
print(f"🎉 Successfully generated all {len(results)} AI Auto-Tagged Indic audios in {total_time:.1f}s!")
print(f"Output files saved to: {downloads_dir}")

