import os
import sys
import argparse
import torchaudio
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'third_party', 'Matcha-TTS'))

from cosyvoice.cli.cosyvoice import AutoModel

def main():
    parser = argparse.ArgumentParser(description="CosyVoice 2 Fast Voice Cloner")
    parser.add_argument("--audio", required=True, help="Path to reference audio (WAV or MP3, 3-20s)")
    parser.add_argument("--prompt", required=True, help="Transcript of what is spoken in the reference audio")
    parser.add_argument("--text", required=True, help="New text to synthesize in the cloned voice")
    parser.add_argument("--output", default="cloned_output.wav", help="Output WAV path")
    parser.add_argument("--speed", type=float, default=1.0, help="Speaking speed (default: 1.0)")
    args = parser.parse_args()

    print(f"Loading CosyVoice 2 model from {os.path.join(ROOT_DIR, 'pretrained_models', 'CosyVoice2-0.5B')} ...")
    cosyvoice = AutoModel(model_dir=os.path.join(ROOT_DIR, 'pretrained_models', 'CosyVoice2-0.5B'))

    print(f"Reference Audio: {args.audio}")
    print(f"Reference Text:  {args.prompt}")
    print(f"Synthesis Text:  {args.text}")
    print("Generating speech...")

    chunks = []
    for chunk in cosyvoice.inference_zero_shot(args.text, args.prompt, args.audio, speed=args.speed):
        chunks.append(chunk['tts_speech'])

    if chunks:
        full_audio = torch.cat(chunks, dim=1)
        torchaudio.save(args.output, full_audio, cosyvoice.sample_rate)
        duration = full_audio.shape[1] / cosyvoice.sample_rate
        print(f"SUCCESS! Cloned speech saved to: {args.output}")
        print(f"Duration: {duration:.2f} seconds | Sample Rate: {cosyvoice.sample_rate} Hz")
    else:
        print("ERROR: Generation failed, no audio chunks were produced.")

if __name__ == "__main__":
    main()

