import argparse
import os
import re
import traceback
import time
import torch
import copy

from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor
from transformers.utils import logging
import uuid

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


class VoiceMapper:
    """Maps speaker names to voice file paths"""
    
    def __init__(self):
        self.setup_voice_presets()

        # change name according to our preset voice file
        new_dict = {}
        for name, path in self.voice_presets.items():
            
            if '_' in name:
                name = name.split('_')[0]
            
            if '-' in name:
                name = name.split('-')[-1]

            new_dict[name] = path
        self.voice_presets.update(new_dict)
        # print(list(self.voice_presets.keys()))

    def setup_voice_presets(self):
        """Setup voice presets by scanning the voices directory."""
        voices_dir = os.path.join(os.path.dirname(__file__), "voices/streaming_model")
        
        # Check if voices directory exists
        if not os.path.exists(voices_dir):
            print(f"Warning: Voices directory not found at {voices_dir}")
            self.voice_presets = {}
            self.available_voices = {}
            return
        
        # Scan for all VOICE files in the voices directory
        self.voice_presets = {}
        
        # Get all .pt files in the voices directory
        pt_files = [f for f in os.listdir(voices_dir) 
                    if f.lower().endswith('.pt') and os.path.isfile(os.path.join(voices_dir, f))]
        
        # Create dictionary with filename (without extension) as key
        for pt_file in pt_files:
            # Remove .pt extension to get the name
            name = os.path.splitext(pt_file)[0]
            # Create full path
            full_path = os.path.join(voices_dir, pt_file)
            self.voice_presets[name] = full_path
        
        # Sort the voice presets alphabetically by name for better UI
        self.voice_presets = dict(sorted(self.voice_presets.items()))
        
        # Filter out voices that don't exist (this is now redundant but kept for safety)
        self.available_voices = {
            name: path for name, path in self.voice_presets.items()
            if os.path.exists(path)
        }
        
        print(f"Found {len(self.available_voices)} voice files in {voices_dir}")
        print(f"Available voices: {', '.join(self.available_voices.keys())}")

    def get_voice_path(self, speaker_name: str) -> str:
        """Get voice file path for a given speaker name"""
        # First try exact match
        if speaker_name in self.voice_presets:
            return self.voice_presets[speaker_name]
        
        # Try partial matching (case insensitive)
        speaker_lower = speaker_name.lower()
        for preset_name, path in self.voice_presets.items():
            if preset_name.lower() in speaker_lower or speaker_lower in preset_name.lower():
                return path
        
        # Default to first voice if no match found
        default_voice = list(self.voice_presets.values())[0]
        print(f"Warning: No voice preset found for '{speaker_name}', using default voice: {default_voice}")
        return default_voice


import json
from aio_pika.abc import AbstractIncomingMessage
async def message_on_callback(message:AbstractIncomingMessage):
    payload = json.loads(message.body)
    model_path = "microsoft/VibeVoice-Realtime-0.5B"
    speaker_name = "Wayne"
    cfg_scale = 1.5
    logger.info(f"Payload: {payload}")
    transcript = payload['transcript']
    device="cuda"
            
    processor = VibeVoiceStreamingProcessor.from_pretrained(model_path)


    # Initialize voice mapper
    voice_mapper = VoiceMapper()
    
    # Check if txt file exists
    
    
    print(f"Loading processor & model from {model_path}")
    processor = VibeVoiceStreamingProcessor.from_pretrained(model_path)

    
    load_dtype = torch.bfloat16
    attn_impl_primary = "flash_attention_2"
    print(f"Using device: cuda, torch_dtype: {load_dtype}, attn_implementation: {attn_impl_primary}")
    # Load model with device-specific logic
    
    model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        model_path,
        torch_dtype=load_dtype,
        device_map="cuda",
        attn_implementation=attn_impl_primary,
    )
       
    


    model.eval()
    model.set_ddpm_inference_steps(num_steps=5)

    if hasattr(model.model, 'language_model'):
       print(f"Language model attention: {model.model.language_model.config._attn_implementation}")
    
    target_device = "cuda"
    voice_sample = voice_mapper.get_voice_path(speaker_name)
    all_prefilled_outputs = torch.load(voice_sample, map_location=target_device, weights_only=False)

    

    try:
        async with message.process(ignore_processed=True):
            # Prepare inputs for the model
            full_transcript = transcript.replace("’", "'").replace('“', '"').replace('”', '"')
            
            inputs = processor.process_input_with_cached_prompt(
                text=full_transcript,
                cached_prompt=all_prefilled_outputs,
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )

            # Move tensors to target device
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    inputs[k] = v.to(target_device)

            print(f"Starting generation with cfg_scale: {cfg_scale}")

            # Generate audio
            start_time = time.time()
            outputs = model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=cfg_scale,
                tokenizer=processor.tokenizer,
                generation_config={'do_sample': False},
                verbose=True,
                all_prefilled_outputs=copy.deepcopy(all_prefilled_outputs) if all_prefilled_outputs is not None else None,
            )
            generation_time = time.time() - start_time
            print(f"Generation time: {generation_time:.2f} seconds")
            
            # Calculate audio duration and additional metrics
            if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
                # Assuming 24kHz sample rate (common for speech synthesis)
                sample_rate = 24000
                audio_samples = outputs.speech_outputs[0].shape[-1] if len(outputs.speech_outputs[0].shape) > 0 else len(outputs.speech_outputs[0])
                audio_duration = audio_samples / sample_rate
                rtf = generation_time / audio_duration if audio_duration > 0 else float('inf')
                
                print(f"Generated audio duration: {audio_duration:.2f} seconds")
                print(f"RTF (Real Time Factor): {rtf:.2f}x")
            else:
                print("No audio output generated")
            
            # Calculate token metrics
            input_tokens = inputs['tts_text_ids'].shape[1]  # Number of input tokens
            output_tokens = outputs.sequences.shape[1]  # Total tokens (input + generated)
            generated_tokens = output_tokens - input_tokens - all_prefilled_outputs['tts_lm']['last_hidden_state'].size(1)
            
            print(f"Prefilling text tokens: {input_tokens}")
            print(f"Generated speech tokens: {generated_tokens}")
            print(f"Total tokens: {output_tokens}")

            # Save output (processor handles device internally)
            output_path = os.path.join(os.getcwd(), f"{uuid.uuid4()}_generated.wav")
            
            processor.save_audio(
                outputs.speech_outputs[0], # First (and only) batch item
                output_path=output_path,
            )
            print(f"Saved output to {output_path}")
            
            # Print summary
            print("\n" + "="*50)
            print("GENERATION SUMMARY")
            print("="*50)
            print(f"Input Transcript: {transcript}")
            print(f"Output file: {output_path}")
            print(f"Speaker names: {speaker_name}")
            print(f"Prefilling text tokens: {input_tokens}")
            print(f"Generated speech tokens: {generated_tokens}")
            print(f"Total tokens: {output_tokens}")
            print(f"Generation time: {generation_time:.2f} seconds")
            print(f"Audio duration: {audio_duration:.2f} seconds")
            print(f"RTF (Real Time Factor): {rtf:.2f}x")
            
            print("="*50)
            print(f"Using device: {device}")
            
            # Save data to csv, if csv not exist, create it
            import pandas as pd
            csv_path = "generation_metrics.csv"
            if not os.path.exists(csv_path):
                df = pd.DataFrame(columns=[
                    "transcript", "speaker_name", "input_tokens", "generated_tokens",
                    "total_tokens", "generation_time", "audio_duration", "rtf", "output_path"
                ])
                df.to_csv(csv_path, index=False)
            
    except Exception as e:
        logger.error(f"Internal service error: {e}",)
        traceback.print_exc()

from api.rabbitmq import RabbitMQHandler
import asyncio
async def main():
    rabbit_listener = RabbitMQHandler(
        endpoint="armadillo.rmq.cloudamqp.com",
        port=5672,
        user="ldzcjxub",
        password="3_Z8V9skHWW-Q7V9Krj4c0tOt2ueZ1UM",
        vhost="ldzcjxub"
    )
    
    await rabbit_listener.start_listener("VIBEVOICE_MAP", callback_function=message_on_callback)


if __name__ == "__main__":
    asyncio.run(main())
