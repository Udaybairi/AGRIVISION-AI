"""
AGRIVISION AI - ElevenLabs High-Fidelity Voice Assistant Service
Provides ultra-realistic multilingual agricultural text-to-speech synthesis and voice management.
"""

import os
import re
import json
import hashlib
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Any, List, Optional
from backend.config import (
    ELEVENLABS_API_KEY,
    ELEVENLABS_VOICE_ID,
    ELEVENLABS_MODEL,
    DATA_DIR
)

# Audio cache directory
AUDIO_CACHE_DIR = DATA_DIR / "audio_cache"
AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Curated High-Fidelity Premade Voices
CURATED_VOICES = [
    {
        "id": "JBFqnCBsd6RMkjVDRZzb",
        "name": "George",
        "gender": "male",
        "accent": "warm_expert",
        "description": "Warm, experienced agronomist tone",
        "sample_text": "Hello farmer! I am your crop advisor."
    },
    {
        "id": "XB0fDUnXU5powFXDhCwa",
        "name": "Charlotte",
        "gender": "female",
        "accent": "calm_friendly",
        "description": "Clear, gentle agriculture specialist",
        "sample_text": "Inspect the leaf undersides for fungal spots."
    },
    {
        "id": "TX3LPaxmHKxFdv7VOQHJ",
        "name": "Liam",
        "gender": "male",
        "accent": "energetic_field",
        "description": "Clear, modern field advisor",
        "sample_text": "Apply copper oxychloride at three grams per liter."
    },
    {
        "id": "EXAVITQu4vr4xnSDxMaL",
        "name": "Sarah",
        "gender": "female",
        "accent": "professional_guide",
        "description": "Crisp and informative plant doctor",
        "sample_text": "Rotate your tomato crop with legumes to restore nitrogen."
    },
    {
        "id": "pNInz6obpgDQGcFmaJgB",
        "name": "Adam",
        "gender": "male",
        "accent": "authoritative",
        "description": "Deep and clear agricultural guide",
        "sample_text": "Follow safety instructions when applying sprays."
    }
]


class ElevenLabsVoiceService:
    """Production Text-to-Speech Engine for AgriMind."""

    def __init__(self, api_key: str = ELEVENLABS_API_KEY, default_voice_id: str = ELEVENLABS_VOICE_ID):
        self.api_key = api_key
        self.default_voice_id = default_voice_id
        self.default_model = ELEVENLABS_MODEL or "eleven_flash_v2_5"

    def clean_text_for_speech(self, text: str) -> str:
        """
        Cleans markdown syntax, citations, and table formatting to make spoken audio natural and fluent.
        """
        if not text:
            return ""

        # Remove markdown headers ##
        cleaned = re.sub(r'#{1,6}\s*', '', text)
        
        # Remove citation brackets [1], [2], [1][2], 【3】
        cleaned = re.sub(r'\[\d+\]', '', cleaned)
        cleaned = re.sub(r'【\d+】', '', cleaned)
        cleaned = re.sub(r'\[\d+†L\d+-L\d+\]', '', cleaned)

        # Remove bold/italic markers
        cleaned = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', cleaned)
        cleaned = re.sub(r'_{1,3}(.*?)_{1,3}', r'\1', cleaned)

        # Replace emojis with natural pauses or remove
        cleaned = re.sub(r'[🌱🔍🩺🛡️⚠️💊📚📝📌✓✕🖼️✨👨‍🌾🧪🔬🐛💬💡•]', '', cleaned)

        # Clean dashes and bullet markers
        cleaned = re.sub(r'^\s*[-*•\d.]+\s*', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'\s*—\s*', ', ', cleaned)

        # Normalize units for spoken phonetics
        cleaned = re.sub(r'\bg\s*L⁻¹\b|\bg/L\b|\bg\s*per\s*L\b', ' grams per liter ', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\bkg\s*ha⁻¹\b|\bkg/ha\b', ' kilograms per hectare ', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\bcm\b', ' centimeters ', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\bPHI\b', ' pre-harvest interval ', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\bKVK\b', ' Krishi Vigyan Kendra ', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\bIPM\b', ' Integrated Pest Management ', cleaned, flags=re.IGNORECASE)

        # Collapse excess whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        return cleaned

    def get_voices(self) -> List[Dict[str, Any]]:
        """Returns the curated high-quality voice list."""
        return CURATED_VOICES

    def synthesize_speech(
        self,
        text: str,
        voice_id: Optional[str] = None,
        model_id: Optional[str] = None,
        stability: float = 0.5,
        similarity_boost: float = 0.8
    ) -> bytes:
        """
        Synthesizes spoken audio from text using ElevenLabs API.
        Returns binary MP3 audio bytes.
        """
        if not self.api_key:
            raise ValueError("ElevenLabs API key is not configured.")

        voice = voice_id or self.default_voice_id or "JBFqnCBsd6RMkjVDRZzb"
        model = model_id or self.default_model or "eleven_flash_v2_5"
        
        cleaned_text = self.clean_text_for_speech(text)
        if not cleaned_text:
            raise ValueError("Text content is empty after normalization.")

        # Check Cache
        cache_key = hashlib.sha256(f"{voice}_{model}_{cleaned_text}".encode("utf-8")).hexdigest()
        cached_file = AUDIO_CACHE_DIR / f"{cache_key}.mp3"
        if cached_file.exists():
            try:
                return cached_file.read_bytes()
            except Exception:
                pass

        # ElevenLabs Text-to-Speech API Endpoint
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice}"
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg"
        }

        payload = {
            "text": cleaned_text,
            "model_id": model,
            "voice_settings": {
                "stability": stability,
                "similarity_boost": similarity_boost,
                "style": 0.0,
                "use_speaker_boost": True
            }
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                audio_data = response.read()
                
                # Save to cache
                try:
                    cached_file.write_bytes(audio_data)
                except Exception as e:
                    print(f"[ElevenLabs] Warning: Failed to save audio cache: {e}")
                
                return audio_data

        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="ignore")
            print(f"[ElevenLabs] HTTP Error {e.code}: {error_body}")
            raise RuntimeError(f"ElevenLabs TTS Error ({e.code}): {error_body}")
        except Exception as e:
            print(f"[ElevenLabs] Connection Error: {e}")
            raise RuntimeError(f"ElevenLabs Connection Failed: {e}")


# Global Singleton Instance
elevenlabs_service = ElevenLabsVoiceService()
