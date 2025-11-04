"""
AWS Voice Service for Chat Functionality

This module provides voice-to-text (Amazon Transcribe) and 
text-to-speech (Amazon Polly) capabilities for the chat interface.

Designed for AWS deployment with proper error handling and streaming support.
"""

import boto3
import json
import io
import time
import uuid
from typing import Optional, Dict, Tuple
from pathlib import Path
import tempfile

from config import config
try:
    from utils.logger import app_logger as logger
except ImportError:
    from utils.logger import logger


class VoiceService:
    """
    AWS Voice Service for speech-to-text and text-to-speech conversion.
    
    Uses:
    - Amazon Transcribe for speech-to-text (STT)
    - Amazon Polly for text-to-speech (TTS)
    """
    
    def __init__(self, region: Optional[str] = None):
        """
        Initialize AWS voice services.
        
        Args:
            region: AWS region (defaults to config.aws.region)
        """
        self.region = region or config.aws.region
        
        # Initialize clients
        try:
            self.transcribe_client = boto3.client('transcribe', region_name=self.region)
            self.polly_client = boto3.client('polly', region_name=self.region)
            logger.info(f"Voice services initialized for region: {self.region}")
        except Exception as e:
            logger.error(f"Failed to initialize voice services: {e}", exc_info=True)
            raise
    
    def transcribe_audio_file(
        self,
        audio_file_path: str,
        language_code: str = "en-US",
        media_format: str = "wav",
        enable_speaker_identification: bool = False
    ) -> Dict[str, any]:
        """
        Transcribe audio file using Amazon Transcribe (Batch mode).
        
        This is suitable for pre-recorded audio files.
        For real-time streaming, use transcribe_streaming() instead.
        
        Args:
            audio_file_path: Path to audio file (local or S3 URI)
            language_code: Language code (e.g., "en-US", "en-GB")
            media_format: Audio format ("wav", "mp3", "flac", etc.)
            enable_speaker_identification: Enable speaker diarization
        
        Returns:
            Dictionary with transcription results:
            {
                "text": str,  # Transcribed text
                "confidence": float,  # Average confidence score
                "job_name": str,  # Transcribe job name
                "status": str  # Job status
            }
        """
        job_name = f"transcribe-job-{uuid.uuid4().hex[:12]}"
        
        try:
            # If file is local, upload to S3 temporarily for Transcribe
            if not audio_file_path.startswith("s3://"):
                s3_uri = self._upload_to_s3_for_transcription(audio_file_path)
                # Delete local file immediately after upload to free space (no need to wait)
                try:
                    Path(audio_file_path).unlink()
                except:
                    pass
            else:
                s3_uri = audio_file_path
            
            # Start transcription job
            job_params = {
                "TranscriptionJobName": job_name,
                "Media": {'MediaFileUri': s3_uri},
                "MediaFormat": media_format,
                "LanguageCode": language_code
            }
            
            # Add settings only if speaker identification is enabled
            if enable_speaker_identification:
                job_params["Settings"] = {
                    "ShowSpeakerLabels": True,
                    "MaxSpeakerLabels": 2,
                    "ChannelIdentification": False
                }
            
            response = self.transcribe_client.start_transcription_job(**job_params)
            
            logger.info(f"Started transcription job: {job_name}")
            
            # Wait for job completion (poll more frequently for faster response)
            max_wait_time = 60
            poll_interval = 0.5  # Poll every 0.5 seconds for even faster response
            elapsed_time = 0
            
            while elapsed_time < max_wait_time:
                status = self.transcribe_client.get_transcription_job(
                    TranscriptionJobName=job_name
                )
                
                job_status = status['TranscriptionJob']['TranscriptionJobStatus']
                
                # Log progress for debugging
                if elapsed_time % 2 == 0:  # Log every 2 seconds
                    logger.debug(f"Transcription job {job_name} status: {job_status} (elapsed: {elapsed_time}s)")
                
                if job_status == 'COMPLETED':
                    # Get transcript URI
                    transcript_uri = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
                    
                    # Download and parse transcript
                    import requests
                    transcript_response = requests.get(transcript_uri)
                    transcript_data = transcript_response.json()
                    
                    # Extract text and confidence
                    transcript_text = transcript_data['results']['transcripts'][0]['transcript']
                    
                    # Calculate average confidence (optimized - limit processing for speed)
                    avg_confidence = 0.0
                    try:
                        items = transcript_data['results'].get('items', [])
                        if items:
                            confidences = []
                            # Limit to first 50 items for speed (confidence doesn't need full accuracy)
                            for item in items[:50]:
                                if item.get('type') == 'pronunciation':
                                    confidence = item.get('alternatives', [{}])[0].get('confidence', 0.0)
                                    try:
                                        confidences.append(float(confidence) if confidence is not None else 0.0)
                                    except (ValueError, TypeError):
                                        continue
                            
                            if confidences:
                                avg_confidence = sum(confidences) / len(confidences)
                    except Exception as e:
                        logger.debug(f"Confidence calculation skipped for speed: {e}")
                        avg_confidence = 0.0
                    
                    # Cleanup: Delete transcription job output (optional)
                    try:
                        # Note: The job output is in S3, you may want to delete it
                        pass
                    except:
                        pass
                    
                    return {
                        "text": transcript_text.strip(),
                        "confidence": avg_confidence,
                        "job_name": job_name,
                        "status": "COMPLETED"
                    }
                
                elif job_status == 'FAILED':
                    failure_reason = status['TranscriptionJob'].get('FailureReason', 'Unknown error')
                    logger.error(f"Transcription job failed: {failure_reason}")
                    raise Exception(f"Transcription failed: {failure_reason}")
                
                time.sleep(poll_interval)
                elapsed_time += poll_interval
            
            # Timeout
            raise Exception(f"Transcription job timed out after {max_wait_time} seconds")
            
        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            raise
    
    def transcribe_audio_bytes(
        self,
        audio_bytes: bytes,
        language_code: str = "en-US",
        media_format: str = "wav"
    ) -> Dict[str, any]:
        """
        Transcribe audio from bytes using Amazon Transcribe.
        
        Args:
            audio_bytes: Audio file as bytes
            language_code: Language code (e.g., "en-US")
            media_format: Audio format ("wav", "mp3", etc.)
        
        Returns:
            Dictionary with transcription results
        """
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{media_format}") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name
        
        try:
            return self.transcribe_audio_file(tmp_file_path, language_code, media_format)
        finally:
            # Cleanup temp file
            try:
                Path(tmp_file_path).unlink()
            except:
                pass
    
    def synthesize_speech(
        self,
        text: str,
        voice_id: str = "Joanna",
        output_format: str = "mp3",
        engine: str = "neural"
    ) -> bytes:
        """
        Convert text to speech using Amazon Polly.
        
        Args:
            text: Text to synthesize (max 3000 characters for neural engine)
            voice_id: Polly voice ID (e.g., "Joanna", "Matthew", "Ivy", "Justin")
            output_format: Output format ("mp3", "ogg_vorbis", "pcm", etc.)
            engine: TTS engine ("neural" for best quality, "standard" for compatibility)
        
        Returns:
            Audio bytes in the specified format
        """
        try:
            # Split long text into chunks (Polly has limits)
            max_chars = 3000 if engine == "neural" else 3000
            if len(text) > max_chars:
                # Split by sentences if possible
                import re
                sentences = re.split(r'([.!?]\s+)', text)
                chunks = []
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) <= max_chars:
                        current_chunk += sentence
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = sentence
                
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Synthesize each chunk and concatenate
                audio_parts = []
                for chunk in chunks:
                    response = self.polly_client.synthesize_speech(
                        Text=chunk,
                        OutputFormat=output_format,
                        VoiceId=voice_id,
                        Engine=engine
                    )
                    audio_parts.append(response['AudioStream'].read())
                
                return b''.join(audio_parts)
            else:
                # Single synthesis
                response = self.polly_client.synthesize_speech(
                    Text=text,
                    OutputFormat=output_format,
                    VoiceId=voice_id,
                    Engine=engine
                )
                return response['AudioStream'].read()
                
        except Exception as e:
            logger.error(f"Text-to-speech error: {e}", exc_info=True)
            raise
    
    def _upload_to_s3_for_transcription(self, local_file_path: str) -> str:
        """
        Upload audio file to S3 for Transcribe processing.
        
        Args:
            local_file_path: Path to local audio file
        
        Returns:
            S3 URI (s3://bucket/key)
        """
        s3_client = boto3.client('s3', region_name=self.region)
        bucket_name = config.aws.s3_bucket
        
        # Generate S3 key
        file_name = Path(local_file_path).name
        s3_key = f"transcribe-temp/{uuid.uuid4().hex}/{file_name}"
        
        # Upload file
        s3_client.upload_file(local_file_path, bucket_name, s3_key)
        
        s3_uri = f"s3://{bucket_name}/{s3_key}"
        logger.debug(f"Uploaded audio file to {s3_uri}")
        
        return s3_uri
    
    def get_available_voices(self, language_code: str = "en-US") -> list:
        """
        Get list of available Polly voices for a language.
        
        Args:
            language_code: Language code (e.g., "en-US")
        
        Returns:
            List of voice dictionaries
        """
        try:
            response = self.polly_client.describe_voices(LanguageCode=language_code)
            return response['Voices']
        except Exception as e:
            logger.error(f"Error fetching voices: {e}", exc_info=True)
            return []


def create_voice_service() -> Optional[VoiceService]:
    """
    Factory function to create VoiceService instance.
    Returns None if initialization fails (graceful degradation).
    """
    try:
        return VoiceService()
    except Exception as e:
        logger.warning(f"Voice service not available: {e}")
        return None

