"""
Default implementation of content enrichment services.
"""

from typing import List, Dict, Any
import logging
from .transcript_service import ContentEnricher

logger = logging.getLogger(__name__)


class DefaultContentEnricher(ContentEnricher):
    """Default implementation of content enricher."""

    async def enrich(self, segments: List[Dict[str, Any]], attendees: List[str], audio_path: str = None) -> List[Dict[str, Any]]:
        """Enrich transcript content using Gemini, name extraction, and embeddings."""
        try:
            # Step 1: Gemini correction and inference
            enriched_segments = await self._apply_gemini_enrichment(segments, attendees)

            # Step 2: Name extraction (NER)
            enriched_segments = await self._apply_name_extraction(enriched_segments, attendees)

            # Step 3: Name correction with Gemini
            enriched_segments = await self._apply_name_correction(enriched_segments)

            # Step 4: Embedding extraction (if audio path provided)
            if audio_path:
                final_segments = await self._apply_embedding_extraction(enriched_segments, audio_path)
            else:
                final_segments = enriched_segments
                logger.warning("Audio path not provided, skipping embedding extraction")

            logger.info(f"Content enrichment completed: {len(final_segments)} segments")
            return final_segments

        except Exception as e:
            logger.error(f"Content enrichment failed: {e}")
            return segments  # Return original segments as fallback

    async def _apply_gemini_enrichment(self, segments: List[Dict[str, Any]], attendees: List[str]) -> List[Dict[str, Any]]:
        """Apply Gemini-based content correction and inference."""
        try:
            from app.pipeline.correct_and_infer import gemini_correct_and_infer_segments_async_v3

            attendee_id_map = {f"u_{attendee.lower()}": attendee for attendee in attendees}

            enriched = await gemini_correct_and_infer_segments_async_v3(
                segments=segments,
                attendee_list=attendees,
                term_list=[],  # Could be configurable
                attendee_id_map=attendee_id_map,
                enable_json_mode=True,
                debug_dir=None
            )

            logger.info(f"Gemini enrichment completed: {len(enriched)} segments")
            return enriched

        except Exception as e:
            logger.error(f"Gemini enrichment failed: {e}")
            return segments

    async def _apply_name_extraction(self, segments: List[Dict[str, Any]], attendees: List[str]) -> List[Dict[str, Any]]:
        """Apply name extraction (NER) to segments."""
        try:
            from app.pipeline.rule_based_name_extraction import apply_name_extraction_to_segments

            if not segments or not attendees:
                logger.warning("Skipping name extraction: no segments or attendees")
                return segments

            extracted = apply_name_extraction_to_segments(
                segments=segments,
                attendees=attendees,
                ner_model="Davlan/bert-base-multilingual-cased-ner-hrl",
                tau_ms=90000,
                threshold=0.4,
                spk_strong_thr=1.5,
                spk_margin=0.6,
                spk_min_dur_ms=3000
            )

            logger.info(f"Name extraction completed: {len(extracted)} segments")
            return extracted

        except Exception as e:
            logger.error(f"Name extraction failed: {e}")
            return segments

    async def _apply_name_correction(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply Gemini-based name correction to extracted names."""
        try:
            from app.pipeline.correct_and_infer import enrich_names_with_gemini

            if not segments:
                logger.warning("Skipping name correction: no segments")
                return segments

            corrected = await enrich_names_with_gemini(segments)

            logger.info(f"Name correction completed: {len(corrected)} segments")
            return corrected

        except Exception as e:
            logger.error(f"Name correction failed: {e}")
            return segments

    async def _apply_embedding_extraction(self, segments: List[Dict[str, Any]], audio_path: str) -> List[Dict[str, Any]]:
        """Extract speaker embeddings from audio segments."""
        try:
            from app.pipeline.embedding_extraction import extract_embeddings_async
            from app.core.model_registry import models

            if not segments:
                logger.warning("Skipping embedding extraction: no segments")
                return segments

            enriched = await extract_embeddings_async(
                segments=segments,
                audio_path=audio_path,
                inference_model=models.inference
            )

            segments_with_embeddings = sum(1 for seg in enriched if seg.get("embedding") is not None)
            logger.info(f"Embedding extraction completed: {segments_with_embeddings}/{len(enriched)} segments")
            return enriched

        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}")
            return segments