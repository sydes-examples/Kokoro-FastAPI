"""Tests for VoiceManager error messages."""

from unittest.mock import AsyncMock, patch

import pytest

from api.src.inference.voice_manager import VoiceManager


@pytest.mark.asyncio
async def test_load_voice_resolve_failure_names_the_voice():
    """A missing voice fails at path resolution, before any tensor load is attempted."""
    manager = VoiceManager()

    with patch.object(
        manager, "get_voice_path", AsyncMock(side_effect=RuntimeError("not found"))
    ):
        with pytest.raises(RuntimeError, match="Failed to resolve voice 'ghost'"):
            await manager.load_voice("ghost")


@pytest.mark.asyncio
async def test_load_voice_load_failure_names_the_resolved_path():
    """A voice that resolves but fails to load reports the path it tried to read."""
    manager = VoiceManager()

    with (
        patch.object(
            manager, "get_voice_path", AsyncMock(return_value="/voices/broken.pt")
        ),
        patch(
            "api.src.inference.voice_manager.paths.load_voice_tensor",
            AsyncMock(side_effect=OSError("corrupt file")),
        ),
    ):
        with pytest.raises(RuntimeError, match=r"from /voices/broken\.pt"):
            await manager.load_voice("broken")
