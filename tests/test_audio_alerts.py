import pytest
import os
import time
import base64
import hashlib
from unittest.mock import MagicMock, patch

import streamlit as st
from app import (
    FIRE_TTS_TEXT,
    CROWD_TTS_TEXT,
    trigger_hindi_alert,
    play_persistent_audio,
    silence_alerts,
    disarm_system,
)

def test_tts_constants_defined():
    assert "आग" in FIRE_TTS_TEXT
    assert "ओवरक्राउडिंग" in CROWD_TTS_TEXT

def test_trigger_hindi_alert_sets_session_state():
    # Setup dummy session state
    st.session_state.active_audio_b64 = None
    st.session_state.active_audio_id = 0
    st.session_state.audio_expire_ts = 0.0

    now_before = time.time()
    trigger_hindi_alert(FIRE_TTS_TEXT)
    now_after = time.time()

    assert st.session_state.active_audio_b64 is not None
    assert isinstance(st.session_state.active_audio_b64, str)
    assert st.session_state.active_audio_id == 1
    # Check that audio expiration timestamp is set ~10 seconds in future
    assert now_before + 9.0 <= st.session_state.audio_expire_ts <= now_after + 11.0

def test_silence_alerts_resets_audio():
    st.session_state.silenced_until = 0.0
    st.session_state.do_clear_alert = False
    st.session_state.active_alert = "FIRE"
    st.session_state.hazard_clear_ts = 100.0
    st.session_state.active_audio_b64 = "dummy_base64"
    st.session_state.audio_expire_ts = time.time() + 10.0

    silence_alerts()

    assert st.session_state.active_audio_b64 is None
    assert st.session_state.audio_expire_ts == 0.0
    assert st.session_state.active_alert is None

def test_disarm_system_resets_audio():
    st.session_state.system_armed = True
    st.session_state.do_clear_alert = False
    st.session_state.active_alert = "CROWD"
    st.session_state.hazard_clear_ts = 100.0
    st.session_state.silenced_until = 50.0
    st.session_state.active_audio_b64 = "dummy_base64"
    st.session_state.audio_expire_ts = time.time() + 10.0
    st.session_state.camera_worker = None

    disarm_system()

    assert st.session_state.system_armed is False
    assert st.session_state.active_audio_b64 is None
    assert st.session_state.audio_expire_ts == 0.0
    assert st.session_state.active_alert is None

def test_play_persistent_audio_generates_script():
    with patch("streamlit.components.v1.html") as mock_html:
        play_persistent_audio("test_b64", 42)
        mock_html.assert_called_once()
        args, kwargs = mock_html.call_args
        assert "audioId = 42" in args[0]
        assert "data:audio/mp3;base64,test_b64" in args[0]
        assert "w._hzAudio.onended" in args[0]
        assert "1200" in args[0]
        assert kwargs["height"] == 0
        assert kwargs["width"] == 0

def test_stop_persistent_audio_generates_stop_script():
    with patch("streamlit.components.v1.html") as mock_html:
        play_persistent_audio(None, 0, force=True)
        mock_html.assert_called_once()
        args, kwargs = mock_html.call_args
        assert "w._hzAudio.pause()" in args[0]
        assert "clearTimeout(w._hzLoopTimer)" in args[0]
        assert kwargs["height"] == 0
        assert kwargs["width"] == 0

def test_play_persistent_audio_skips_when_state_unchanged():
    with patch("streamlit.components.v1.html") as mock_html:
        st.session_state._last_rendered_audio_state = None
        st.session_state.stop_audio_id = 99
        # First call triggers render
        play_persistent_audio(None, 0)
        assert mock_html.call_count == 1

        # Second call with unchanged state MUST NOT call mock_html again!
        play_persistent_audio(None, 0)
        assert mock_html.call_count == 1

def test_play_persistent_audio_renders_on_state_transition():
    with patch("streamlit.components.v1.html") as mock_html:
        st.session_state._last_rendered_audio_state = None
        st.session_state.stop_audio_id = 1
        st.session_state.audio_trigger_id = 1

        # State 1: Stop
        play_persistent_audio(None, 0)
        assert mock_html.call_count == 1

        # State 2: Transition to Play
        st.session_state.audio_trigger_id = 2
        play_persistent_audio("sample_b64", 2)
        assert mock_html.call_count == 2

        # State 3: Identical tick (must skip)
        play_persistent_audio("sample_b64", 2)
        assert mock_html.call_count == 2
