import streamlit as st


def render_reset_button():
    if st.button(
        "🔄 Start New Assessment",
        key="reset_assessment",
        use_container_width=True,
        type="secondary",
    ):
        st.session_state.clear()
        st.query_params.clear()
        st.rerun()
