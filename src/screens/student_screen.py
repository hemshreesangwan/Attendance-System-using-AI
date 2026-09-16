import streamlit as st
import time

from PIL import Image
import numpy as np

from src.ui.base_layout import (
    style_background_dashboard,
    style_base_layout
)

from src.components.header import header_dashboard
from src.components.footer import footer_dashboard
from src.components.dialog_enroll import enroll_dialog
from src.components.subject_card import subject_card

from src.pipelines.face_pipeline import (
    predict_attendance,
    get_face_embeddings,
    train_classifier
)

from src.pipelines.voice_pipeline import get_voice_embedding

from src.database.db import (
    get_all_students,
    create_student,
    get_student_subjects,
    get_student_attendance,
    unenroll_student_to_subject
)


# =========================================================
# STUDENT DASHBOARD
# =========================================================

def student_dashboard():

    student_data = st.session_state.get("student_data")

    # Safety check
    if not student_data:
        st.session_state.pop("student_data", None)
        st.session_state["is_logged_in"] = False
        st.rerun()

    student_id = student_data["student_id"]

    c1, c2 = st.columns(
        2,
        vertical_alignment="center",
        gap="xxlarge"
    )

    with c1:
        header_dashboard()

    with c2:

        st.subheader(
            f"Welcome, {student_data['name']}"
        )

        if st.button(
            "Logout",
            type="secondary",
            key="student_logout",
            shortcut="control+backspace"
        ):

            # Completely remove current student session
            st.session_state.pop("student_data", None)

            st.session_state["is_logged_in"] = False
            st.session_state["user_role"] = None
            st.session_state["login_type"] = None

            st.rerun()

    st.space()

    c1, c2 = st.columns(2)

    with c1:
        st.header("Your Enrolled Subjects")

    with c2:

        if st.button(
            "Enroll in Subject",
            type="primary",
            width="stretch"
        ):
            enroll_dialog()

    st.divider()

    # =====================================================
    # LOAD STUDENT DATA
    # =====================================================

    with st.spinner(
        "Loading your enrolled subjects..."
    ):

        subjects = get_student_subjects(student_id)
        logs = get_student_attendance(student_id)

    # =====================================================
    # ATTENDANCE STATISTICS
    # =====================================================

    stats_map = {}

    for log in logs:

        subject_id = log["subject_id"]

        if subject_id not in stats_map:

            stats_map[subject_id] = {
                "total": 0,
                "attended": 0
            }

        stats_map[subject_id]["total"] += 1

        if log.get("is_present"):
            stats_map[subject_id]["attended"] += 1

    # =====================================================
    # SUBJECT CARDS
    # =====================================================

    cols = st.columns(2)

    for i, sub_node in enumerate(subjects):

        sub = sub_node["subjects"]

        subject_id = sub["subject_id"]

        stats = stats_map.get(
            subject_id,
            {
                "total": 0,
                "attended": 0
            }
        )

        def unenroll_button(
            subject_id=subject_id,
            subject_name=sub["name"]
        ):

            if st.button(
                "Unenroll from this course",
                type="tertiary",
                width="stretch",
                icon=":material/delete_forever:",
                key=f"unenroll_{subject_id}"
            ):

                unenroll_student_to_subject(
                    student_id,
                    subject_id
                )

                st.toast(
                    f"Unenrolled from {subject_name} successfully!"
                )

                st.rerun()

        with cols[i % 2]:

            subject_card(
                name=sub["name"],
                code=sub["subject_code"],
                section=sub["section"],
                stats=[
                    (
                        "📅",
                        "Total",
                        stats["total"]
                    ),
                    (
                        "✅",
                        "Attended",
                        stats["attended"]
                    )
                ],
                footer_callback=unenroll_button
            )

    footer_dashboard()


# =========================================================
# STUDENT LOGIN / REGISTRATION SCREEN
# =========================================================

def student_screen():

    style_background_dashboard()
    style_base_layout()

    # =====================================================
    # IF ALREADY LOGGED IN
    # =====================================================

    if "student_data" in st.session_state:

        student_dashboard()
        return

    # =====================================================
    # HEADER
    # =====================================================

    c1, c2 = st.columns(
        2,
        vertical_alignment="center",
        gap="xxlarge"
    )

    with c1:
        header_dashboard()

    with c2:

        if st.button(
            "Go back to Home",
            type="secondary",
            key="student_home_button",
            shortcut="control+backspace"
        ):

            st.session_state["login_type"] = None
            st.session_state.pop("student_data", None)
            st.session_state["is_logged_in"] = False
            st.session_state["user_role"] = None

            st.rerun()

    st.header(
        "Login using FaceID",
        text_alignment="center"
    )

    st.space()
    st.space()

    # =====================================================
    # CAMERA
    # =====================================================

    photo_source = st.camera_input(
        "Position your face in the center"
    )

    show_registration = False

    # =====================================================
    # FACE LOGIN
    # =====================================================

    if photo_source:

        img = np.array(
            Image.open(photo_source).convert("RGB")
        )

        with st.spinner("AI is scanning..."):

            detected, all_ids, num_faces = (
                predict_attendance(img)
            )

        # -------------------------------------------------
        # NO FACE
        # -------------------------------------------------

        if num_faces == 0:

            st.warning(
                "Face not found! Please position your face properly."
            )

        # -------------------------------------------------
        # MULTIPLE FACES
        # -------------------------------------------------

        elif num_faces > 1:

            st.warning(
                "Multiple faces found. "
                "Please make sure only your face is visible."
            )

        # -------------------------------------------------
        # ONE FACE
        # -------------------------------------------------

        else:

            if detected:

                # Get detected student ID
                detected_ids = list(
                    detected.keys()
                )

                if detected_ids:

                    student_id = int(
                        detected_ids[0]
                    )

                    # -------------------------------------
                    # FETCH THAT STUDENT FROM SUPABASE
                    # -------------------------------------

                    all_students = get_all_students()

                    student = next(
                        (
                            s
                            for s in all_students
                            if int(s["student_id"])
                            == student_id
                        ),
                        None
                    )

                    if student:

                        # IMPORTANT:
                        # Clear any previous student
                        # before storing the new one.

                        st.session_state.pop(
                            "student_data",
                            None
                        )

                        st.session_state[
                            "student_data"
                        ] = student

                        st.session_state[
                            "is_logged_in"
                        ] = True

                        st.session_state[
                            "user_role"
                        ] = "student"

                        st.session_state[
                            "login_type"
                        ] = "student"

                        st.toast(
                            f"Welcome back, {student['name']}!"
                        )

                        time.sleep(1)

                        st.rerun()

            else:

                st.info(
                    "Face not recognized! "
                    "You might be a new student."
                )

                show_registration = True

    # =====================================================
    # NEW STUDENT REGISTRATION
    # =====================================================

    if show_registration:

        with st.container(border=True):

            st.header(
                "Register New Profile"
            )

            new_name = st.text_input(
                "Enter your name",
                placeholder="E.g. Hamza Rizvi"
            )

            st.subheader(
                "Optional: Voice Enrollment"
            )

            st.info(
                "Enroll your voice for voice-only attendance."
            )

            audio_data = None

            try:

                audio_data = st.audio_input(
                    "Record a short phrase like "
                    "'I am present' or "
                    "'My name is Akash.'"
                )

            except Exception:

                st.error(
                    "Audio data failed!"
                )

            # ---------------------------------------------
            # CREATE ACCOUNT
            # ---------------------------------------------

            if st.button(
                "Create Account",
                type="primary"
            ):

                if not new_name.strip():

                    st.warning(
                        "Please enter your name!"
                    )

                elif photo_source is None:

                    st.warning(
                        "Please capture your face first."
                    )

                else:

                    with st.spinner(
                        "Creating profile..."
                    ):

                        img = np.array(
                            Image.open(
                                photo_source
                            ).convert("RGB")
                        )

                        # ---------------------------------
                        # GET FACE EMBEDDING
                        # ---------------------------------

                        encodings = (
                            get_face_embeddings(img)
                        )

                        if not encodings:

                            st.error(
                                "Couldn't capture your "
                                "facial features. "
                                "Please try again."
                            )

                        else:

                            face_emb = (
                                encodings[0].tolist()
                            )

                            # -----------------------------
                            # VOICE EMBEDDING
                            # -----------------------------

                            voice_emb = None

                            if audio_data:

                                voice_emb = (
                                    get_voice_embedding(
                                        audio_data.read()
                                    )
                                )

                                if hasattr(
                                    voice_emb,
                                    "tolist"
                                ):
                                    voice_emb = (
                                        voice_emb.tolist()
                                    )

                            # -----------------------------
                            # SAVE STUDENT
                            # -----------------------------

                            try:

                                response_data = (
                                    create_student(
                                        new_name.strip(),
                                        face_embedding=face_emb,
                                        voice_embedding=voice_emb
                                    )
                                )

                                if response_data:

                                    # -------------------------
                                    # RETRAIN FACE CLASSIFIER
                                    # -------------------------

                                    train_classifier()

                                    # -------------------------
                                    # GET EXACT NEW STUDENT
                                    # -------------------------

                                    new_student = (
                                        response_data[0]
                                    )

                                    # Clear previous session
                                    st.session_state.pop(
                                        "student_data",
                                        None
                                    )

                                    # Store ONLY newly created
                                    # student
                                    st.session_state[
                                        "student_data"
                                    ] = new_student

                                    st.session_state[
                                        "is_logged_in"
                                    ] = True

                                    st.session_state[
                                        "user_role"
                                    ] = "student"

                                    st.session_state[
                                        "login_type"
                                    ] = "student"

                                    st.success(
                                        f"Profile created successfully! "
                                        f"Hi {new_student['name']}!"
                                    )

                                    time.sleep(1)

                                    st.rerun()

                                else:

                                    st.error(
                                        "Student profile "
                                        "could not be created."
                                    )

                            except Exception as e:

                                st.error(
                                    f"Registration failed: {e}"
                                )

    footer_dashboard()