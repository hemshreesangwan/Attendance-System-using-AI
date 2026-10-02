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
# STUDENT LOGIN & REGISTRATION HELPERS
# =========================================================

def login_student_session(student):
    """Store student in session state and navigate to student dashboard."""
    st.session_state.pop("student_data", None)
    st.session_state["student_data"] = student
    st.session_state["is_logged_in"] = True
    st.session_state["user_role"] = "student"
    st.session_state["login_type"] = "student"


# =========================================================
# STUDENT LOGIN SCREEN
# =========================================================

def student_screen_login():
    c1, c2 = st.columns(2, vertical_alignment="center", gap="xxlarge")
    with c1:
        header_dashboard()
    with c2:
        if st.button("Go back to Home", type="secondary", key="student_login_home_btn", shortcut="control+backspace"):
            st.session_state["login_type"] = None
            st.session_state.pop("student_data", None)
            st.session_state["is_logged_in"] = False
            st.session_state["user_role"] = None
            st.rerun()

    st.header("Student Portal: Login", text_alignment="center")
    st.space()

    tab_cam, tab_upload, tab_manual = st.tabs([
        "📷 FaceID (Camera)",
        "📁 FaceID (Upload Photo)",
        "🆔 Student Profile Login"
    ])

    photo_source = None

    with tab_cam:
        st.write("Align your face in the center of the frame and take a snapshot.")
        cam_pic = st.camera_input("Take Snapshot", key="student_cam_login")
        if cam_pic:
            photo_source = cam_pic

    with tab_upload:
        st.write("Upload a selfie or portrait photo with a clear view of your face.")
        up_pic = st.file_uploader(
            "Choose a JPG, JPEG, or PNG image",
            type=["jpg", "jpeg", "png"],
            key="student_up_login"
        )
        if up_pic:
            photo_source = up_pic

    with tab_manual:
        all_students = get_all_students()
        if all_students:
            st.write("Select your registered student profile to login directly:")
            student_options = {f"{s['name']} (ID: {s['student_id']})": s for s in all_students}
            selected_label = st.selectbox(
                "Choose Profile",
                options=list(student_options.keys()),
                key="student_manual_select"
            )
            if st.button("Login with Selected Profile", type="primary", icon=":material/login:", width="stretch", key="student_manual_login_btn"):
                student = student_options[selected_label]
                login_student_session(student)
                st.toast(f"Welcome back, {student['name']}!", icon="👋")
                time.sleep(1)
                st.rerun()
        else:
            st.info("No students registered yet. Click 'Register Instead' below to create the first student profile.")

    # Process Face Login if photo provided
    if photo_source:
        img = np.array(Image.open(photo_source).convert("RGB"))
        with st.spinner("AI is scanning and matching face..."):
            detected, all_ids, num_faces = predict_attendance(img)

        if num_faces == 0:
            st.warning("Face not found! Please ensure your face is well-lit and facing the camera directly.")
        elif num_faces > 1:
            st.warning("Multiple faces detected. Please make sure only your face is visible in the frame.")
        else:
            if detected:
                detected_ids = list(detected.keys())
                if detected_ids:
                    student_id = int(detected_ids[0])
                    all_students = get_all_students()
                    student = next((s for s in all_students if int(s["student_id"]) == student_id), None)
                    if student:
                        login_student_session(student)
                        st.toast(f"Welcome back, {student['name']}!", icon="👋")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Recognized face ID was not found in the student database.")
            else:
                st.info("Face not recognized in the system. You might need to register first.")
                if st.button("Register New Student Profile Now", type="primary", icon=":material/person_add:"):
                    st.session_state.student_login_type = "register"
                    st.rerun()

    st.divider()

    btnc1, btnc2 = st.columns(2)
    with btnc1:
        if st.button("New Student? Register Instead", type="primary", icon=":material/person_add:", width="stretch", key="goto_student_register"):
            st.session_state.student_login_type = "register"
            st.rerun()

    with btnc2:
        if st.button("Teacher Portal Instead", type="tertiary", icon=":material/school:", width="stretch", key="goto_teacher_from_student"):
            st.session_state.login_type = "teacher"
            st.rerun()

    footer_dashboard()


# =========================================================
# STUDENT REGISTRATION SCREEN
# =========================================================

def student_screen_register():
    c1, c2 = st.columns(2, vertical_alignment="center", gap="xxlarge")
    with c1:
        header_dashboard()
    with c2:
        if st.button("Go back to Home", type="secondary", key="student_reg_home_btn", shortcut="control+backspace"):
            st.session_state["login_type"] = None
            st.session_state.pop("student_data", None)
            st.session_state["is_logged_in"] = False
            st.session_state["user_role"] = None
            st.rerun()

    st.header("Student Portal: Register Profile", text_alignment="center")
    st.space()

    with st.container(border=True):
        st.subheader("1. Personal Details")
        new_name = st.text_input("Full Name", placeholder="E.g. Akash Sharma", key="reg_student_fullname")

        st.subheader("2. Face ID Biometric Registration")
        st.write("Take a snapshot with your webcam or upload a clear photo of your face.")

        reg_cam_tab, reg_upload_tab = st.tabs(["📷 Camera Snapshot", "📁 Upload Face Image"])
        reg_photo_source = None

        with reg_cam_tab:
            cam_p = st.camera_input("Capture facial snapshot", key="student_reg_cam")
            if cam_p:
                reg_photo_source = cam_p

        with reg_upload_tab:
            up_p = st.file_uploader("Upload clear face photo", type=["jpg", "jpeg", "png"], key="student_reg_upload")
            if up_p:
                reg_photo_source = up_p

        st.subheader("3. Voice Profile (Optional)")
        st.info("Record a short voice sample for acoustic attendance verification.")
        audio_data = None
        try:
            audio_data = st.audio_input("Record phrase: 'I am present'", key="student_reg_audio")
        except Exception:
            pass

        st.divider()

        btnc1, btnc2 = st.columns(2)
        with btnc1:
            if st.button("Register & Create Profile", type="primary", icon=":material/how_to_reg:", width="stretch", key="submit_student_reg"):
                if not new_name.strip():
                    st.warning("Please enter your full name!")
                elif reg_photo_source is None:
                    st.warning("Please take a camera snapshot or upload a face photo to complete registration!")
                else:
                    with st.spinner("Analyzing biometric facial features..."):
                        img = np.array(Image.open(reg_photo_source).convert("RGB"))
                        encodings = get_face_embeddings(img)

                        if not encodings:
                            st.error("No face detected in the photo. Please ensure good lighting and look directly at the camera.")
                        else:
                            face_emb = encodings[0].tolist()
                            voice_emb = None

                            if audio_data:
                                try:
                                    voice_emb = get_voice_embedding(audio_data.read())
                                    if hasattr(voice_emb, "tolist"):
                                        voice_emb = voice_emb.tolist()
                                except Exception as e:
                                    st.warning(f"Voice embedding skipped: {e}. Registering with facial profile.")

                            try:
                                response_data = create_student(
                                    new_name.strip(),
                                    face_embedding=face_emb,
                                    voice_embedding=voice_emb
                                )

                                if response_data:
                                    train_classifier()
                                    new_student = response_data[0]
                                    login_student_session(new_student)
                                    st.success(f"Profile created successfully! Welcome, {new_student['name']}!")
                                    time.sleep(1.5)
                                    st.rerun()
                                else:
                                    st.error("Could not save student profile in the database.")
                            except Exception as e:
                                st.error(f"Registration failed: {e}")

        with btnc2:
            if st.button("Already Registered? Login Instead", type="secondary", icon=":material/login:", width="stretch", key="goto_student_login"):
                st.session_state.student_login_type = "login"
                st.rerun()

    footer_dashboard()


# =========================================================
# MAIN STUDENT SCREEN CONTROLLER
# =========================================================

def student_screen():
    style_background_dashboard()
    style_base_layout()

    # 1. If already logged in, show dashboard
    if "student_data" in st.session_state:
        student_dashboard()
        return

    # 2. Track login/register mode
    if "student_login_type" not in st.session_state:
        st.session_state.student_login_type = "login"

    # 3. Route to login or register screen
    match st.session_state.student_login_type:
        case "register":
            student_screen_register()
        case _:
            student_screen_login()