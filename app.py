import streamlit as st
import json
import shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from src.inference import YOLOv11Inference
from src.utils import save_metadata, get_unique_classes_counts


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="YOLOv11 Image Search",
    page_icon="🔎",
    layout="wide"
)


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

def init_session_state():

    defaults = {
        "metadata": None,
        "unique_classes": [],
        "count_options": {},
        "search_results": [],
        "search_params": {
            "search_mode": "Any of selected classes (OR)",
            "selected_classes": [],
            "thresholds": {}
        },
        "show_boxes": True,
        "grid_columns": 3,
        "highlight_matches": True,
        "upload_dir": None,
    }

    for key, value in defaults.items():

        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ---------------------------------------------------------
# CSS
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    .image-card {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0,0,0,0.12);
        margin-bottom: 20px;
        background: #f8f9fa;
    }

    .meta-overlay {
        padding: 10px;
        background: rgba(0,0,0,0.85);
        color: white;
        font-size: 13px;
        line-height: 1.5;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("🔎 YOLOv11 Image Search")

st.write(
    "Upload images, run YOLOv11 object detection, "
    "and search the images using detected object classes and counts."
)


# ---------------------------------------------------------
# CREATE SESSION UPLOAD DIRECTORY
# ---------------------------------------------------------

if st.session_state.upload_dir is None:

    upload_dir = Path("streamlit_uploads")

    upload_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    st.session_state.upload_dir = str(upload_dir)


upload_dir = Path(st.session_state.upload_dir)


# ---------------------------------------------------------
# UPLOAD IMAGES
# ---------------------------------------------------------

st.header("📤 Upload Images")

uploaded_files = st.file_uploader(
    "Choose one or more images",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True
)


# ---------------------------------------------------------
# PROCESS IMAGES
# ---------------------------------------------------------

if uploaded_files:

    st.write(f"**{len(uploaded_files)} image(s) selected**")

    if st.button(
        "🚀 Run YOLOv11 Inference",
        type="primary"
    ):

        try:

            # Clear previous search results
            st.session_state.search_results = []

            # -------------------------------------------------
            # Save uploaded files
            # -------------------------------------------------

            current_upload_dir = upload_dir / "current"

            current_upload_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            # Remove previous uploaded images
            for old_file in current_upload_dir.iterdir():

                if old_file.is_file():
                    old_file.unlink()

                elif old_file.is_dir():
                    shutil.rmtree(old_file)


            image_paths = []

            for uploaded_file in uploaded_files:

                image_path = current_upload_dir / uploaded_file.name

                with open(image_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                image_paths.append(image_path)


            # -------------------------------------------------
            # YOLO inference
            # -------------------------------------------------

            with st.spinner(
                "Running YOLOv11 object detection..."
            ):

                inferencer = YOLOv11Inference(
                    "yolo11m.pt"
                )

                metadata = []

                for image_path in image_paths:

                    try:

                        result = inferencer.process_image(
                            image_path
                        )

                        metadata.append(result)

                    except Exception as e:

                        st.warning(
                            f"Could not process "
                            f"{image_path.name}: {e}"
                        )


            # -------------------------------------------------
            # Save metadata
            # -------------------------------------------------

            st.session_state.metadata = metadata

            (
                st.session_state.unique_classes,
                st.session_state.count_options
            ) = get_unique_classes_counts(metadata)


            # Reset search parameters
            st.session_state.search_params = {
                "search_mode":
                    "Any of selected classes (OR)",

                "selected_classes": [],

                "thresholds": {}
            }


            st.success(
                f"Successfully processed "
                f"{len(metadata)} image(s)."
            )


        except Exception as e:

            st.error(
                f"Error during inference: {e}"
            )


# ---------------------------------------------------------
# SEARCH ENGINE
# ---------------------------------------------------------

if st.session_state.metadata:

    st.divider()

    st.header("🔍 Search Engine")


    # -----------------------------------------------------
    # SEARCH MODE
    # -----------------------------------------------------

    search_mode = st.radio(
        "Search mode",
        [
            "Any of selected classes (OR)",
            "All selected classes (AND)"
        ],
        horizontal=True
    )

    st.session_state.search_params[
        "search_mode"
    ] = search_mode


    # -----------------------------------------------------
    # CLASS SELECTION
    # -----------------------------------------------------

    selected_classes = st.multiselect(
        "Classes to search for",
        options=st.session_state.unique_classes
    )

    st.session_state.search_params[
        "selected_classes"
    ] = selected_classes


    # -----------------------------------------------------
    # COUNT THRESHOLDS
    # -----------------------------------------------------

    if selected_classes:

        st.subheader("Count Thresholds")

        threshold_columns = st.columns(
            len(selected_classes)
        )

        for i, cls in enumerate(selected_classes):

            with threshold_columns[i]:

                options = (
                    ["None"]
                    + st.session_state.count_options[cls]
                )

                threshold = st.selectbox(
                    f"Maximum {cls} count",
                    options=options,
                    key=f"threshold_{cls}"
                )

                st.session_state.search_params[
                    "thresholds"
                ][cls] = threshold


    # -----------------------------------------------------
    # SEARCH BUTTON
    # -----------------------------------------------------

    if st.button(
        "🔎 Search Images",
        type="primary"
    ):

        if not selected_classes:

            st.warning(
                "Select at least one class."
            )

        else:

            results = []

            search_params = (
                st.session_state.search_params
            )


            # ---------------------------------------------
            # SEARCH EACH IMAGE
            # ---------------------------------------------

            for item in st.session_state.metadata:

                class_matches = {}


                for cls in selected_classes:

                    class_count = sum(
                        1
                        for detection in item["detections"]
                        if detection["class"] == cls
                    )


                    threshold = (
                        search_params[
                            "thresholds"
                        ].get(cls, "None")
                    )


                    if threshold == "None":

                        class_matches[cls] = (
                            class_count >= 1
                        )

                    else:

                        class_matches[cls] = (
                            class_count >= 1
                            and
                            class_count <= int(threshold)
                        )


                # -----------------------------------------
                # OR SEARCH
                # -----------------------------------------

                if (
                    search_params["search_mode"]
                    == "Any of selected classes (OR)"
                ):

                    matches = any(
                        class_matches.values()
                    )


                # -----------------------------------------
                # AND SEARCH
                # -----------------------------------------

                else:

                    matches = all(
                        class_matches.values()
                    )


                if matches:

                    results.append(item)


            st.session_state.search_results = results


# ---------------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------------

if st.session_state.search_results:

    results = st.session_state.search_results

    search_params = (
        st.session_state.search_params
    )

    selected_classes = (
        search_params["selected_classes"]
    )


    st.divider()

    st.subheader(
        f"📷 Results: {len(results)} matching image(s)"
    )


    # -----------------------------------------------------
    # DISPLAY OPTIONS
    # -----------------------------------------------------

    with st.expander(
        "⚙️ Display Options",
        expanded=True
    ):

        option_columns = st.columns(3)


        with option_columns[0]:

            st.session_state.show_boxes = st.checkbox(
                "Show bounding boxes",
                value=st.session_state.show_boxes
            )


        with option_columns[1]:

            st.session_state.grid_columns = st.slider(
                "Grid columns",
                min_value=1,
                max_value=6,
                value=st.session_state.grid_columns
            )


        with option_columns[2]:

            st.session_state.highlight_matches = st.checkbox(
                "Highlight matching classes",
                value=st.session_state.highlight_matches
            )


    # -----------------------------------------------------
    # IMAGE GRID
    # -----------------------------------------------------

    grid_columns = st.columns(
        st.session_state.grid_columns
    )


    for index, result in enumerate(results):

        with grid_columns[
            index % st.session_state.grid_columns
        ]:

            try:

                image_path = Path(
                    result["image_path"]
                )


                # -----------------------------------------
                # CHECK IMAGE EXISTS
                # -----------------------------------------

                if not image_path.exists():

                    st.error(
                        f"Image not found:\n"
                        f"{image_path}"
                    )

                    continue


                # -----------------------------------------
                # OPEN IMAGE
                # -----------------------------------------

                img = Image.open(
                    image_path
                ).convert("RGB")


                # -----------------------------------------
                # DRAW BOUNDING BOXES
                # -----------------------------------------

                if st.session_state.show_boxes:

                    draw = ImageDraw.Draw(img)


                    try:

                        font = ImageFont.truetype(
                            "arial.ttf",
                            14
                        )

                    except:

                        font = ImageFont.load_default()


                    for detection in result[
                        "detections"
                    ]:

                        cls = detection["class"]
                        bbox = detection["bbox"]
                        confidence = detection[
                            "confidence"
                        ]


                        # ---------------------------------
                        # Matching class
                        # ---------------------------------

                        if cls in selected_classes:

                            outline_width = 3


                        # ---------------------------------
                        # Non-matching class
                        # ---------------------------------

                        elif not st.session_state.highlight_matches:

                            outline_width = 1


                        # ---------------------------------
                        # Skip non-matching classes
                        # ---------------------------------

                        else:

                            continue


                        draw.rectangle(
                            bbox,
                            outline="green",
                            width=outline_width
                        )


                        label = (
                            f"{cls} "
                            f"{confidence:.2f}"
                        )


                        text_bbox = draw.textbbox(
                            (0, 0),
                            label,
                            font=font
                        )


                        text_width = (
                            text_bbox[2]
                            - text_bbox[0]
                        )

                        text_height = (
                            text_bbox[3]
                            - text_bbox[1]
                        )


                        draw.rectangle(
                            [
                                bbox[0],
                                bbox[1],
                                bbox[0]
                                + text_width
                                + 8,
                                bbox[1]
                                + text_height
                                + 4
                            ],
                            fill="green"
                        )


                        draw.text(
                            (
                                bbox[0] + 4,
                                bbox[1] + 2
                            ),
                            label,
                            fill="white",
                            font=font
                        )


                # -----------------------------------------
                # DISPLAY IMAGE
                # -----------------------------------------

                st.image(
                    img,
                    width="stretch"
                )


                # -----------------------------------------
                # METADATA
                # -----------------------------------------

                st.markdown(
                    f"**{image_path.name}**"
                )


                meta_items = []

                for cls in selected_classes:

                    count = result[
                        "class_counts"
                    ].get(cls, 0)

                    if count > 0:

                        meta_items.append(
                            f"{cls}: {count}"
                        )


                if meta_items:

                    st.caption(
                        " • ".join(meta_items)
                    )

                else:

                    st.caption(
                        "No matching objects"
                    )


            except Exception as e:

                st.error(
                    f"Error displaying "
                    f"{result.get('image_path', 'image')}: "
                    f"{e}"
                )


# ---------------------------------------------------------
# EXPORT
# ---------------------------------------------------------

if st.session_state.search_results:

    st.divider()

    with st.expander("📦 Export Results"):

        st.download_button(
            label="Download Results (JSON)",

            data=json.dumps(
                st.session_state.search_results,
                indent=2
            ),

            file_name="search_results.json",

            mime="application/json"
        )