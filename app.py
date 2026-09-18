
import streamlit as st
import sys
import tempfile
import json
import io
import base64
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.inference import YOLOv11Inference
from src.utils import (
    save_metadata,
    get_unique_classes_counts,
)


# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------

sys.path.append(str(Path(__file__).parent))


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="YOLOv11 Image Search",
    page_icon="🔍",
    layout="wide",
)


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def img_to_base64(image: Image.Image) -> str:
    """Convert PIL image to base64 for HTML display."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def init_session_state():

    defaults = {
        "metadata": None,
        "unique_classes": [],
        "count_options": {},
        "search_results": [],
        "search_params": {
            "search_mode": "Any of selected classes (OR)",
            "selected_classes": [],
            "thresholds": {},
        },
        "show_boxes": True,
        "grid_columns": 3,
        "highlight_matches": True,
    }

    for key, value in defaults.items():

        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ---------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    .image-card {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.12);
        margin-bottom: 20px;
        background: #f8f9fa;
    }

    .image-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.18);
        transition: all 0.2s ease;
    }

    .image-container {
        position: relative;
        width: 100%;
        aspect-ratio: 4/3;
    }

    .image-container img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }

    .meta-overlay {
        padding: 10px;
        background: rgba(0,0,0,0.85);
        color: white;
        font-size: 13px;
        line-height: 1.4;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Title
# ---------------------------------------------------------

st.title("🔍 YOLOv11 Computer Vision Search")

st.write(
    "Upload images, run YOLOv11 object detection, "
    "then search the images using detected objects and object counts."
)


# =========================================================
# STEP 1: UPLOAD IMAGES
# =========================================================

st.header("📤 Upload Images")

uploaded_files = st.file_uploader(
    "Choose one or more images",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
)


if uploaded_files:

    st.success(f"{len(uploaded_files)} image(s) uploaded.")

    # -----------------------------------------------------
    # Show uploaded images
    # -----------------------------------------------------

    with st.expander(
        f"Preview uploaded images ({len(uploaded_files)})",
        expanded=False,
    ):

        preview_cols = st.columns(4)

        for index, uploaded_file in enumerate(uploaded_files):

            with preview_cols[index % 4]:

                image = Image.open(uploaded_file)

                st.image(
                    image,
                    caption=uploaded_file.name,
                    use_container_width=True,
                )


    # -----------------------------------------------------
    # Model selection
    # -----------------------------------------------------

    st.subheader("⚙️ Detection Settings")

    model_path = st.text_input(
        "YOLO model",
        value="yolo11m.pt",
        help="Enter the YOLO model filename available in your repository.",
    )


    # -----------------------------------------------------
    # Run inference
    # -----------------------------------------------------

    if st.button(
        "🚀 Analyze Images",
        type="primary",
        use_container_width=True,
    ):

        try:

            with st.spinner(
                "Running YOLOv11 object detection..."
            ):

                # Temporary directory for uploaded images
                temp_dir = tempfile.mkdtemp(
                    prefix="yolo_uploads_"
                )

                temp_path = Path(temp_dir)


                # -----------------------------------------
                # Save uploaded files
                # -----------------------------------------

                for uploaded_file in uploaded_files:

                    file_path = temp_path / uploaded_file.name

                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())


                # -----------------------------------------
                # Load YOLO model
                # -----------------------------------------

                inferencer = YOLOv11Inference(
                    model_path
                )


                # -----------------------------------------
                # Run existing inference pipeline
                # -----------------------------------------

                metadata = inferencer.process_directory(
                    str(temp_path)
                )


                # -----------------------------------------
                # Store metadata
                # -----------------------------------------

                st.session_state.metadata = metadata

                (
                    st.session_state.unique_classes,
                    st.session_state.count_options,
                ) = get_unique_classes_counts(
                    metadata
                )


                # Reset previous search
                st.session_state.search_results = []

                st.session_state.search_params = {
                    "search_mode":
                        "Any of selected classes (OR)",

                    "selected_classes": [],

                    "thresholds": {},
                }


            st.success(
                f"✅ Successfully analyzed "
                f"{len(metadata)} image(s)."
            )

        except Exception as e:

            st.error(
                f"❌ Error during inference: {str(e)}"
            )


# =========================================================
# SEARCH ENGINE
# =========================================================

if st.session_state.metadata:

    st.divider()

    st.header("🔎 Search Engine")

    st.caption(
        "Search your uploaded images using the objects "
        "detected by YOLOv11."
    )


    # -----------------------------------------------------
    # Search mode
    # -----------------------------------------------------

    st.session_state.search_params["search_mode"] = st.radio(
        "Search mode",

        (
            "Any of selected classes (OR)",
            "All selected classes (AND)",
        ),

        horizontal=True,
    )


    # -----------------------------------------------------
    # Select classes
    # -----------------------------------------------------

    st.session_state.search_params[
        "selected_classes"
    ] = st.multiselect(
        "Objects to search for",

        options=st.session_state.unique_classes,

        placeholder="Select objects...",
    )


    selected_classes = (
        st.session_state.search_params[
            "selected_classes"
        ]
    )


    # -----------------------------------------------------
    # Count thresholds
    # -----------------------------------------------------

    if selected_classes:

        st.subheader("🔢 Object Count Limits")

        threshold_cols = st.columns(
            len(selected_classes)
        )

        for i, cls in enumerate(selected_classes):

            with threshold_cols[i]:

                options = [
                    "None"
                ] + st.session_state.count_options[cls]

                st.session_state.search_params[
                    "thresholds"
                ][cls] = st.selectbox(
                    f"Maximum {cls} count",
                    options=options,
                    key=f"threshold_{cls}",
                )


    # -----------------------------------------------------
    # Search
    # -----------------------------------------------------

    if st.button(
        "🔍 Search Images",
        type="primary",
        use_container_width=True,
    ):

        if not selected_classes:

            st.warning(
                "Please select at least one object."
            )

        else:

            results = []

            search_params = (
                st.session_state.search_params
            )


            for item in st.session_state.metadata:

                class_matches = {}


                # -----------------------------------------
                # Check every selected class
                # -----------------------------------------

                for cls in selected_classes:

                    class_detections = [
                        d
                        for d in item["detections"]
                        if d["class"] == cls
                    ]

                    class_count = len(
                        class_detections
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
                # OR search
                # -----------------------------------------

                if (
                    search_params["search_mode"]
                    ==
                    "Any of selected classes (OR)"
                ):

                    matches = any(
                        class_matches.values()
                    )


                # -----------------------------------------
                # AND search
                # -----------------------------------------

                else:

                    matches = all(
                        class_matches.values()
                    )


                if matches:

                    results.append(item)


            st.session_state.search_results = results


# =========================================================
# DISPLAY RESULTS
# =========================================================

if st.session_state.search_results:

    results = st.session_state.search_results

    search_params = (
        st.session_state.search_params
    )


    st.divider()

    st.subheader(
        f"📷 {len(results)} Matching Image(s)"
    )


    # -----------------------------------------------------
    # Display controls
    # -----------------------------------------------------

    with st.expander(
        "🎛️ Display Options",
        expanded=True,
    ):

        option_cols = st.columns(3)


        with option_cols[0]:

            st.session_state.show_boxes = st.checkbox(
                "Show bounding boxes",
                value=st.session_state.show_boxes,
            )


        with option_cols[1]:

            st.session_state.grid_columns = st.slider(
                "Grid columns",
                min_value=2,
                max_value=6,
                value=st.session_state.grid_columns,
            )


        with option_cols[2]:

            st.session_state.highlight_matches = (
                st.checkbox(
                    "Highlight matching classes",
                    value=st.session_state.highlight_matches,
                )
            )


    # -----------------------------------------------------
    # Create image grid
    # -----------------------------------------------------

    grid_cols = st.columns(
        st.session_state.grid_columns
    )


    for index, result in enumerate(results):

        with grid_cols[
            index % st.session_state.grid_columns
        ]:

            try:

                image_path = Path(
                    result["image_path"]
                )


                img = Image.open(
                    image_path
                ).convert("RGB")


                draw = ImageDraw.Draw(img)


                # -----------------------------------------
                # Font
                # -----------------------------------------

                try:

                    font = ImageFont.truetype(
                        "arial.ttf",
                        12,
                    )

                except:

                    font = ImageFont.load_default()


                # -----------------------------------------
                # Bounding boxes
                # -----------------------------------------

                if st.session_state.show_boxes:

                    for det in result["detections"]:

                        cls = det["class"]

                        bbox = det["bbox"]


                        # Matching class
                        if cls in selected_classes:

                            outline_color = "#30C938"
                            thickness = 3


                        # Non-matching class
                        elif not st.session_state.highlight_matches:

                            outline_color = "#666666"
                            thickness = 1


                        else:

                            continue


                        draw.rectangle(
                            bbox,
                            outline=outline_color,
                            width=thickness,
                        )


                        label = (
                            f"{cls} "
                            f"{det['confidence']:.2f}"
                        )


                        text_bbox = draw.textbbox(
                            (0, 0),
                            label,
                            font=font,
                        )


                        text_width = (
                            text_bbox[2]
                            -
                            text_bbox[0]
                        )


                        text_height = (
                            text_bbox[3]
                            -
                            text_bbox[1]
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
                                + 4,
                            ],
                            fill=outline_color,
                        )


                        draw.text(
                            (
                                bbox[0] + 4,
                                bbox[1] + 2,
                            ),
                            label,
                            fill="white",
                            font=font,
                        )


                # -----------------------------------------
                # Metadata
                # -----------------------------------------

                meta_items = [

                    f"{k}: {v}"

                    for k, v
                    in result["class_counts"].items()

                    if k in selected_classes

                ]


                # -----------------------------------------
                # Image card
                # -----------------------------------------

                st.markdown(
                    f"""
                    <div class="image-card">

                        <div class="image-container">

                            <img
                                src="data:image/png;base64,
                                {img_to_base64(img)}"
                            >

                        </div>

                        <div class="meta-overlay">

                            <strong>
                                {image_path.name}
                            </strong>

                            <br>

                            {
                                ", ".join(meta_items)
                                if meta_items
                                else "No matches"
                            }

                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )


            except Exception as e:

                st.error(
                    f"Error displaying image: {str(e)}"
                )


    # -----------------------------------------------------
    # Export results
    # -----------------------------------------------------

    with st.expander("📦 Export Results"):

        st.download_button(
            label="Download Results (JSON)",

            data=json.dumps(
                results,
                indent=2,
            ),

            file_name="search_results.json",

            mime="application/json",
        )


# =========================================================
# NO RESULTS
# =========================================================

elif (
    st.session_state.metadata
    and st.session_state.search_params[
        "selected_classes"
    ]
):

    st.info(
        "No images matched your search criteria."
    )
