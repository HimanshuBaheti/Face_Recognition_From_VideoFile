# Face Recognition from a Video File

A Python computer-vision project that **detects and identifies faces in a prerecorded video** by comparing them against a gallery of reference images. Built with **OpenCV** and the **`face_recognition`** library, the application annotates video frames with bounding boxes and identity labels, displaying **Unknown** for faces that do not match the reference gallery.

> **Project status:** A local desktop demonstration. The script uses absolute Windows paths and must be configured for your machine before running. It displays annotated frames but does not save a processed video file.

## Features

- **Reference image registration:** Loads known face images from a directory and computes a 128-dimensional face embedding for each detected face.
- **Video-based face detection:** Reads a local video file frame by frame using OpenCV.
- **Face identification:** Compares each detected face embedding against the stored known-face embeddings.
- **Configurable matching tolerance:** Uses a similarity matching threshold of `0.50` (as passed to `face_recognition.compare_faces`).
- **Unknown face handling:** Labels detected faces `Unknown` if none of the reference embeddings match.
- **Annotated video preview:** Draws a red bounding box and an identity label over each detected face.
- **Manual termination:** Press `q` to exit the playback window.

## Technology stack

| Component | Technology |
| --- | --- |
| Programming language | Python |
| Image and video processing | OpenCV (`cv2`) |
| Face detection and recognition | `face_recognition` (built on dlib) |
| File and folder handling | Python `os` |
| Input | Reference JPG images and an MP4 video |
| Output | Live on-screen annotated video frames |

## How it works

```text
Reference images (dataset1/img/)
            |
            v
Load each image and extract face encoding
            |
            v
Store known encodings and filenames as labels
            |
            +-------------------------------+
                                            |
Input video (dataset1/Video_clip/)          |
            |                               |
            v                               |
Read video frames with OpenCV               |
            |                               |
            v                               |
Convert BGR frame to RGB                    |
            |                               |
            v                               |
Detect faces and compute encodings          |
            |                               |
            v                               |
Compare each encoding to known faces <------+
            |
            v
Match found? ---- Yes ---> Label using reference filename
      |
      No
      |
      v
Label "Unknown"
            |
            v
Draw boxes and names on frame
            |
            v
Display video until completed or 'q' pressed
```

### Face recognition logic

The project relies on `face_recognition.face_locations()` and `face_recognition.face_encodings()` to locate faces and represent their facial features numerically.

For each detected face, the script runs:

```python
matches = face_recognition.compare_faces(
    known_faces,
    face_encoding,
    tolerance=0.50
)

name = "Unknown"

if True in matches:
    idx = matches.index(True)
    name = known_face_names[idx]
```

The **first** matched reference encoding determines the displayed name. If there is no match, the label remains `Unknown`.

**Important:** This is a pretrained face-recognition pipeline, not a custom model trained from scratch. The `0.50` tolerance is a distance threshold, not an accuracy score, and the repository contains no benchmark establishing recognition accuracy.

## Repository structure

```text
Face_Recognition_From_VideoFile/
├── README.md
├── main2.py                          # Face registration, video processing, annotation
└── dataset1/
    ├── img/                          # Reference face images (JPG)
    │   ├── 1.jpg
    │   ├── 2.jpg
    │   └── ...                       # Other numbered reference images
    └── Video_clip/
        ├── VID_20230302_170157 (online-video-cutter.com).mp4
        └── output.avi
```

The image filenames without extensions (e.g., `1` and `2`) become the identity labels. To display meaningful labels, rename the reference images appropriately, such as `Alice.jpg` and `Bob.jpg`, and ensure they show the corresponding subjects.

## Getting started

### Prerequisites

- Python 3
- A desktop environment capable of displaying OpenCV windows
- Working installations of `opencv-python` and `face-recognition`
- CMake/compiler tooling if required for building `dlib` on your system

> Dependency installation, particularly `dlib` / `face-recognition`, may require a supported Python version and platform-specific build tools. The repository does not currently provide a pinned dependency file.

### 1. Clone the repository

```bash
git clone https://github.com/HimanshuBaheti/Face_Recognition_From_VideoFile.git
cd Face_Recognition_From_VideoFile
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install opencv-python face-recognition
```

If `dlib` fails to install, review its Python/platform compatibility and install any build dependencies needed for your operating system.

### 4. Configure local file paths

**Before running**, open `main2.py` and update these two existing lines, which currently reference the original developer's Windows filesystem:

```python
known_faces_dir = "C:/Himanshu/FaceRecognition/dataset1/img"
```

```python
cap = cv2.VideoCapture("C:/Himanshu/FaceRecognition/dataset1/Video_clip/VID_20230302_170157 (online-video-cutter.com).mp4")
```

For a portable setup, replace the paths with:

```python
known_faces_dir = "dataset1/img"
video_path = "dataset1/Video_clip/VID_20230302_170157 (online-video-cutter.com).mp4"
cap = cv2.VideoCapture(video_path)
```

This is an **example of the change to make**; the committed `main2.py` has not been modified.

### 5. Run the script

From the repository root:

```bash
python main2.py
```

An OpenCV window titled `Video` displays each frame with detected faces, red bounding boxes, and identity labels. Press **q** to close the window early. When playback ends, the script releases the video capture and closes OpenCV windows.

## Core processing steps

1. **Load reference images.** Each image in `dataset1/img/` is read with `face_recognition.load_image_file()`.
2. **Extract known encodings.** The first detected face encoding in each image is stored alongside its filename stem.
3. **Read video frames.** `cv2.VideoCapture()` sequentially loads frames from the specified MP4 file.
4. **Convert color space.** Frames are converted from OpenCV's BGR order to the RGB order expected by the recognition library.
5. **Detect and encode faces.** The library finds face locations and computes encodings within each frame.
6. **Find matches.** `compare_faces()` tests each detected encoding against reference encodings at tolerance `0.50`.
7. **Annotate frames.** OpenCV draws red rectangles and overlays the matched label or `Unknown`.
8. **Render preview.** `cv2.imshow()` displays the result. Processing ends at the end of the file or when `q` is pressed.

## Limitations and considerations

- **Fixed input paths:** The current script uses Windows-specific absolute paths. Update them before running on a different machine.
- **One face per reference file:** The code accesses `face_recognition.face_encodings(image)[0]`. An image without a detectable face raises an `IndexError`; an image with multiple faces only contributes the first encoding.
- **No explicit image filtering:** The loader iterates over every entry in the reference directory. Non-image files or nested directories may cause errors.
- **First-match behavior:** When more than one reference falls within the tolerance threshold, it selects the first match, not necessarily the nearest encoding.
- **No output video export:** The script displays modified frames via `cv2.imshow()` but does not use `cv2.VideoWriter()`. The existing `output.avi` file is in the repository; the current script does not create or update it.
- **GUI required:** OpenCV's display window generally will not work in a headless cloud/server environment without additional changes.
- **No benchmark or evaluation dataset:** The repo does not provide measured precision, recall, false match rates, or face recognition accuracy.
- **Biometric privacy:** Face embeddings and identifiable video frames are sensitive biometric-related information. Only process images and videos where you have appropriate permission, protect the data, and comply with applicable privacy laws.

## Potential improvements

- Make image and video paths configurable with command-line arguments.
- Validate images, check that a face was found, and handle multiple faces safely.
- Use `face_recognition.face_distance()` to select the closest candidate rather than the first threshold match.
- Add an option to save processed video using `cv2.VideoWriter()`.
- Add frame resizing or frame skipping for faster processing.
- Add clear error messages when paths or video codecs are invalid.
- Introduce automated tests and benchmark recognition performance across varied lighting, pose, and occlusion.

## Author

Repository maintained by [Himanshu Baheti](https://github.com/HimanshuBaheti).

---

*An educational computer-vision project demonstrating face detection, pretrained face embeddings, video-frame processing, and identity matching.*
