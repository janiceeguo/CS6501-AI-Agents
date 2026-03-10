import cv2
import base64
import io
from PIL import Image
import ollama

# ---------- IMAGE PREP (reuse your safe resize idea) ----------

def prepare_image_for_llava(frame,
                            max_dim=256,
                            jpeg_quality=70):

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    img.thumbnail((max_dim, max_dim), Image.LANCZOS)

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=jpeg_quality)

    img_b64 = base64.b64encode(buffer.getvalue()).decode()

    return img_b64


# ---------- FRAME EXTRACTION ----------

def extract_frames(video_path, seconds=2):

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * seconds)

    frames = []
    timestamps = []

    frame_num = 0

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        if frame_num % interval == 0:
            frames.append(frame)
            timestamps.append(frame_num / fps)

        frame_num += 1

    cap.release()

    return frames, timestamps


# ---------- LLaVA PERSON DETECTOR ----------

def detect_person(frame):

    img_b64 = prepare_image_for_llava(frame)

    response = ollama.chat(
        model="llava:7b-v1.5-q4_0",
        messages=[{
            "role": "user",
            "content": "Answer ONLY yes or no. Is there a person visible in this scene?",
            "images": [img_b64]
        }]
    )

    answer = response["message"]["content"].lower()

    return "yes" in answer


# ---------- SURVEILLANCE ANALYSIS ----------

def analyze_video(video_path):

    frames, timestamps = extract_frames(video_path, seconds=2)

    print(f"Extracted {len(frames)} frames")

    events = []

    person_present = False

    for i, frame in enumerate(frames):

        time = timestamps[i]

        print(f"Analyzing frame {i} at {time:.1f}s")

        detected = detect_person(frame)

        # person enters
        if detected and not person_present:
            events.append(("enter", time))
            person_present = True

        # person exits
        elif not detected and person_present:
            events.append(("exit", time))
            person_present = False

    return events


# ---------- RUN ----------

video_path = "/content/walking.mov"

events = analyze_video(video_path)

print("\nDetected Events:")
for event, t in events:
    print(f"Person {event} at {t:.1f} seconds")