import streamlit as st
import whisper
import os
import tempfile
from st_audiorec import st_audiorec
import torch
import torchaudio
from torchaudio.transforms import Resample
import transformers
import gspread

gc = gspread.service_account("./client.json")
sh = gc.open("DB")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def levenshtein_distance(s1, s2):
    # Ensuring s1 is the larger string to simplify the calculations
    if len(s1) < len(s2):
        s1, s2 = s2, s1

    # Edge case: one of the strings is empty
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def check(input_text, targets, confidence_threshold=0.6):
    words = input_text.split()
    results = []

    for target_phrase in targets:
        target_words = target_phrase.split()
        best_similarity_score = 0  # Track the highest similarity score for this target phrase

        for i in range(len(words) - len(target_words) + 1):
            subset = words[i:i + len(target_words)]
            # Calculate the Levenshtein distance and so, the similarity per word, then average to get a phrase-level approximation.
            distance_sum = sum(levenshtein_distance(target.lower(), word.lower())
                               for target, word in zip(target_words, subset))
            max_length = sum(len(word) for word in target_words)
            similarity_score = 1 - distance_sum / max_length
            best_similarity_score = max(best_similarity_score, similarity_score)

        if best_similarity_score > confidence_threshold:
            results.append((target_phrase, best_similarity_score))

    return results


target_phrases = ["vápaszarufa", "gerenda", "vápa szarufa", "korhadt", "gomba kár", "gombakár", "rovar kár", "rovarkár",
                  "kismélységű", "kis mélységű", "egyharmad keresztmetszetű", "egy harmad keresztmetszetű",
                  "egyharmadkeresztmetszetű", "fél keresztmetszetű", "félkeresztmetszetű", "esetlegesen megerősítés",
                  "megerősítés", "esetlegesen bárdolás", "bárdolás", "esetlegesen vegykezelés", "vegykezelés",
                  "esetlegesen csere", "csere", "székoszlop", "könyökfa", "talpszelemen", "térdfali oszlop", "ferdedúc",
                  "szarufa", "derékszelemen", "könyökfa", "derékszelemen alatti könyökfa"]

print(torch.version.cuda)
print(torch.backends.cuda.is_built())
print(torch.cuda.is_available())

st.title("DB STT Manipulation - Demo")

# Upload audio file with streamlit element and path definition
audio_file_path = st.file_uploader("Upload", type=["wav", "mp3", "m4a", "ogg"])

# Record audio with streamlit_audio_recorder element and path definition
recorded_audio_file_path = st_audiorec()

# Initialize session state variables
if 'loadedtmodel' not in st.session_state and 'currentproc' not in st.session_state and 'cindex' not in st.session_state:
    st.session_state.loadedtmodel = None
    st.session_state.currentproc = "None"
    st.session_state.cindex = 3

# Create a placeholder for displaying the currently loaded model
proc_display_placeholder = st.sidebar.empty()

# Display the currently loaded model and processor
proc_display_placeholder.text("Currently loaded into: " + str(st.session_state.currentproc))

LoadButton = st.sidebar.button("Load Model")

reset_button = st.sidebar.button("Table Cell Reset")
# Use radio buttons to choose between the recorded and uploaded file
file_choice = st.sidebar.radio("Choose audio source", ["Uploaded File", "Recorded File"])

# Use radio buttons to choose between CUDA and CPU
Processing_choice = st.sidebar.radio("Choose processor", ["CUDA - High GPU VRAM Usage", "CPU + High RAM Usage"])

if reset_button:
    st.session_state.cindex = 3
    print(st.session_state.cindex)

# Load the selected model when the user clicks the "Load Model" button
if LoadButton:
    # Display a progress bar while loading the model and load the model based on processor selection
    with st.spinner(f"Loading STT model..."):
        if Processing_choice == "CUDA - High GPU VRAM Usage":
            if not torch.cuda.is_available():
                st.success(f"failed successfully")
                st.info(f"Try running pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121, works with newest nvidia drivers.")
                st.error(f"otherwise, get an expensive GPU, poor guy lmao")
            st.session_state.loadedtmodel = transformers.pipeline(model="DrFumes/w2v-bert-2.0-HUN-CV16.1-test-incl",
                                                                  use_fast=True, device=0)

        elif Processing_choice == "CPU + High RAM Usage":
            st.session_state.loadedtmodel = transformers.pipeline(model="DrFumes/w2v-bert-2.0-HUN-CV16.1-test-incl",
                                                                  use_fast=True, device=-1)

    st.success(f"STT model loaded successfully!")

    # Update the displayed model load info after loading the model
    st.session_state.currentproc = Processing_choice
    proc_display_placeholder.text("Currently loaded into: " + Processing_choice)

# Transcribe process
if st.sidebar.button("Transcribe"):
    if (audio_file_path is not None and file_choice == "Uploaded File") or \
            (recorded_audio_file_path is not None and file_choice == "Recorded File"):
        st.sidebar.success("Transcribing...")

        # Save the uploaded or recorded file to a temporary location
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, "temp_audio_file.wav")

        if file_choice == "Recorded File" and recorded_audio_file_path:
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(recorded_audio_file_path)  # Read the content of the file
            # Convert the audio to 16 kHz sample rate
            waveform, sample_rate = torchaudio.load(temp_file_path)
            resampler = Resample(orig_freq=sample_rate, new_freq=16000)
            resampled_waveform = resampler(waveform)
            # Save the resampled audio to the temporary location
            torchaudio.save(temp_file_path, resampled_waveform, 16000)

        elif file_choice == "Uploaded File" and audio_file_path:
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(audio_file_path.read())  # Read the content of the file
            # Convert the audio to 16 kHz sample rate
            waveform, sample_rate = torchaudio.load(temp_file_path)
            resampler = Resample(orig_freq=sample_rate, new_freq=16000)
            resampled_waveform = resampler(waveform)
            # Save the resampled audio to the temporary location
            torchaudio.save(temp_file_path, resampled_waveform, 16000)

        # Check if the model is loaded before using it
        if st.session_state.loadedtmodel is not None:
            # Transcribe magic using the temporary file path
            with st.spinner(f"Detecting commands..."):
                transcription = st.session_state.loadedtmodel(temp_file_path)["text"]
            st.sidebar.success("Transcription done.")
            st.markdown(transcription)

            with st.spinner(f"Detecting commands..."):
                detected_phrases = check(transcription, target_phrases)

                if detected_phrases:
                    for index, (phrase, confidence) in enumerate(detected_phrases, start=1):
                        print(
                            f"The string contains '{phrase}' or a similar phrase with a confidence level of {confidence:.2f}.")
                    st.info(f"Command(s) detected: {detected_phrases}.")
                    st.success(f"Processing commands...")
                else:
                    print("The string does not contain any of the specified phrases.")
                    st.info(f"No commands detected, letting the program know.")

                print(detected_phrases)

            if detected_phrases:
                with st.spinner(f"Detecting intentions of commands..."):
                    # Create a set of detected phrases for easier checking
                    detected_set = set(phrase for phrase, confidence in detected_phrases)

                    # Updating cells based on specific phrases detected

                    # Row Titles
                    if "szarufa" or "szarufa" in detected_set:
                        sh.sheet1.update_cell(st.session_state.cindex, 1, 'Szarufa')
                        print("Detected 'szarufa'")
                        st.info(f"Command detected: 'szarufa'. Sent to DB.")
                        
                    if "vápa szarufa" or "vápaszarufa" in detected_set:
                        sh.sheet1.update_cell(st.session_state.cindex, 1, 'Vápaszarufa')
                        print("Detected 'vápa szarufa'")
                        st.info(f"Command detected: 'vápaszarufa'. Sent to DB.")

                    if "gerenda" in detected_set:
                        sh.sheet1.update_cell(st.session_state.cindex, 1, 'Gerenda')
                        print("Detected 'gerenda'")
                        st.info(f"Command detected: 'gerenda'. Sent to DB.")

                    if "székoszlop" in detected_set:
                        sh.sheet1.update_cell(st.session_state.cindex, 1, 'Székoszlop')
                        print("Detected 'székoszlop'")
                        st.info(f"Command detected: 'székoszlop'. Sent to DB.")
                    
                    if "talpszelemen" in detected_set:
                        sh.sheet1.update_cell(st.session_state.cindex, 1, 'Talpszelemen')
                        print("Detected 'talpszelemen'")
                        st.info(f"Command detected: 'talpszelemen'. Sent to DB.")

                    if "térdfali oszlop" in detected_set:
                        sh.sheet1.update_cell(st.session_state.cindex, 1, 'Térdfali oszlop')
                        print("Detected 'térdfali oszlop'")
                        st.info(f"Command detected: 'térdfali oszlop'. Sent to DB.")
                        
                    if "ferdedúc" in detected_set:
                        sh.sheet1.update_cell(st.session_state.cindex, 1, 'Ferdedúc')
                        print("Detected 'ferdedúc'")
                        st.info(f"Command detected: 'ferdedúc'. Sent to DB.")

                    if "derékszelemen" in detected_set:
                        sh.sheet1.update_cell(st.session_state.cindex, 1, 'Derékszelemen')
                        print("Detected 'derékszelemen'")
                        st.info(f"Command detected: 'derékszelemen'. Sent to DB.")
                        
                    if "könyökfa" in detected_set:
                        sh.sheet1.update_cell(st.session_state.cindex, 1, 'Könyökfa')
                        print("Detected 'könyökfa'")
                        st.info(f"Command detected: 'könyökfa'. Sent to DB.")

                    if "derékszelemen alatti könyökfa" in detected_set:
                        sh.sheet1.update_cell(st.session_state.cindex, 1, 'Derékszelemen alatti könyökfa')
                        print("Detected 'derékszelemen alatti könyökfa'")
                        st.info(f"Command detected: 'derékszelemen alatti könyökfa'. Sent to DB.")

                    # Damage
                    if "korhadt" in detected_set:
                        sh.sheet1.update_cell(st.session_state.cindex, 4, 'X')
                        print("Detected 'korhadt'")
                        st.info(f"Command detected: 'korhadt'. Sent to DB.")

                    if "gomba kár" or "gombakár" in detected_set:
                        sh.sheet1.update_cell(st.session_state.cindex, 2, 'X')
                        print("Detected 'gombakár'")
                        st.info(f"Command detected: 'gombakár'. Sent to DB.")

                    if "rovar kár" or "rovarkár" in detected_set:
                        sh.sheet1.update_cell(st.session_state.cindex, 3, 'X')
                        print("Detected 'rovarkár'")
                        st.info(f"Command detected: 'rovarkár'. Sent to DB.")

                    # Depth of Damage
                    if "felületi" in detected_set:
                        sh.sheet1.update_cell(st.session_state.cindex, 8, 'X')
                        print("Detected 'felületi'")
                        st.info(f"Command detected: 'felületi'. Sent to DB.")
                        
                    if "kismélységű" or "kis mélységű" in detected_set:
                        sh.sheet1.update_cell(st.session_state.cindex, 9, 'X')
                        print("Detected 'kismélységű'")
                        st.info(f"Command detected: 'kismélységű'. Sent to DB.")

                    if "egyharmad keresztmetszetű" or "egy harmad keresztmetszetű" or "egyharmadkeresztmetszetű" in detected_set:
                        sh.sheet1.update_cell(st.session_state.cindex, 10, 'X')
                        print("Detected 'egyharmad keresztmetszetű'")
                        st.info(f"Command detected: 'egyharmad keresztmetszetű'. Sent to DB.")

                    if "fél keresztmetszetű" or "félkeresztmetszetű" in detected_set:
                        sh.sheet1.update_cell(st.session_state.cindex, 11, 'X')
                        print("Detected 'fél keresztmetszetű'")
                        st.info(f"Command detected: 'félkeresztmetszetű'. Sent to DB.")

                    # Course of Action, special handling considering the presence of "esetlegesen" comment intent
                    if "bárdolás" in detected_set and "esetlegesen bárdolás" not in detected_set:
                        sh.sheet1.update_cell(st.session_state.cindex, 12, 'X')
                        print("Detected 'bárdolás' without 'esetlegesen bárdolás'")
                        st.info(f"Command detected: 'bárdolás'. Sent to DB.")
                    if "vegykezelés" in detected_set and "esetlegesen vegykezelés" not in detected_set:
                        sh.sheet1.update_cell(st.session_state.cindex, 12, 'X')
                        print("Detected 'vegykezelés' without 'esetlegesen vegykezelés'")
                        st.info(f"Command detected: 'vegykezelés'. Sent to DB.")

                    if "csere" in detected_set and "esetlegesen csere" not in detected_set:
                        sh.sheet1.update_cell(st.session_state.cindex, 13, 'X')
                        print("Detected 'csere' without 'esetlegesen csere'")
                        st.info(f"Command detected: 'csere'. Sent to DB.")

                    if "megerősítés" in detected_set and "esetlegesen megerősítés" not in detected_set:
                        sh.sheet1.update_cell(st.session_state.cindex, 15, 'X')
                        print("Detected 'megerősítés' without 'esetlegesen megerősítés'")
                        st.info(f"Command detected: 'megerősítés'. Sent to DB.")

                    # Comments
                    if "esetlegesen bárdolás" in detected_set:
                        sh.sheet1.update_cell(st.session_state.cindex, 16, 'esetleg bárdolás')
                        print("Detected 'esetlegesen bárdolás'")
                        st.info(f"Command detected: 'esetlegesen bárdolás' comment intent. Sent to DB.")

                    if "esetlegesen vegykezelés" in detected_set:
                        sh.sheet1.update_cell(st.session_state.cindex, 16, 'esetleg vegykezelés')
                        print("Detected 'esetlegesen vegykezelés'")
                        st.info(f"Command detected: 'Esetlegesen vegykezelés' comment intent. Sent to DB.")

                    if "esetlegesen csere" in detected_set:
                        sh.sheet1.update_cell(st.session_state.cindex, 16, 'esetleg csere')
                        print("Detected 'esetlegesen csere'")
                        st.info(f"Command detected: 'Esetlegesen csere' comment intent. Sent to DB.")

                    if "esetlegesen megerősítés" in detected_set:
                        sh.sheet1.update_cell(st.session_state.cindex, 16, 'esetleg megerősítés')
                        print("Detected 'esetlegesen megerősítés'")
                        st.info(f"Command detected: 'esetlegesen megerősítés' comment intent. Sent to DB.")

                    st.session_state.cindex = st.session_state.cindex + 1
                    print(st.session_state.cindex)
                st.success(f"All commands processed, DB updated accordingly.")

            else:
                print("The string does not contain any of the specified phrases.")
                st.success(f"No action required, no commands detected.")

        else:
            st.sidebar.error("Model not loaded. Please load the STT model first.")
    else:
        st.sidebar.error("Select a valid, non-empty audio source queue (Uploaded or Recorded).")

# Display and play your little audio things based on record/upload selection
st.sidebar.header("Play Audio File")
if audio_file_path and file_choice == "Uploaded File":
    st.sidebar.audio(audio_file_path)
elif recorded_audio_file_path and file_choice == "Recorded File":
    st.sidebar.audio(recorded_audio_file_path)
    