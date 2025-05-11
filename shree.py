"""
Shree - Advanced AI Desktop Assistant

This AI assistant can monitor screens, automate tasks, send messages, and provide
intelligent responses using various Python libraries and AI models.
"""
import os
import datetime
import logging
import time
import webbrowser
import cv2
import pyautogui
import mediapipe as mp
import psutil
import qrcode
import requests
import base64
import subprocess
from dotenv import load_dotenv
import openai
import pywhatkit
from PIL import Image, ImageDraw, ImageFont

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Environment setup
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load environment variables from .env file
load_dotenv("openai.env")
api_key = os.getenv("OPENAI_API_KEY", default=None)

if api_key is None:
    raise ValueError("OPENAI_API_KEY is not set in the environment or .env file.")

if not api_key:
    print("API key not loaded.")
else:
    print("✅ API key loaded successfully!")
    print("🔒 Key preview:", api_key[:5] + "..." + api_key[-5:])

# Set OpenAI API key
openai.api_key = api_key

# Placeholder speak and take_command (replace with your implementation)
def speak(text):
    print("Shree:", text)

def take_command():
    return input("You: ").lower()

# AI Response Function
def generate_ai_response(prompt):
    try:
        pass  # Add your code here
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant named Shree."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500
        )
        return response.choices[0].message["content"]
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return f"I had trouble connecting to my AI capabilities: {str(e)}"

# Image Analysis Function
def analyze_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            base64_image = base64.b64encode(image_data).decode('utf-8')

        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this image in detail and describe its key elements, context, and any notable aspects."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            max_tokens=500
        )
        return response.choices[0].message["content"]
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        return f"I couldn't analyze the image due to a technical error: {str(e)}"

# Configuration
ASSISTANT_NAME = "Shree"
VOICE_RATE = 170
VOICE_INDEX = 1  # Female voice
LANGUAGE = 'en-in'
PAUSE_THRESHOLD = 1
LISTEN_TIMEOUT = 5

# File paths
DOWNLOAD_DIR = os.path.expanduser("~/Downloads")
DOCUMENTS_DIR = os.path.expanduser("~/Documents")
SCREENSHOT_DIR = os.path.join(os.path.expanduser("~"), "Screenshots")
QR_CODE_DIR = os.path.join(os.path.expanduser("~"), "QRCodes")

# Create directories
for directory in [SCREENSHOT_DIR, QR_CODE_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Play music function
def play_music():
    speak("What kind of music do you want to play? Gujarati, Hindi, or English?")
    lang = take_command()

    if 'hindi' in lang:
        speak("Which singer do you want? Arman Malik, Arijit Singh, Atif Aslam or Shreya Ghoshal?")
        singer = take_command()

        if any(name in singer for name in ['arman', 'armaan']):
            song = "Armaan Malik songs"
        elif 'arijit' in singer:
            song = "Arijit Singh songs"
        elif 'atif' in singer:
            song = "Atif Aslam songs"
        elif 'shreya' in singer or 'ghoshal' in singer:
            song = "Shreya Ghoshal songs"
        else:
            song = "Hindi songs"
        speak(f"Playing {song}")
        pywhatkit.playonyt(song)

    elif 'gujarati' in lang:
        speak("Do you want Garba or Gujarati movie songs?")
        g_type = take_command()

        if 'garba' in g_type:
            song = "Gujarati Garba"
        else:
            song = "Gujarati movie songs"
        speak(f"Playing {song}")
        pywhatkit.playonyt(song)

    elif 'english' in lang:
        speak("Playing a random English song.")
        pywhatkit.playonyt("random English song")

    else:
        speak(f"Searching YouTube for {lang}")
        pywhatkit.playonyt(lang)

# Utility Functions
import datetime
import pytz

def get_time_for_location(location):
    try:
        # Dictionary mapping common city/country names to timezones
        timezone_map = {
            "india": "Asia/Kolkata",
            "new york": "America/New_York",
            "london": "Europe/London",
            "japan": "Asia/Tokyo",
            "australia": "Australia/Sydney",
            "canada": "America/Toronto",
            "germany": "Europe/Berlin",
            "china": "Asia/Shanghai",
            "dubai": "Asia/Dubai",
        }

        # Normalize location
        location = location.lower().strip()
        timezone_str = timezone_map.get(location)

        if timezone_str:
            timezone = pytz.timezone(timezone_str)
            now = datetime.datetime.now(timezone)
            current_time = now.strftime("%I:%M %p")
            current_date = now.strftime("%B %d, %Y")
            return current_time, current_date
        else:
            return None, None
    except Exception as e:
        print(f"Error: {e}")
        return None, None



def take_screenshot(region=None):
        screenshot = pyautogui.screenshot(region=region)
        if not os.path.exists(SCREENSHOT_DIR):
            os.makedirs(SCREENSHOT_DIR)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = os.path.join(SCREENSHOT_DIR, f"screenshot_{timestamp}.png")

def generate_qr_code(data, output_path=None):
    try:
        qr_img = qrcode.make(data)
        if not os.path.exists(QR_CODE_DIR):
            os.makedirs(QR_CODE_DIR)
        if output_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(QR_CODE_DIR, f"qr_{timestamp}.png")
        qr_img.save(output_path)
        return output_path
    except Exception as e:
        logger.error(f"Error generating QR code: {e}")
        return None
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        qr_img.save(output_path)
        return qr_img
    except Exception as e:
        logger.error(f"Error generating QR code: {e}")
        return None

def check_internet_connection():
    try:
        requests.get("https://www.google.com", timeout=5)
        return True
    except:
        return False


def generate_handwriting(text, output_path=None, font_size=25, font_path="arial.ttf"):
    try:
        img = Image.new("RGB", (800, 400), color="white")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            logger.warning("Custom font not found. Using default font.")
            font = ImageFont.load_default()
        draw.text((50, 100), text, fill="black", font=font)
        if not os.path.exists(DOCUMENTS_DIR):
            os.makedirs(DOCUMENTS_DIR)
        if output_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(DOCUMENTS_DIR, f"handwriting_{timestamp}.png")
        img.save(output_path)
        return output_path
    except Exception as e:
        logger.error(f"Error generating handwriting: {e}")
        return None
    try:
        img = Image.new("RGB", (800, 400), color="white")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            logger.warning("Custom font not found. Using default font.")
            font = ImageFont.load_default()
        draw.text((50, 100), text, fill="black", font=font)
        if not os.path.exists(DOCUMENTS_DIR):
            os.makedirs(DOCUMENTS_DIR)
        if output_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(DOCUMENTS_DIR, f"handwriting_{timestamp}.png")
        img.save(output_path)
        return output_path
    except Exception as e:
        logger.error(f"Error generating handwriting: {e}")
def get_battery_info():
    try:
        battery = psutil.sensors_battery()
        if battery:
            return {
                "percent": battery.percent,
                "power_plugged": battery.power_plugged,
                "time_left": str(datetime.timedelta(seconds=battery.secsleft)) if battery.secsleft != psutil.POWER_TIME_UNLIMITED else "Unlimited"
            }
        else:
            speak("Battery information is not available on this system.")
            return None
    except Exception as e:
        logger.error(f"Battery info error: {e}")
        return None
    try:
        battery = psutil.sensors_battery()
        if battery:
            return {
                "percent": battery.percent,
                "power_plugged": battery.power_plugged,
                "time_left": str(datetime.timedelta(seconds=battery.secsleft)) if battery.secsleft != psutil.POWER_TIME_UNLIMITED else "Unlimited"
            }
        return None
    except Exception as e:
        logger.error(f"Battery info error: {e}")
        return None

def get_system_info():
    try:
        return {
            "system": os.name,
            "cpu_cores": psutil.cpu_count(logical=False),
            "cpu_threads": psutil.cpu_count(logical=True),
            "memory_total": round(psutil.virtual_memory().total / (1024**3), 2),
            "memory_available": round(psutil.virtual_memory().available / (1024**3), 2),
            "disk_total": round(psutil.disk_usage('/').total / (1024**3), 2),
            "disk_free": round(psutil.disk_usage('/').free / (1024**3), 2)
        }
    except Exception as e:
        logger.error(f"System info error: {e}")
        return None

# -------- Desktop Automation Functions --------
def open_application(app_name):
    try:
        app = app_name.lower()
        if app in ["chrome", "google chrome"]:
            webbrowser.open("https://www.google.com")
        elif app == "firefox":
            webbrowser.open("https://www.mozilla.org")
        elif app in ["notepad", "text editor"]:
            if os.name == "nt":
                os.system("notepad")
            else:
                os.system("gedit")
        elif app in ["terminal", "command prompt", "cmd"]:
            if os.name == "nt":
                os.system("start cmd")
            else:
                os.system("gnome-terminal")
        elif app == "whatsapp":
            webbrowser.open("https://web.whatsapp.com/")
        elif app in ["gmail", "email"]:
            webbrowser.open("https://mail.google.com/")
        elif app == "youtube":
            webbrowser.open("https://www.youtube.com/")
        else:
            os.system(app_name)
        return True
    except Exception as e:
        logger.error(f"Failed to open application '{app_name}': {e}")
        speak(f"Failed to open {app_name}")
        return False

def close_application(app_name):
    try:
        found = False
        for proc in psutil.process_iter(['pid', 'name']):
            if app_name.lower() in proc.info['name'].lower():
                proc.terminate()
                found = True
        return found
    except Exception as e:
        logger.error(f"Error closing '{app_name}': {e}")
        speak(f"Could not close {app_name}")
        return False

def open_camera():
    try:
        if os.name == 'nt':
            os.startfile("microsoft.windows.camera:")
        elif os.name == 'posix':
            os.system("xdg-open /dev/video0")
        speak("Camera opened.")
    except Exception as e:
        logger.error(f"Error opening camera: {e}")
        speak("I couldn't open the camera.")

def type_text(text):
    try:
        pyautogui.typewrite(text)
        return True
    except Exception as e:
        logger.error(f"Typing error: {e}")
        return False

def press_key(key):
    try:
        pyautogui.press(key)
        return True
    except Exception as e:
        logger.error(f"Key press error for '{key}': {e}")
        return False

def press_key_combination(keys):
    try:
        if isinstance(keys, str):
            keys = keys.split('+')
        pyautogui.hotkey(*keys)
        return True
    except Exception as e:
        logger.error(f"Key combination error '{keys}': {e}")
        return False

# -------- Messaging Functions --------

def send_whatsapp_message(contact, message):
    try:
        if not contact.startswith("+") or not contact[1:].isdigit():
            speak("Please provide the full phone number with country code, like +919876543210.")
            return

        if not message:
            speak("The message is empty.")
            return

        speak("Opening WhatsApp Web...")
        webbrowser.open("https://web.whatsapp.com")
        time.sleep(15)  # Wait for WhatsApp Web to load

        speak(f"Sending your message to {contact}")
        pywhatkit.sendwhatmsg_instantly(contact, message, wait_time=10, tab_close=False)
        speak("Message sent successfully.")
    except Exception as e:
        speak("Sorry, I couldn't send the message.")
        logger.error(f"WhatsApp send error: {e}")


def open_messaging_app(app_name):
    try:
        app = app_name.lower()
        urls = {
            "whatsapp": "https://web.whatsapp.com/",
            "gmail": "https://mail.google.com/",
            "email": "https://mail.google.com/",
            "slack": "https://app.slack.com/",
            "teams": "https://teams.microsoft.com/",
            "discord": "https://discord.com/app",
            "telegram": "https://web.telegram.org/"
        }

        for name, url in urls.items():
            if name in app:
                webbrowser.open(url)
                return True

        speak(f"I don't know how to open {app_name}")
        return False
    except Exception as e:
        logger.error(f"Messaging app open error for '{app_name}': {e}")
        speak("Something went wrong while opening the app.")
        return False

# -------- Input Handling --------
def chat_input():
    while True:
        user_input = input("You (chat): ").lower()
        if user_input == "exit":
            global input_mode
            input_mode = "voice"
            speak("Voice mode activated.")
            return "none"
        return user_input

def get_input(mode):
    return chat_input() if mode == "chat" else take_command()

# -------- Voice Interface Functions --------

try:
    import pyttsx3
    import speech_recognition as sr
    import logging

    # Setup logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    # Assistant configuration
    ASSISTANT_NAME = "Shree"
    VOICE_RATE = 160
    VOICE_INDEX = 1
    LANGUAGE = 'en-in'
    PAUSE_THRESHOLD = 1
    LISTEN_TIMEOUT = 5

    # Initialize TTS engine
    engine = pyttsx3.init()
    engine.setProperty('rate', VOICE_RATE)

    voices = engine.getProperty('voices')
    if voices and len(voices) > VOICE_INDEX:
        engine.setProperty('voice', voices[VOICE_INDEX].id)
    else:
        logger.warning("Requested voice index not found, using default voice.")

    def speak(text):
        """Speak the given text using TTS engine."""
        print(f"{ASSISTANT_NAME}: {text}")
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            logger.error(f"Error in text-to-speech: {e}")
            print(f"Error speaking: {e}")

    def take_command():
        """Listen for a voice command and return it as lowercase text."""
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            recognizer.pause_threshold = PAUSE_THRESHOLD
            try:
                audio = recognizer.listen(source, timeout=LISTEN_TIMEOUT)
                query = recognizer.recognize_google(audio, language=LANGUAGE)
                print(f"You said: {query}")
                return query.lower()
            except sr.WaitTimeoutError:
                speak("I didn't hear anything. Please try again.")
                return "none"
            except sr.UnknownValueError:
                speak("Sorry, I didn't understand that.")
                return "none"
            except sr.RequestError as e:
                speak("I'm having trouble connecting to the recognition service.")
                logger.error(f"Speech recognition service error: {e}")
                return "error"

except ImportError as e:
    # If voice modules are not available
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    ASSISTANT_NAME = "Shree"

    def speak(text):
        print(f"{ASSISTANT_NAME}: {text} (TTS module missing)")
        logger.error(f"Text-to-speech module missing: {e}")
        print(f"Error speaking: {e}")

def start_virtual_mouse():
    import cv2
    import mediapipe as mp
    import pyautogui
    import random
    import os
    from pynput.mouse import Button, Controller
    import util  # You must create this module with get_angle() and get_distance()

    mouse = Controller()
    screen_width, screen_height = pyautogui.size()

    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    hands = mp_hands.Hands(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        max_num_hands=1
    )

    if not os.path.exists('screenshots'):
        os.makedirs('screenshots')

    def move_mouse(tip):
        x = int(tip.x * screen_width)
        y = int(tip.y * screen_height)
        pyautogui.moveTo(x, y)

    def detect_gesture(frame, landmarks):
        if len(landmarks) < 21:
            return

        tip = landmarks[8]          # Index finger tip
        thumb_tip = landmarks[4]
        thumb_base = landmarks[5]

        thumb_index_dist = util.get_distance(
            (thumb_tip.x, thumb_tip.y),
            (thumb_base.x, thumb_base.y)
        )

        def angle(a, b, c):
            return util.get_angle((a.x, a.y), (b.x, b.y), (c.x, c.y))

        index_angle = angle(landmarks[5], landmarks[6], landmarks[8])
        middle_angle = angle(landmarks[9], landmarks[10], landmarks[12])

        # Move mouse
        if thumb_index_dist < 0.05 and index_angle > 90:
            move_mouse(tip)

        # Left click
        elif index_angle < 50 and middle_angle > 90 and thumb_index_dist > 0.05:
            mouse.click(Button.left)
            cv2.putText(frame, "Left Click", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Right click
        elif middle_angle < 50 and index_angle > 90 and thumb_index_dist > 0.05:
            mouse.click(Button.right)
            cv2.putText(frame, "Right Click", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Double click
        elif index_angle < 50 and middle_angle < 50 and thumb_index_dist > 0.05:
            pyautogui.doubleClick()
            cv2.putText(frame, "Double Click", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Screenshot
        elif index_angle < 50 and middle_angle < 50 and thumb_index_dist < 0.05:
            filename = f'screenshots/screenshot_{random.randint(1000,9999)}.png'
            pyautogui.screenshot(filename)
            cv2.putText(frame, "Screenshot Taken", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cap = cv2.VideoCapture(0)

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
                detect_gesture(frame, landmarks.landmark)

            cv2.imshow('Virtual Mouse', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

###play game

import random

number_words = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
}

def word_to_number(text):
    text = text.lower().strip()
    if text.isdigit():
        return int(text)
    elif text in number_words:
        return number_words[text]
    else:
        return None

def play_game():
    secret_number = random.randint(1, 10)
    attempts = 0

    speak("Welcome to the Number Guessing Game!")
    speak("I'm thinking of a number between 1 and 10. Try to guess it.")

    while True:
        speak("Please say your guess.")
        guess_input = take_command()

        guess = word_to_number(guess_input)
        if guess is None:
            speak("That's not a valid number. Please try again.")
            continue

        attempts += 1

        if guess < secret_number:
            speak("Too low! Try again.")
        elif guess > secret_number:
            speak("Too high! Try again.")
        else:
            speak(f"Correct! You guessed the number in {attempts} attempts.")
            break



# -------- File Management Functions --------
def list_directory(directory=None):
    """List files and folders in the specified directory."""
    try:
        if directory is None:
            directory = os.getcwd()
        
        if not os.path.exists(directory):
            print(f"Directory {directory} does not exist.")
            return []
        
        items = os.listdir(directory)
        
        # Separate files and directories
        dirs = [f"{item}/" for item in items if os.path.isdir(os.path.join(directory, item))]
        files = [item for item in items if os.path.isfile(os.path.join(directory, item))]
        
        # Sort alphabetically
        dirs.sort()
        files.sort()
        
        return dirs + files
        
    except Exception as e:
        logger.error(f"Error listing directory {directory}: {e}")
        print(f"I encountered an error while listing the directory: {str(e)}")
        return []

def search_files(search_term, directory=None, file_extension=None):
    """Search for files matching the given criteria."""
    try:
        if directory is None:
            directory = os.getcwd()
        
        if not os.path.exists(directory):
            print(f"Directory {directory} does not exist.")
            return []
        
        # Build the search pattern
        pattern = search_term.lower()
        if file_extension and not file_extension.startswith('.'):
            file_extension = f".{file_extension}"
        
        # Perform the search
        matches = []
        for root, _, files in os.walk(directory):
            for filename in files:
                # If extension filter is applied
                if file_extension and not filename.lower().endswith(file_extension.lower()):
                    continue
                    
                # If filename contains search term
                if pattern in filename.lower():
                    matches.append(os.path.join(root, filename))
        
        return matches
        
    except Exception as e:
        logger.error(f"Error searching for {search_term} in {directory}: {e}")
        print(f"I encountered an error while searching for files: {str(e)}")
        return []
    

def create_directory(directory_path):
    """Create a new directory."""
    try:
        if os.path.exists(directory_path):
            print(f"Directory {directory_path} already exists.")
            return False
        
        os.makedirs(directory_path)
        print(f"Directory {directory_path} created successfully.")
        return True
        
    except Exception as e:
        logger.error(f"Error creating directory {directory_path}: {e}")
        print(f"I couldn't create the directory: {str(e)}")
        return False

def delete_file(file_path):
    """Delete a file."""
    try:
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist.")
            return False
        
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            import shutil
            shutil.rmtree(file_path)
        
        print(f"Deleted {file_path}.")
        return True
        
    except Exception as e:
        logger.error(f"Error deleting {file_path}: {e}")
        print(f"I couldn't delete the file: {str(e)}")
        return False

##calculator

def calculator():
    speak("Do you want to add, subtract, or multiply?")
    operation = take_command().lower()

    if operation not in ["add", "subtract", "multiply"]:
        speak("Sorry, I can only add, subtract, or multiply.")
        return

    speak("How many numbers do you want to use?")
    try:
        count = int(take_command())
    except ValueError:
        speak("I didn't understand the number. Please try again.")
        return

    numbers = []
    for i in range(count):
        speak(f"Please say number {i + 1}")
        try:
            number = float(take_command())
            numbers.append(number)
        except ValueError:
            speak("That wasn't a valid number. Skipping.")
            continue

    if not numbers:
        speak("No valid numbers received. Cancelling calculator.")
        return

    result = numbers[0]
    if operation == "add":
        result = sum(numbers)
    elif operation == "subtract":
        for num in numbers[1:]:
            result -= num
    elif operation == "multiply":
        for num in numbers[1:]:
            result *= num

    speak(f"The result is {result}")

#more features

#to do list
import time

def add_reminder():
    speak("What should I remind you?")
    reminder = take_command()
    speak("In how many minutes should I remind you?")
    try:
        minutes = int(take_command())
        speak(f"Okay, I will remind you to {reminder} in {minutes} minutes.")
        time.sleep(minutes * 60)
        speak(f"This is your reminder: {reminder}")
    except:
        speak("Sorry, I didn't understand the time.")

def show_todo_list():
    try:
        with open("todo_list.txt", "r") as f:
            tasks = f.readlines()
        if tasks:
            speak("Here are your to-do items.")
            for task in tasks:
                speak(task.strip())
        else:
            speak("Your to-do list is empty.")
    except FileNotFoundError:
        speak("You don't have a to-do list yet.")

def add_todo():
    speak("What do you want to add to your to-do list?")
    task = take_command()
    with open("todo_list.txt", "a") as f:
        f.write(task + "\n")
    speak(f"I have added {task} to your to-do list.")

#note tasking

def take_note():
    speak("What should I note down?")
    note = take_command()
    with open("notes.txt", "a") as f:
        f.write(note + "\n")
    speak("Note saved.")

#calender events

def add_event():
    speak("What is the event?")
    event = take_command()
    speak("When is it? For example, say 'on Friday at 4 PM'")
    time_info = take_command()
    with open("calendar.txt", "a") as f:
        f.write(f"{time_info} - {event}\n")
    speak("Event added to your calendar.")

def show_events():
    speak("Here are your upcoming events.")
    try:
        with open("calendar.txt", "r") as f:
            events = f.readlines()
        for event in events:
            speak(event.strip())
    except FileNotFoundError:
        speak("You don't have any events yet.")

#wifi control

import os

def toggle_wifi(turn_on=True):
    command = "netsh interface set interface name=\"Wi-Fi\" admin=enabled" if turn_on else "netsh interface set interface name=\"Wi-Fi\" admin=disabled"
    os.system(command)
    speak("Wi-Fi turned on." if turn_on else "Wi-Fi turned off.")

def toggle_hotspot(turn_on=True):
    if turn_on:
        os.system("netsh wlan set hostednetwork mode=allow ssid=MyHotspot key=12345678")
        os.system("netsh wlan start hostednetwork")
        speak("Hotspot started.")
    else:
        os.system("netsh wlan stop hostednetwork")
        speak("Hotspot stopped.")

#shutdown 

def shutdown():
    speak("Shutting down the computer.")
    os.system("shutdown /s /t 1")

def restart():
    speak("Restarting the computer.")
    os.system("shutdown /r /t 1")

def sleep():
    speak("Putting the computer to sleep.")
    os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")

#handwritten text image

from PIL import Image, ImageDraw, ImageFont

def handwritten_text():
    speak("What do you want me to write?")
    text = take_command()

    img = Image.new('RGB', (800, 200), color='white')
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 36)
    draw.text((50, 80), text, fill='black', font=font)
    img.save("handwritten_output.png")
    speak("I have saved your handwritten image.")

#drawing pad

import cv2
import numpy as np

def draw_pad():
    speak("Opening drawing pad. Use your mouse to draw. Press ESC to exit.")
    canvas = np.ones((600, 800, 3), dtype="uint8") * 255
    drawing = False
    last_point = None

    def draw(event, x, y, flags, param):
        nonlocal drawing, last_point
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            last_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            cv2.line(canvas, last_point, (x, y), (0, 0, 0), 2)
            last_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    cv2.namedWindow("Doodle")
    cv2.setMouseCallback("Doodle", draw)

    while True:
        cv2.imshow("Doodle", canvas)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()

#joke and riddles
import pyjokes
import random

def tell_joke():
    joke = pyjokes.get_joke()
    speak(joke)

def tell_riddle():
    riddles = [
        ("What has keys but can't open locks?", "A piano."),
        ("What has hands but can’t clap?", "A clock."),
        ("What comes down but never goes up?", "Rain."),
    ]
    riddle, answer = random.choice(riddles)
    speak(riddle)
    time.sleep(5)
    speak("The answer is: " + answer)


##age detection

       
def detect_age_in_video():
    """Detects age from live webcam feed using pre-trained age detection model."""

    import cv2
    import numpy as np

    # Load the face detection model
    face_cascade = cv2.CascadeClassifier(
        r"C:/Users/Hetshree/AppData/Local/Programs/Python/Python312/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml"
    )

    # Load the age detection model
    age_net = cv2.dnn.readNetFromCaffe(
        r"C:/Users/Hetshree/Downloads/age_deploy.prototxt",
        r"C:/Users/Hetshree/Downloads/age_net.caffemodel"
    )

    # Age buckets as per the model
    AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
                   '(25-32)', '(38-43)', '(48-53)', '(60-100)']

    # Start video capture from the default webcam
    video_cap = cv2.VideoCapture(0)

    if not video_cap.isOpened():
        print("Error: Webcam not detected.")
        return

    while True:
        ret, video_frame = video_cap.read()
        if not ret:
            print("Error: Failed to capture video frame.")
            break

        # Flip the frame horizontally for a mirror effect
        video_frame = cv2.flip(video_frame, 1)

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Loop through detected faces
        for (x, y, w, h) in faces:
            face_img = video_frame[y:y+h, x:x+w].copy()

            # Prepare blob for age model
            blob = cv2.dnn.blobFromImage(
                face_img, 1.0, (227, 227),
                (78.4263377603, 87.7689143744, 114.895847746),
                swapRB=False
            )
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = AGE_BUCKETS[age_preds[0].argmax()]

            # Draw rectangle and label
            label = f"Age: {age}"
            cv2.rectangle(video_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(video_frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Show video frame
        cv2.imshow("Age Detection", video_frame)

        # Break on pressing 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Cleanup
    video_cap.release()
    cv2.destroyAllWindows()


# -------- Main Assistant Function --------
def wish_user():
    """Greet the user based on the time of day."""
    hour = datetime.datetime.now().hour
    
    if 5 <= hour < 12:
        speak("Good Morning!")
    elif 12 <= hour < 17:
        speak("Good Afternoon!")
    elif 17 <= hour < 21:
        speak("Good Evening!")
    else:
        speak("Good Night!")
    
    speak(f"I am {ASSISTANT_NAME}. How can I help you today?")
    
    # Show system status
    battery_info = get_battery_info()
    if battery_info:
        if battery_info["power_plugged"]:
            speak(f"Your battery is at {battery_info['percent']}% and charging.")
        elif battery_info["percent"] < 20:
            speak(f"Warning: Your battery is low at {battery_info['percent']}%. Please connect the charger.")

def process_command(command):
    """Process user commands."""

    global input_mode
    
    if command == "none" or command == "error":
        return True
    
    logger.info(f"Processing command: {command}")

   

    
    # File management commands
    if "list files" in command or "show files" in command:
        directory = os.getcwd()
        if "in" in command:
            dir_parts = command.split("in")[1].strip().split()
            if dir_parts:
                directory = " ".join(dir_parts)
        
        files = list_directory(directory)
        if files:
            speak(f"Found {len(files)} items in {directory}:")
            for i, file in enumerate(files[:10]):  # Limit to first 10 items
                speak(f"{i+1}. {file}")
            if len(files) > 10:
                speak(f"And {len(files) - 10} more items.")
        else:
            speak(f"No files found in {directory} or the directory doesn't exist.")
    
    elif "search for file" in command or "find file" in command:
        search_term = command.split("file")[1].strip()
        speak(f"Searching for files containing '{search_term}'...")
        
        files = search_files(search_term)
        if files:
            speak(f"Found {len(files)} matching files:")
            for i, file in enumerate(files[:5]):  # Limit to first 5 items
                speak(f"{i+1}. {os.path.basename(file)}")
            if len(files) > 5:
                speak(f"And {len(files) - 5} more files.")
        else:
            speak(f"No files found containing '{search_term}'.")
    
    elif "create folder" in command or "create directory" in command:
        folder_name = command.split("folder" if "folder" in command else "directory")[1].strip()
        if create_directory(folder_name):
            speak(f"Folder '{folder_name}' created successfully.")
        else:
            speak(f"Failed to create folder '{folder_name}'.")
    
    elif "calculator" in command or "calculate" in command:
        speak("Opening calculator.")
        calculator()

    # Screen monitoring commands
    elif "take screenshot" in command:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = os.path.join(SCREENSHOT_DIR, f"screenshot_{timestamp}.png")
        screenshot = take_screenshot()
        if screenshot:
            screenshot.save(screenshot_path)
            speak(f"Screenshot saved to {screenshot_path}")
        else:
            speak("Failed to take screenshot.")
    
    # Automation commands
    elif "start virtual mouse" in command or "gesture control" in command:
        speak("Starting gesture-controlled mouse. Show your hand to the camera...")
        start_virtual_mouse()

    
    elif "open" in command:
        app_name = command.split("open")[1].strip()
        speak(f"Opening {app_name}...")
        if open_application(app_name):
            speak(f"{app_name} opened successfully.")
        else:
            speak(f"I couldn't open {app_name}.")
    
    elif "close" in command and ("application" in command or "program" in command or "app" in command):
        app_name = command.split("close")[1].strip()
        for term in ["application", "program", "app"]:
            app_name = app_name.replace(term, "").strip()
        
        speak(f"Closing {app_name}...")
        if close_application(app_name):
            speak(f"{app_name} closed successfully.")
        else:
            speak(f"I couldn't close {app_name}.")
    
    elif "type a command" in command:
        text = command.split("type")[1].strip()
        speak(f"Typing: {text}")
        type_text(text)
       
    # Messaging commands
    
    elif "open whatsapp" in command:
        open_messaging_app("whatsapp")
    
    elif "send message" in command or "send whatsapp" in command:
        speak("Please say the phone number including country code, like +919876543210.")
        contact = take_command().strip()

        if not contact.startswith("+") or not contact[1:].replace(" ", "").isdigit():
            speak("Invalid phone number format. Please try again with the correct format.")
            return

        speak("Now say the message.")
        message = take_command().strip()

        if not message or message.lower() == "none":
            speak("Message was not understood or was empty. Please try again.")
            return
    elif "time" in command:
        try:
            current_time = datetime.datetime.now().strftime("%I:%M %p")
            speak(f"The current time is {current_time}")
        except Exception as e:
            logger.error(f"Error fetching time: {e}")
            speak("I couldn't fetch the current time.")

    elif "date" in command:
        try:
            current_date = datetime.datetime.now().strftime("%B %d, %Y")
            speak(f"Today's date is {current_date}")
        except Exception as e:
            logger.error(f"Error fetching date: {e}")
            speak("I couldn't fetch today's date.")
            pywhatkit.sendwhatmsg_instantly(contact, message, wait_time=10, tab_close=False)
            speak("Message sent successfully.")
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            speak("Something went wrong while sending the message.")

    # Utility commands
    elif "time" in command:
        current_time = datetime.datetime.now().strftime("%I:%M %p")
        speak(f"The current time is {current_time}")

    elif "date" in command:
        current_date = datetime.datetime.now().strftime("%B %d, %Y")
        speak(f"Today's date is {current_date}")


    elif "remind me" in command:
        add_reminder()

    elif "to-do list" in command or "what's on my to do" in command:
        show_todo_list()

    elif "add to do" in command or "add task" in command:
        add_todo()

    elif "note" in command or "take a note" in command:
        take_note()

    elif "add event" in command or ("calendar" in command and "add" in command):
        add_event()

    elif "show calendar" in command or "events" in command:
        show_events()

    elif "turn on wifi" in command or "enable wifi" in command:
        toggle_wifi(turn_on=True)

    elif "turn off wifi" in command or "disable wifi" in command:
        toggle_wifi(turn_on=False)

    elif "turn on hotspot" in command or "enable hotspot" in command:
        toggle_hotspot(turn_on=True)

    elif "turn off hotspot" in command or "disable hotspot" in command:
        toggle_hotspot(turn_on=False)

    elif "shutdown" in command or "shut down" in command:
        shutdown()

    elif "restart" in command:
        restart()

    elif "sleep" in command:
        sleep()

    elif "handwritten" in command or "write image" in command:
        handwritten_text()

    elif "draw" in command or "doodle" in command:
        draw_pad()

    elif "tell me a joke" in command:
        tell_joke()

    elif "riddle" in command:
        tell_riddle()

    elif "play game" in command:
        play_game()

    elif "calculator" in command or "calculate" in command:
        calculator()

    elif "generate qr" in command or "create qr code" in command:
        speak("Please say the text for the QR code.")
        qr_text = take_command()
        if qr_text != "none":
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            qr_path = os.path.join(QR_CODE_DIR, f"qr_{timestamp}.png")
            qr_img = generate_qr_code(qr_text, qr_path)
            if qr_img:
                speak(f"QR code generated and saved to {qr_path}")
                # Open the image
                try:
                    if os.name == 'nt':  # Windows
                        os.startfile(qr_path)
                        subprocess.call(['xdg-open', qr_path])
                        subprocess.call(['xdg-open', qr_path])
                except Exception as e:
                    logger.error(f"Error opening QR code image: {e}")
            else:
                speak("Failed to generate QR code.")
    
 


    elif "generate handwriting" in command:
        speak("How many minutes do you need to speak your text? Maximum is 30 minutes.")
        try:
            duration_text = take_command()
            if duration_text and duration_text != "none":
                import re
                match = re.search(r'(\d+)', duration_text)
                minutes = int(match.group(1)) if match else 2  # Default to 2 if no number is detected
                if minutes > 30:
                    minutes = 30
                    speak("Maximum limit is 30 minutes. So I will wait for 30 minutes.")
                else:
                    speak(f"Okay, I will wait for {minutes} minutes. You can start speaking when ready.")
                
                end_time = time.time() + (minutes * 60)
                collected_text = ""

                while time.time() < end_time:
                    temp_text = take_command()
                    if temp_text and temp_text.lower() != "none":
                        collected_text += " " + temp_text
                        speak("Text received. Do you want to add more or should I generate the handwriting?")
                        followup = take_command()
                        if "generate" in followup or "done" in followup or "stop" in followup:
                            break
                        else:
                            speak("Okay, continue speaking.")

                if collected_text.strip():
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    img_path = os.path.join(DOWNLOAD_DIR, f"handwritten_{timestamp}.png")
                    img = generate_handwriting(collected_text.strip(), img_path)
                    if img:
                        speak(f"Handwritten image saved to {img_path}")
                        try:
                            if os.name == 'nt':
                                os.startfile(img_path)
                            else:
                                import subprocess
                                subprocess.call(['xdg-open', img_path])
                        except Exception as e:
                            logger.error(f"Error opening handwritten image: {e}")
                    else:
                        speak("Failed to generate handwriting.")
                else:
                    speak("No text was provided.")
        except Exception as e:
            logger.error(f"Error during handwriting input collection: {e}")
            speak("Something went wrong while collecting your input.")

    
    # System commands
    elif "system info" in command or "system status" in command:
        speak("Checking system information...")
        sys_info = get_system_info()
        if sys_info:
            speak(
                f"You are running {sys_info['system']} system. "
                f"You have {sys_info['cpu_cores']} CPU cores with {sys_info['cpu_threads']} threads. "
                f"Total memory is {sys_info['memory_total']} GB with {sys_info['memory_available']} GB available. "
                f"Disk space: {sys_info['disk_free']} GB free out of {sys_info['disk_total']} GB total."
            )
        else:
            speak("I couldn't retrieve system information.")
    
    elif "cpu use" in command:
        cpu_percent = psutil.cpu_percent(interval=1)
        speak(f"Current CPU usage is {cpu_percent} percent.")

    elif "memory" in command:
        memory = psutil.virtual_memory()
        speak(f"Memory usage is {memory.percent} percent. {memory.available / (1024**3):.1f} GB available out of {memory.total / (1024**3):.1f} GB total.")
    
    # AI and knowledge commands
    elif "who are you" in command or "what can you do" in command:
        speak(
            f"I am {ASSISTANT_NAME}, an AI desktop assistant. I can help you with tasks like "
            "monitoring your screen, automating repetitive tasks, sending messages, "
            "managing files, and answering questions. Feel free to ask "
            "me anything or tell me what you'd like me to do!"
        )

    elif "play music" in command or "music" in command:
            play_music()
    
    elif "open camera" in command or "start camera" in command:
        open_camera()
    
    elif 'voice mode' in command or 'talk mode' in command:
        input_mode = "voice"
        speak("Switched to voice mode.")

    # Exit command
    elif "exit" in command or "stop" in command or "goodbye" in command:
        speak("Goodbye! Have a nice day.") 
        return False
    
    elif "age detection" in command:
        speak("Starting age detection...")
        detect_age_in_video()

    elif "analyze image" in command or "describe image" in command or "interpret image" in command:
        speak("Please provide the full path of the image you want me to analyze.")
        image_path = take_command()
        
        if image_path != "none" and os.path.exists(image_path):
            speak("Analyzing the image. Please wait...")
            result = analyze_image(image_path)
            if result:
                speak("Here is what I found in the image:")
                speak(result)
            else:
                speak("Sorry, I couldn't understand the image.")
        else:
            speak("Invalid file path or image not found.")

    elif "play game" in command or "start game" in command:
        speak("Starting the number guessing game.")
        play_game()

    elif "battery status" in command or "battery level" in command:
        battery_info = get_battery_info()
        if battery_info:
            if battery_info["power_plugged"]:
                speak(f"Your battery is at {battery_info['percent']}% and charging.")
            else:
                speak(f"Your battery is at {battery_info['percent']}%.")
        else:
            speak("I couldn't retrieve battery information.")
    
    elif "chat" or "convert chat mode" in take_command:
        mode = "chat"
        speak("Chat mode activated. Type your message below.")
    # Removed the dangling else block as it is not associated with any condition
            
    # For any other command, use AI to generate a response
    else:
        speak("Let me think about that...")
        response = generate_ai_response(command)
        speak(response)
    
    return True

from datetime import datetime

def run_shree():
    # Get current hour
    hour = datetime.now().hour

    # Time-based greeting
    if 5 <= hour < 12:
        greeting = "Good morning"
    elif 12 <= hour < 17:
        greeting = "Good afternoon"
    elif 17 <= hour < 21:
        greeting = "Good evening"
    else:
        greeting = "Good night"

    speak(f"{greeting}! I am Shree. How can I help you today?")

    running = True
    while running:
        command = take_command()
        if command:
            if "start virtual mouse" in command or "gesture control" in command:
                speak("Starting gesture-controlled mouse. Show your hand to the camera...")
                start_virtual_mouse()
            else:
                running = process_command(command)


# -------- Run Assistant --------
if __name__ == "__main__":
    run_shree()
