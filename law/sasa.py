import os
import time
import subprocess
import pyautogui
import pyperclip

# --- Configuration ---
# Replace with the path to the folder containing your PDFs
pdf_folder = r"C:\Users\hungn\Downloads\darkin"

# Replace with your actual Chrome path (This is the default Windows path)
# Mac users: chrome_path = "open -a 'Google Chrome'"
chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

# ---------------------

def process_pdfs():
    screen_width, screen_height = pyautogui.size()
    
    # Calculate coordinates for a drag in the center of the screen
    start_x, start_y = int(screen_width * 0.4), int(screen_height * 0.4)
    end_x, end_y = int(screen_width * 0.6), int(screen_height * 0.6)


    for filename in os.listdir(pdf_folder):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            txt_path = os.path.join(pdf_folder, filename.replace(".pdf", ".txt"))

            print(f"Processing: {filename}...")

            # Clear clipboard to avoid pasting old data if the copy fails
            pyperclip.copy('')

            # 1. Open the PDF in Chrome
            subprocess.Popen([chrome_path, pdf_path])
            time.sleep(5)
            # 3. THE TRIGGER: Mouse drag to wake up Chrome's OCR
            print("Holding and dragging mouse to activate OCR...")
            
            # Move to the starting position
            pyautogui.moveTo(start_x, start_y)
            time.sleep(0.5)
            
            # Press and HOLD the left mouse button
            pyautogui.mouseDown(button='left')
            time.sleep(0.5) # Pause to let Chrome register the click
            
            # Drag the mouse to the new coordinates while holding the button
            pyautogui.moveTo(end_x, end_y, duration=1.0)
            time.sleep(0.5) # Pause to let Chrome register the selection
            
            # Release the mouse button
            
            # Give the AI a moment to process the text it just discovered
            time.sleep(2)
            # 2. Wait for Chrome to load and the Screen AI OCR to finish processing.
            # You may need to increase this number if your PDFs are long or your PC is slow.
            time.sleep(25) 

            # 3. Simulate Ctrl + A (Select All)
            pyautogui.hotkey('ctrl', 'a')  # Mac users: change 'ctrl' to 'command'
            time.sleep(0.5)

            # 4. Simulate Ctrl + C (Copy)
            pyautogui.hotkey('ctrl', 'c')  # Mac users: change 'ctrl' to 'command'
            time.sleep(1)

            # 5. Grab the text from the clipboard
            extracted_text = pyperclip.paste()

            # 6. Save it to a text file
            if extracted_text.strip():
                with open(txt_path, "w", encoding="utf-8") as file:
                    file.write(extracted_text)
                print(f"Success! Saved to {txt_path}\n")
            else:
                print("Failed to copy text. You might need to increase the sleep time.\n")

            # 7. Close the Chrome tab (Ctrl + W)
            pyautogui.hotkey('ctrl', 'w')  # Mac users: change 'ctrl' to 'command'
            time.sleep(1)

if __name__ == "__main__":
    process_pdfs()