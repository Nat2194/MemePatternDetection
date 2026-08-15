import os

def create_workspace():
    project_name = "MemePatternDetection"
    
    # 1. Define the folders we need
    folders = [
        project_name,
        os.path.join(project_name, "memes"),
        os.path.join(project_name, "utils")
    ]
    
    # 2. Define the files and their starter content
    files = {
        os.path.join(project_name, "main.py"): "# We will put the MediaPipe and OpenCV logic here!\n",
        os.path.join(project_name, "requirements.txt"): "opencv-python\nmediapipe\n",
        os.path.join(project_name, "README.md"): "# Meme Pattern Detection\nMachine learning pattern detector that matches hand movements with meme images.\n",
        os.path.join(project_name, "LICENSE"): "MIT License\n\n(You can add your full license details here later.)\n"
    }

    # 3. Build the folders
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Created folder: {folder}")

    # 4. Build the files
    for file_path, content in files.items():
        with open(file_path, "w") as file:
            file.write(content)
        print(f"Created file: {file_path}")

    print("\nSuccess! Your project workspace is ready.")

if __name__ == "__main__":
    create_workspace()