# Mathable Automatic Score Calculator

[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="https://github.com/user-attachments/assets/f275ff4c-f908-4224-aae4-de9a3cc70e9a" alt="Mathable Game Board" width="400"/>
</p> 


## 📋 Overview

An intelligent computer vision system that automatically calculates scores for the Mathable board game. The project applies image processing techniques to detect the game board, identify newly placed pieces, recognize their values, and calculate player scores according to the official Mathable rules.

## 🎲 About Mathable

Mathable is a mathematical board game similar to Scrabble but using numbers instead of letters:
- Players place numbered tiles on a 14×14 grid board to form valid mathematical equations
- The board contains regular squares, constraint squares with mathematical operators, and bonus squares
- Players score points based on the numerical value of placed pieces, with bonus multipliers for special squares
- Equations are formed by addition, subtraction, multiplication, or division of adjacent pieces

## ✨ Key Features

- **Automated Board Detection**: Extracts and processes the Mathable game board from images
- **Piece Position Identification**: Determines where new pieces have been placed on the board
- **Number Recognition**: Uses custom OCR techniques to identify the numerical values of placed pieces
- **Score Calculation**: Implements all Mathable scoring rules including:
  - Base scores (the value of the placed piece)
  - Double and triple point multipliers
  - Multiple equation bonuses
- **Game State Tracking**: Follows the game progression through multiple rounds

## 🛠️ Technical Implementation

### Board and Piece Detection

The system processes images using OpenCV to identify the game board and detect piece placements:

```python
# Apply Gaussian filter to reduce noise
blurred_image = cv2.GaussianBlur(blue, (5, 5), 0)

# Apply thresholding to create binary image
_, thresh = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Find contours to detect the board
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
```

### Template Matching for Number Recognition

The system creates and uses templates to recognize piece values:

```python
# Loop through templates for matching
for i in lista_templates:
    template = cv2.imread(f"./templates/{i}.jpg")
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template = cv2.GaussianBlur(template, (5, 5), 0)
    _, template = cv2.threshold(template, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    result = cv2.matchTemplate(roi_mega, template, cv2.TM_CCOEFF_NORMED)
    corr = np.max(result)
    # Find best match
    if i < 10:
        dict_cifre_similaritati[i] = corr
    else:
        dict_numere_similaritati[i] = corr
```

### Score Calculation

Three main functions handle the scoring logic:

1. **`verifica_bonus()`**: Determines if a piece placement satisfies mathematical constraints
2. **`verifica_constrangeri()`**: Calculates scores based on piece value and board position  
3. **`proceseaza_miscari()`**: Manages game flow and accumulates scores for each player

## 📊 Results

The system successfully:
- Detects piece positions with high accuracy
- Recognizes numeric values on the pieces
- Calculates scores according to Mathable rules

## 🚀 Installation and Usage

### Prerequisites
- Python 3.6 or higher
- OpenCV
- NumPy
- Additional dependencies listed in requirements.txt

### Setup

```bash
# Clone this repository
git clone https://github.com/yourusername/mathable-score-calculator.git

# Navigate to the project directory
cd mathable-score-calculator

# Install required dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
# Run the full pipeline
python main.py --input_folder=path/to/images --output_folder=results

# Run individual tasks
python main.py --task=1  # Piece position detection
python main.py --task=2  # Number recognition
python main.py --task=3  # Score calculation
```

## 📁 Project Structure

```
project/
├── main.py                   # Main execution script
├── board_detection.py        # Functions for detecting the game board
├── piece_recognition.py      # Functions for recognizing pieces and values
├── score_calculation.py      # Logic for calculating player scores
├── templates/                # Number templates for matching
├── docs/                     # Documentation and examples
│   └── images/               # Example images
├── results/                  # Output folder
└── requirements.txt          # Required dependencies
```

## 👩‍💻 Author

**Tismanaru Artemis**  
Computer Vision and Artificial Intelligence  
University of Bucharest

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Made with ❤️ for computer vision and board games
</p>
