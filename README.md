# Automatic-Score-Calculator-for-the-Mathable-Game
Introduction The Mathable Board Game project automates game board analysis and score calculation using computer vision techniques and Python algorithms. Objective To develop software that identifies the board, game pieces, positions, and values, calculating scores according to Mathable rules.
# Mathable Automatic Score Calculator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org/)

<p align="center">
  <img src="docs/images/mathable_board.png" alt="Mathable Game Board" width="400"/>
</p>

## 📋 Overview
An intelligent computer vision system that automates score calculation for the Mathable board game. This project applies image processing techniques and Python algorithms to identify the game board, detect placed pieces, recognize their values, and calculate scores according to official game rules.

## ✨ Features

- **Automated Board Recognition**: Detects and isolates the Mathable game board from images
- **Piece Detection and Value Extraction**: Identifies newly placed pieces and their numerical values
- **Rule Enforcement**: Validates moves according to Mathable's mathematical operation rules
- **Score Calculation**: Automatically calculates scores including bonus squares
- **Game State Tracking**: Follows the game progression through multiple moves
- **Template Matching**: Uses custom OCR approach for accurate piece value recognition

## 🔧 Tech Stack

- **Python 3.6+**: Core programming language
- **OpenCV**: Image processing and computer vision operations
- **NumPy**: Numerical operations and matrix handling
- **Matplotlib**: Visualization of images and processing steps (optional)

## 🖼️ How It Works

### 1. Board Detection and Processing
The system uses a multi-step approach to extract and process the game board:

```python
# Apply Gaussian filter to reduce image noise
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# Apply thresholding to create binary image
_, thresh = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Find contours to detect the board
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
```

### 2. Template Creation and Matching
The system creates templates for number recognition:

<p align="center">
  <img src="docs/images/number_template.png" alt="Number Template Example" width="100"/>
</p>

### 3. Game Rules Implementation
The board is internally represented as a matrix where:
- `30`: Triple bonus squares
- `20`: Double bonus squares
- `11-14`: Mathematical operations (addition, subtraction, multiplication, division)
- `-1`: Empty cells

## 🚀 Installation

```bash
# Clone this repository
git clone https://github.com/yourusername/mathable-score-calculator.git

# Navigate to the project directory
cd mathable-score-calculator

# Install the required dependencies
pip install -r requirements.txt
```

## 💻 Usage

```bash
# Basic usage
python main.py --input_folder path/to/game/images

# Specify a game number (for multiple games)
python main.py --input_folder path/to/game/images --game_number 1
```

## 📊 Results
The system outputs the calculated scores for each player after processing all moves:

```
Game 1 Results:
Player 1: 76 points
Player 2: 89 points
```

## 🛠️ Main Functions

1. **verifica_bonus()**: Validates mathematical relationships between pieces
2. **verifica_constrangeri()**: Calculates scores including bonuses
3. **proceseaza_miscari()**: Processes game moves sequentially

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author
Tismanaru Artemis

---

<p align="center">Made with ❤️ for the love of board games and computer vision</p>
